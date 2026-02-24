#!/usr/bin/env python3
"""Interactive SAM 2.1 segmentation tool.

Left-click to add positive points, right-click for negative points.
The mask updates live after each click.

Controls:
    Left-click   — Add positive (foreground) point
    Right-click  — Add negative (background) point
    u            — Undo last point
    r            — Reset all points
    s            — Save mask + overlay and exit
    q / Escape   — Quit without saving
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive SAM 2.1 segmentation")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: same as image)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/sam2.1-hiera-large",
        help="HuggingFace model ID (default: facebook/sam2.1-hiera-large)",
    )
    return parser.parse_args()


class SegmentationUI:
    def __init__(self, image: np.ndarray, predictor: SAM2ImagePredictor, save_stem: str, output_dir: Path):
        self.image = image
        self.predictor = predictor
        self.save_stem = save_stem
        self.output_dir = output_dir

        # Point storage: list of (x, y) coords and corresponding labels
        self.points: list[tuple[float, float]] = []
        self.labels: list[int] = []  # 1 = positive, 0 = negative

        # Mask state
        self.current_mask: np.ndarray | None = None
        self.current_logits: np.ndarray | None = None

        # Set up matplotlib figure
        self.fig, self.ax = plt.subplots(1, 1, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("quick_sam — Interactive Segmentation")
        self.ax.set_axis_off()

        # Display layers
        self.img_display = self.ax.imshow(self.image)
        self.mask_overlay = self.ax.imshow(
            np.zeros((*self.image.shape[:2], 4), dtype=np.uint8),
            alpha=1.0,
        )
        self.point_scatter_pos = self.ax.scatter([], [], c="lime", s=60, edgecolors="white", linewidths=1.5, zorder=5)
        self.point_scatter_neg = self.ax.scatter([], [], c="red", s=60, edgecolors="white", linewidths=1.5, zorder=5, marker="x")

        self._update_title()

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _update_title(self):
        n_pos = sum(1 for l in self.labels if l == 1)
        n_neg = sum(1 for l in self.labels if l == 0)
        self.ax.set_title(
            f"Points: {n_pos} positive, {n_neg} negative  |  "
            "L-click=add  R-click=negative  u=undo  r=reset  s=save  q=quit",
            fontsize=10,
        )

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        # Ignore clicks from toolbar interactions
        if self.fig.canvas.toolbar and self.fig.canvas.toolbar.mode:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.button == 1:  # Left click — positive
            self.points.append((x, y))
            self.labels.append(1)
        elif event.button == 3:  # Right click — negative
            self.points.append((x, y))
            self.labels.append(0)
        else:
            return

        self._run_prediction()
        self._refresh_display()

    def _on_key(self, event):
        if event.key == "u":
            if self.points:
                self.points.pop()
                self.labels.pop()
                if self.points:
                    self._run_prediction()
                else:
                    self.current_mask = None
                    self.current_logits = None
                self._refresh_display()
        elif event.key == "r":
            self.points.clear()
            self.labels.clear()
            self.current_mask = None
            self.current_logits = None
            self._refresh_display()
        elif event.key == "s":
            self._save_outputs()
            plt.close(self.fig)
        elif event.key in ("q", "escape"):
            print("Exiting without saving.")
            plt.close(self.fig)

    def _run_prediction(self):
        if not self.points:
            return

        point_coords = np.array(self.points, dtype=np.float32)
        point_labels = np.array(self.labels, dtype=np.int32)

        # For a single point, use multimask and pick best.
        # For multiple points, use single mask with logit refinement.
        use_multimask = len(self.points) == 1

        predict_kwargs = dict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=use_multimask,
        )

        # Feed back previous logits for refinement when we have 2+ points
        if self.current_logits is not None and not use_multimask:
            predict_kwargs["mask_input"] = self.current_logits[None, :, :]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(**predict_kwargs)

        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        # Ensure mask is 2D boolean (squeeze any extra dims from SAM output)
        while mask.ndim > 2:
            mask = mask[0]
        self.current_mask = mask.astype(bool)
        logit = logits[best_idx]
        while logit.ndim > 2:
            logit = logit[0]
        self.current_logits = logit

    def _refresh_display(self):
        # Update mask overlay
        if self.current_mask is not None:
            overlay = np.zeros((*self.image.shape[:2], 4), dtype=np.uint8)
            # Semi-transparent blue mask
            overlay[self.current_mask, 0] = 30   # R
            overlay[self.current_mask, 1] = 144  # G
            overlay[self.current_mask, 2] = 255  # B
            overlay[self.current_mask, 3] = 100  # Alpha
        else:
            overlay = np.zeros((*self.image.shape[:2], 4), dtype=np.uint8)
        self.mask_overlay.set_data(overlay)

        # Update point markers
        pos_pts = [(x, y) for (x, y), l in zip(self.points, self.labels) if l == 1]
        neg_pts = [(x, y) for (x, y), l in zip(self.points, self.labels) if l == 0]

        if pos_pts:
            self.point_scatter_pos.set_offsets(np.array(pos_pts))
        else:
            self.point_scatter_pos.set_offsets(np.empty((0, 2)))

        if neg_pts:
            self.point_scatter_neg.set_offsets(np.array(neg_pts))
        else:
            self.point_scatter_neg.set_offsets(np.empty((0, 2)))

        self._update_title()
        self.fig.canvas.draw_idle()

    def _save_outputs(self):
        if self.current_mask is None:
            print("No mask to save — add points first.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save binary mask
        mask_path = self.output_dir / f"{self.save_stem}_mask.png"
        mask_img = Image.fromarray((self.current_mask * 255).astype(np.uint8), mode="L")
        mask_img.save(mask_path)

        # Save overlay
        overlay_path = self.output_dir / f"{self.save_stem}_overlay.png"
        overlay = self.image.copy()
        # Apply blue tint to masked region
        mask_color = np.array([30, 144, 255], dtype=np.uint8)
        alpha = 0.4
        overlay[self.current_mask] = (
            overlay[self.current_mask] * (1 - alpha) + mask_color * alpha
        ).astype(np.uint8)

        # Draw points on overlay
        overlay_pil = Image.fromarray(overlay)
        # Use matplotlib to draw points on the overlay for cleaner output
        fig_save, ax_save = plt.subplots(1, 1, figsize=(overlay.shape[1] / 100, overlay.shape[0] / 100), dpi=100)
        ax_save.imshow(overlay)
        ax_save.set_axis_off()

        pos_pts = [(x, y) for (x, y), l in zip(self.points, self.labels) if l == 1]
        neg_pts = [(x, y) for (x, y), l in zip(self.points, self.labels) if l == 0]
        if pos_pts:
            pts = np.array(pos_pts)
            ax_save.scatter(pts[:, 0], pts[:, 1], c="lime", s=40, edgecolors="white", linewidths=1, zorder=5)
        if neg_pts:
            pts = np.array(neg_pts)
            ax_save.scatter(pts[:, 0], pts[:, 1], c="red", s=40, edgecolors="white", linewidths=1, zorder=5, marker="x")

        fig_save.savefig(overlay_path, bbox_inches="tight", pad_inches=0, dpi=150)
        plt.close(fig_save)

        print(f"Saved mask:    {mask_path}")
        print(f"Saved overlay: {overlay_path}")

    def run(self):
        plt.show()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir = Path(args.output_dir) if args.output_dir else image_path.parent
    save_stem = image_path.stem

    print(f"Loading image: {image_path}")
    image = np.array(Image.open(image_path).convert("RGB"))

    print(f"Loading SAM 2.1 model: {args.model}")
    predictor = SAM2ImagePredictor.from_pretrained(args.model)

    print("Computing image embeddings...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)

    print("Ready! Click on the image to segment.")
    ui = SegmentationUI(image, predictor, save_stem, output_dir)
    ui.run()


if __name__ == "__main__":
    main()
