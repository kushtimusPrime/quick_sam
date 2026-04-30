#!/usr/bin/env python3
"""Interactive SAM 2.1 segmentation tool.

Open http://localhost:<port> in your browser after launch.
Click the image to add points; use the sidebar to switch modes and save.

Sidebar controls:
    Negative mode checkbox  — toggle positive/negative point type
    Undo                    — remove last point
    Reset                   — clear all points
    Save                    — save mask + overlay and exit
    Lift to 3D              — run SAM 3D Objects, save all, exit
    Quit                    — exit without saving
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import trimesh

import numpy as np
import torch
import viser
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM3D_PYTHON = "/home/eyeballisticmissile/miniforge3/envs/sam3d-objects/bin/python"
SAM3D_LIFT_SCRIPT = Path(__file__).parent / "lift_3d.py"
TO_MUJOCO_SCRIPT = Path(__file__).parent / "to_mujoco.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive SAM 2.1 segmentation")
    parser.add_argument("--image-path", type=str, required=True, help="Path to input image")
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
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    return parser.parse_args()


def _draw_circle(arr: np.ndarray, cx: int, cy: int, radius: int, color: tuple) -> None:
    """Fill a circle on an HxWx3 uint8 array in-place."""
    H, W = arr.shape[:2]
    y0, y1 = max(0, cy - radius), min(H, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(W, cx + radius + 1)
    if y0 >= y1 or x0 >= x1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
    arr[ys[mask], xs[mask]] = color


class SegmentationUI:
    def __init__(
        self,
        image: np.ndarray,
        predictor: SAM2ImagePredictor,
        save_stem: str,
        output_dir: Path,
        port: int = 8080,
    ):
        self.image = image  # HxWx3 uint8
        self.predictor = predictor
        self.save_stem = save_stem
        self.output_dir = output_dir

        self.points: list[tuple[float, float]] = []
        self.labels: list[int] = []
        self.current_mask: np.ndarray | None = None
        self.current_logits: np.ndarray | None = None
        self._done = False
        self._calibration_active = False

        self._H, self._W = image.shape[:2]

        self.server = viser.ViserServer(port=port, verbose=False)

        # Sidebar — store folder handles so calibration mode can remove them
        self._header_md = self.server.gui.add_markdown("## quick_sam")
        self._point_mode_folder = self.server.gui.add_folder("Point mode")
        with self._point_mode_folder:
            self.neg_mode = self.server.gui.add_checkbox(
                "Negative (background) mode", initial_value=False
            )
        self._actions_folder = self.server.gui.add_folder("Actions")
        with self._actions_folder:
            self._status = self.server.gui.add_markdown("0 positive · 0 negative")
            undo_btn = self.server.gui.add_button("Undo")
            reset_btn = self.server.gui.add_button("Reset")
            save_btn = self.server.gui.add_button("Save")
            self.mesh_output = self.server.gui.add_checkbox(
                "Mesh output (for MuJoCo)", initial_value=False
            )
            lift_btn = self.server.gui.add_button("Lift to 3D")
            quit_btn = self.server.gui.add_button("Quit (no save)")

        # Click handler — screen_pos is normalized [0,1], top-left origin
        @self.server.scene.on_pointer_event("click")
        def _on_click(event: viser.ScenePointerEvent) -> None:
            if self._calibration_active:
                return
            sx, sy = event.screen_pos[0]
            self.points.append((sx * self._W, sy * self._H))
            self.labels.append(0 if self.neg_mode.value else 1)
            self._run_prediction()
            self._refresh_display()

        @undo_btn.on_click
        def _undo(_) -> None:
            if self.points:
                self.points.pop()
                self.labels.pop()
                if self.points:
                    self._run_prediction()
                else:
                    self.current_mask = None
                    self.current_logits = None
                self._refresh_display()

        @reset_btn.on_click
        def _reset(_) -> None:
            self.points.clear()
            self.labels.clear()
            self.current_mask = None
            self.current_logits = None
            self._refresh_display()

        @save_btn.on_click
        def _save(_) -> None:
            self._save_outputs()
            self._done = True

        @lift_btn.on_click
        def _lift(_) -> None:
            result = self._lift_to_3d()
            if result is True:
                self._done = True

        @quit_btn.on_click
        def _quit(_) -> None:
            print("Exiting without saving.")
            self._done = True

        self._refresh_display()

    def _run_prediction(self) -> None:
        if not self.points:
            return

        point_coords = np.array(self.points, dtype=np.float32)
        point_labels = np.array(self.labels, dtype=np.int32)
        use_multimask = len(self.points) == 1

        predict_kwargs: dict = dict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=use_multimask,
        )
        if self.current_logits is not None and not use_multimask:
            predict_kwargs["mask_input"] = self.current_logits[None, :, :]

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(**predict_kwargs)

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        while mask.ndim > 2:
            mask = mask[0]
        self.current_mask = mask.astype(bool)
        logit = logits[best_idx]
        while logit.ndim > 2:
            logit = logit[0]
        self.current_logits = logit

    def _refresh_display(self) -> None:
        display = self.image.copy()

        if self.current_mask is not None:
            mask_color = np.array([30, 144, 255], dtype=np.uint8)
            display[self.current_mask] = (
                display[self.current_mask] * 0.55 + mask_color * 0.45
            ).astype(np.uint8)

        for (px, py), label in zip(self.points, self.labels):
            color = (0, 230, 0) if label == 1 else (220, 0, 0)
            _draw_circle(display, int(px), int(py), 11, (255, 255, 255))
            _draw_circle(display, int(px), int(py), 9, color)

        self.server.scene.set_background_image(display, format="jpeg")

        n_pos = sum(1 for l in self.labels if l == 1)
        n_neg = sum(1 for l in self.labels if l == 0)
        self._status.content = f"**{n_pos}** positive · **{n_neg}** negative"

    def _save_outputs(self) -> None:
        if self.current_mask is None:
            print("No mask to save — add points first.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        mask_path = self.output_dir / f"{self.save_stem}_mask.png"
        Image.fromarray((self.current_mask * 255).astype(np.uint8), mode="L").save(mask_path)

        overlay_path = self.output_dir / f"{self.save_stem}_overlay.png"
        overlay = self.image.copy()
        mask_color = np.array([30, 144, 255], dtype=np.uint8)
        overlay[self.current_mask] = (
            overlay[self.current_mask] * 0.6 + mask_color * 0.4
        ).astype(np.uint8)
        for (px, py), label in zip(self.points, self.labels):
            color = (0, 230, 0) if label == 1 else (220, 0, 0)
            _draw_circle(overlay, int(px), int(py), 11, (255, 255, 255))
            _draw_circle(overlay, int(px), int(py), 9, color)
        Image.fromarray(overlay).save(overlay_path)

        print(f"Saved mask:    {mask_path}")
        print(f"Saved overlay: {overlay_path}")

    def _lift_to_3d(self) -> bool:
        if self.current_mask is None:
            print("No mask to lift — add points first.")
            return False

        self._save_outputs()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "image.png"
            mask_path = tmp / "mask.png"
            out_stem = tmp / "lift_out"

            Image.fromarray(self.image).save(image_path)
            Image.fromarray(
                (self.current_mask * 255).astype(np.uint8), mode="L"
            ).save(mask_path)

            cmd = [
                SAM3D_PYTHON,
                str(SAM3D_LIFT_SCRIPT),
                "--image", str(image_path),
                "--mask", str(mask_path),
                "--out", str(out_stem),
            ]
            if self.mesh_output.value:
                cmd.append("--mesh")
            print(f"Lifting to 3D (~60s)...\n  $ {' '.join(cmd)}")
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            except FileNotFoundError as e:
                print(f"SAM 3D env not found: {e}\nExpected: {SAM3D_PYTHON}")
                return False
            except subprocess.CalledProcessError as e:
                print(f"lift_3d.py failed (exit {e.returncode}).\nstderr:\n{e.stderr}")
                return False

            saved_path = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            saved = Path(saved_path)
            if not saved.exists():
                print(
                    f"lift_3d.py exited 0 but output not found at {saved_path!r}.\n"
                    f"stdout:\n{proc.stdout}"
                )
                return False

            dest = self.output_dir / f"{self.save_stem}_3d{saved.suffix}"
            shutil.copy(saved, dest)
            print(f"Saved 3D:      {dest}")

            # Copy JSON sidecar (metric size info) before the tempdir closes
            json_src = saved.with_suffix(".json")
            json_dest = dest.with_suffix(".json")
            if json_src.exists():
                shutil.copy(json_src, json_dest)

            if dest.suffix == ".glb":
                obj_path = dest.with_suffix(".obj")
                _glb_to_obj(dest, obj_path)
                if self.mesh_output.value:
                    # Stay open for scale calibration; caller must not set _done
                    self._enter_calibration_mode(dest, json_dest)
                    return None

            return True

    def _enter_calibration_mode(self, glb_path: Path, json_path: Path) -> None:
        """Show the 3D mesh in viser with its PCA longest axis highlighted.

        The user orbits the mesh, measures the orange axis on the physical object,
        and types the real length into the number input.  Confirm writes
        corrected_longest_axis_m back into the JSON sidecar.
        """
        self._calibration_active = True

        # Wipe all segmentation UI and release pointer capture so orbit works
        self._header_md.remove()
        self._point_mode_folder.remove()
        self._actions_folder.remove()
        self.server.scene.remove_pointer_callback()
        self.server.scene.set_background_image(None)

        mesh = _load_glb_mesh(glb_path)
        mesh.vertices -= mesh.vertices.mean(axis=0)
        mesh.vertices /= (mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)).max()

        # PCA: first right singular vector = direction of greatest variance
        verts = mesh.vertices.astype(np.float32)
        _, _, Vt = np.linalg.svd(verts, full_matrices=False)
        axis = Vt[0]
        projs = verts @ axis
        lo = float(np.percentile(projs, 5))
        hi = float(np.percentile(projs, 95))
        center = verts.mean(axis=0)
        p1 = (center + axis * lo).astype(np.float32)
        p2 = (center + axis * hi).astype(np.float32)

        moge_m = 0.1
        if json_path.exists():
            info = json.loads(json_path.read_text())
            moge_m = float(info.get("longest_axis_m", 0.1))

        self.server.scene.add_mesh_trimesh("/calib/mesh", mesh)
        self.server.scene.add_line_segments(
            "/calib/axis",
            points=np.array([[p1, p2]]),
            colors=(255, 140, 0),
            line_width=4.0,
        )
        self.server.scene.add_point_cloud(
            "/calib/endpoints",
            points=np.array([p1, p2]),
            colors=np.array([[255, 140, 0], [255, 140, 0]], dtype=np.uint8),
            point_size=0.03,
        )

        with self.server.gui.add_folder("Scale Calibration"):
            self.server.gui.add_markdown(
                f"Rotate the mesh to see the **orange axis**.\n\n"
                f"MoGe depth estimate: **{moge_m:.3f} m**\n\n"
                "Measure that axis on the real object and enter below:"
            )
            real_input = self.server.gui.add_number(
                "Real length (m)", initial_value=round(moge_m, 4), min=0.001, step=0.001
            )
            confirm_btn = self.server.gui.add_button("Confirm & Exit")
            skip_btn = self.server.gui.add_button("Skip (use MoGe estimate)")

        @confirm_btn.on_click
        def _confirm(_) -> None:
            real_m = float(real_input.value)
            if json_path.exists():
                data = json.loads(json_path.read_text())
                data["corrected_longest_axis_m"] = real_m
                if moge_m > 0:
                    data["scale_correction_factor"] = real_m / moge_m
                json_path.write_text(json.dumps(data, indent=2))
                print(f"Corrected scale: {real_m:.4f} m → {json_path}")
            self._run_to_mujoco(glb_path)
            self._done = True

        @skip_btn.on_click
        def _skip(_) -> None:
            print("Using MoGe estimate as-is.")
            self._run_to_mujoco(glb_path)
            self._done = True

    def _run_to_mujoco(self, glb_path: Path) -> None:
        cmd = [sys.executable, str(TO_MUJOCO_SCRIPT), str(glb_path)]
        print(f"Building MuJoCo assets...\n  $ {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"to_mujoco.py failed (exit {e.returncode}). Run it manually on {glb_path}.")

    def run(self) -> None:
        try:
            while not self._done:
                time.sleep(0.25)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.server.stop()


def _load_glb_mesh(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(glb_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = loaded
    return mesh


def _glb_to_obj(glb_path: Path, obj_path: Path) -> None:
    mesh = _load_glb_mesh(glb_path)
    mesh.export(str(obj_path))
    print(f"Saved OBJ:     {obj_path}")


def main() -> None:
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

    print("Ready!")
    ui = SegmentationUI(image, predictor, save_stem, output_dir, port=args.port)
    print(f"\nOpen in browser: http://localhost:{args.port}")
    print("Default: positive points. Check 'Negative mode' for background points.\n")
    ui.run()


if __name__ == "__main__":
    main()
