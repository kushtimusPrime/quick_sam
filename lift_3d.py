#!/usr/bin/env python
"""Run SAM 3D Objects on (image, mask) and save 3D output.

Usage:
    python lift_3d.py --image PATH --mask PATH --out STEM [--seed N] [--config PATH] [--mesh]

Picks .glb if the pipeline produced one (requires --mesh), otherwise saves a
Gaussian splat .ply.  Prints the actual saved path on the last line of stdout.

Invoked as a subprocess by segment.py because the SAM 2.1 and SAM 3D dep
stacks are incompatible and must run in separate Python environments.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Path to the SAM 3D Objects repo clone.  Update if yours lives elsewhere.
SAM3D_REPO = Path("/home/eyeballisticmissile/sam-3d-objects")

# inference.py does `os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]` at
# import time.  When called as a subprocess without activating the env,
# CONDA_PREFIX may be unset or point at the wrong env.  Force it to this
# interpreter's env root so CUDA_HOME resolves correctly.
os.environ["CONDA_PREFIX"] = str(Path(sys.executable).resolve().parent.parent)

sys.path.insert(0, str(SAM3D_REPO / "notebook"))
sys.path.insert(0, str(SAM3D_REPO))
import numpy as np
from PIL import Image
from inference import Inference  # noqa: E402


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., -1]
    return arr > 0


def _save_3d(output: dict, out_stem: Path) -> Path:
    glb = output.get("glb")
    if glb is not None:
        target = out_stem.with_suffix(".glb")
        if isinstance(glb, (bytes, bytearray)):
            target.write_bytes(glb)
        elif isinstance(glb, (str, Path)) and Path(glb).exists():
            shutil.copy(glb, target)
        elif hasattr(glb, "export"):
            glb.export(str(target))
        else:
            glb = None
        if glb is not None:
            return target

    target = out_stem.with_suffix(".ply")
    output["gs"].save_ply(str(target))
    return target


def main():
    parser = argparse.ArgumentParser(description="Lift (image, mask) to a 3D file via SAM 3D Objects")
    parser.add_argument("--image", required=True, type=Path, help="RGB image path")
    parser.add_argument("--mask", required=True, type=Path, help="Binary mask path (white=foreground)")
    parser.add_argument("--out", required=True, type=Path, help="Output path; extension replaced with .glb or .ply")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=SAM3D_REPO / "checkpoints" / "hf" / "pipeline.yaml")
    parser.add_argument("--mesh", action="store_true", help="Extract triangle mesh (.glb) instead of Gaussian splat (.ply)")
    args = parser.parse_args()

    image = _load_rgb(args.image)
    mask = _load_mask(args.mask)
    if image.shape[:2] != mask.shape:
        raise SystemExit(f"image/mask shape mismatch: {image.shape[:2]} vs {mask.shape}")
    if not mask.any():
        raise SystemExit("mask is empty — refusing to run inference")

    inference = Inference(str(args.config), compile=False)
    if args.mesh:
        rgba = inference.merge_mask_to_rgba(image, mask)
        output = inference._pipeline.run(
            rgba, None, seed=args.seed,
            with_mesh_postprocess=True,
            with_texture_baking=False,
            use_vertex_color=True,
        )
    else:
        output = inference(image, mask, seed=args.seed)

    saved = _save_3d(output, args.out.with_suffix(""))

    # Compute metric size from MoGe's pointmap at the masked pixels.
    # pointmap is HxWx3 in camera-space metres (before SSI normalisation).
    pointmap = output.get("pointmap")
    if pointmap is not None:
        pm = pointmap.cpu().float().numpy() if hasattr(pointmap, "cpu") else np.array(pointmap, dtype=np.float32)
        pm_h, pm_w = pm.shape[:2]
        if mask.shape != (pm_h, pm_w):
            mask_resized = np.array(
                Image.fromarray(mask.astype(np.uint8) * 255).resize((pm_w, pm_h), Image.NEAREST)
            ) > 0
        else:
            mask_resized = mask
        pts = pm[mask_resized]  # Nx3
        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if len(pts) > 0:
            extents = pts.max(axis=0) - pts.min(axis=0)
            longest_axis_m = float(extents.max())
            json_path = saved.with_suffix(".json")
            json_path.write_text(json.dumps({
                "metric_extents_m": extents.tolist(),
                "longest_axis_m": longest_axis_m,
            }, indent=2))
            print(f"Metric size (MoGe): {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f} m  "
                  f"(longest: {longest_axis_m:.3f} m)")

    print(saved)


if __name__ == "__main__":
    main()
