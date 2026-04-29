#!/usr/bin/env python3
"""Convert a SAM 3D Objects .glb to MuJoCo-ready OBJ + MJCF template.

Usage:
    uv run python to_mujoco.py model_3d.glb
    uv run python to_mujoco.py model_3d.glb --scale 0.12   # override to 12cm

Output (next to the .glb):
    <stem>/
      <stem>.obj      — triangle mesh (longest axis normalized to 1m, then scaled)
      <stem>.xml      — MJCF template (freejoint body, ready to include or simulate)

Scale:
    If lift_3d.py ran with --mesh, it saves a <stem>_3d.json sidecar with MoGe's
    metric estimate of the object size. to_mujoco.py reads that automatically.
    You can override with --scale <meters> (longest axis of the real object).
"""

import argparse
import json
import textwrap
from pathlib import Path

import trimesh


def load_as_single_mesh(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(glb_path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        meshes = list(loaded.geometry.values())
        if not meshes:
            raise ValueError(f"No meshes found in {glb_path}")
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    else:
        mesh = loaded
    return mesh


def main() -> None:
    parser = argparse.ArgumentParser(description="GLB → MuJoCo OBJ + MJCF")
    parser.add_argument("glb", type=Path, help=".glb file from SAM 3D (requires --mesh flag at lift time)")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override: real-world size of longest axis in meters. Auto-read from sidecar .json if omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <glb_stem>/ next to the GLB)",
    )
    args = parser.parse_args()

    if not args.glb.exists():
        raise FileNotFoundError(args.glb)

    stem = args.glb.stem.replace("_3d", "")
    out_dir = args.out_dir or (args.glb.parent / stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve metric scale: --scale flag > sidecar JSON > warn
    scale_m = args.scale
    scale_source = "--scale flag"
    if scale_m is None:
        json_path = args.glb.with_suffix(".json")
        if json_path.exists():
            info = json.loads(json_path.read_text())
            scale_m = info.get("corrected_longest_axis_m") or info["longest_axis_m"]
            extents = info.get("metric_extents_m")
            scale_source = (
                f"calibrated measurement ({json_path.name})"
                if "corrected_longest_axis_m" in info
                else f"MoGe depth estimate ({json_path.name})"
            )
            if extents:
                print(f"MoGe metric size: {extents[0]:.3f} x {extents[1]:.3f} x {extents[2]:.3f} m")
        else:
            print(
                "No scale sidecar found and --scale not given. "
                "Mesh will be 1m on longest axis. Re-run with --scale <meters> to correct."
            )
            scale_m = 1.0
            scale_source = "default (1m — no sidecar)"

    print(f"Scale: {scale_m:.4f}m (source: {scale_source})")

    print(f"Loading {args.glb}...")
    mesh = load_as_single_mesh(args.glb)
    print(f"  {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Normalize mesh so longest axis = 1 unit, then MJCF scale= applies metric size
    mesh.vertices -= mesh.vertices.mean(axis=0)
    extent = mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0)
    mesh.vertices /= extent.max()

    obj_path = out_dir / f"{stem}.obj"
    mesh.export(str(obj_path))
    print(f"Saved OBJ:  {obj_path}")

    scale_str = f"{scale_m} {scale_m} {scale_m}"
    mjcf_path = out_dir / f"{stem}.xml"
    mjcf = textwrap.dedent(f"""\
        <mujoco model="{stem}">
          <compiler meshdir="{out_dir.name}/"/>

          <asset>
            <!-- longest axis = {scale_m:.4f}m  (source: {scale_source}) -->
            <mesh name="{stem}" file="{stem}.obj" scale="{scale_str}"/>
          </asset>

          <worldbody>
            <!--
              Drop this <body> into your scene.
              Remove <freejoint/> to make it a static object.
              Add mass/inertia for dynamics, or let MuJoCo infer from geometry.
            -->
            <body name="{stem}" pos="0 0 0">
              <freejoint/>
              <geom name="{stem}" type="mesh" mesh="{stem}"
                    condim="4" friction="0.8 0.01 0.01" rgba="0.8 0.7 0.6 1"/>
            </body>
          </worldbody>
        </mujoco>
        """)
    mjcf_path.write_text(mjcf)
    print(f"Saved MJCF: {mjcf_path}")
    print(f"\n  Viewer:  python -m mujoco.viewer --mjcf {mjcf_path}")
    print(f"  Include: <include file=\"{mjcf_path}\"/>")


if __name__ == "__main__":
    main()
