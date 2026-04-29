# quick_sam

Interactive image segmentation using SAM 2.1, served as a browser UI via [viser](https://viser.studio). Click to add positive/negative points, see the mask update live, save results, and optionally lift the mask to a 3D mesh for use in MuJoCo.

## Setup

```bash
cd quick_sam
uv sync
```

## Usage

```bash
uv run python segment.py --image-path <PATH> [--output-dir DIR] [--model MODEL_ID] [--port PORT]
```

Opens a viser server at `http://localhost:8080` (or `--port` value). If running over SSH, forward the port first:

```bash
ssh -L 8080:localhost:8080 <host>
```

**Examples:**

```bash
# Basic usage
uv run python segment.py --image-path photo.jpg

# Custom output directory and port
uv run python segment.py --image-path photo.jpg --output-dir ./masks --port 8081

# Smaller/faster model
uv run python segment.py --image-path photo.jpg --model facebook/sam2.1-hiera-small
```

## Controls

All controls are in the browser sidebar.

| Action | How |
|--------|-----|
| Add positive (foreground) point | Click image |
| Add negative (background) point | Check **Negative mode**, then click |
| Undo last point | Sidebar → **Undo** |
| Reset all points | Sidebar → **Reset** |
| Save mask + overlay and exit | Sidebar → **Save** |
| Lift mask to 3D mesh + calibrate scale | Check **Mesh output (for MuJoCo)**, then sidebar → **Lift to 3D** |
| Lift mask to Gaussian splat and exit | Uncheck **Mesh output**, then sidebar → **Lift to 3D** |
| Quit without saving | Sidebar → **Quit** |

## Output

**Save** writes two files next to the input image (or in `--output-dir`):

- `<stem>_mask.png` — binary mask (white = foreground)
- `<stem>_overlay.png` — image with mask overlay and point markers

**Lift to 3D (Gaussian splat)** additionally produces:

- `<stem>_3d.ply` — Gaussian splat; open in [SuperSplat](https://playcanvas.com/super-splat) or Polycam

**Lift to 3D (Mesh output checked)** produces:

- `<stem>_3d.glb` — triangle mesh with vertex colours
- `<stem>_3d.obj` — same mesh in OBJ format, ready for MuJoCo
- `<stem>_3d.json` — metric size estimate from MoGe depth (and corrected value after calibration)

## Scale calibration (3D mesh mode)

After the mesh lift completes, the segmentation UI is replaced by a 3D viewer showing the reconstructed mesh and its **PCA longest axis** highlighted in orange. Use this to calibrate the metric scale:

1. Orbit the mesh in the browser to orient the orange axis
2. Measure that same dimension on the real physical object (in metres)
3. Type the measurement into **Real length (m)** in the sidebar
4. Click **Confirm & Exit** — saves `corrected_longest_axis_m` to the JSON sidecar

Skip calibration (keep MoGe's depth estimate) by clicking **Skip**.

## MuJoCo export

After lifting with **Mesh output** checked, convert to a MuJoCo-ready OBJ + MJCF template:

```bash
uv run python to_mujoco.py <stem>_3d.glb
```

If the JSON sidecar exists (it always will after a mesh lift), the scale is read automatically. Override with `--scale`:

```bash
uv run python to_mujoco.py photo_3d.glb --scale 0.22   # object is 22 cm on longest axis
```

Outputs (in `<stem>/` next to the GLB):

- `<stem>.obj` — mesh normalised so longest axis = 1 unit
- `<stem>.xml` — MJCF template with `scale=` set to the calibrated metric size

```bash
# Quick viewer check
python -m mujoco.viewer --mjcf photo/photo.xml

# Include in an existing scene
<include file="photo/photo.xml"/>
```

## SAM 3D Objects setup (mesh lift)

The mesh lift shells out to a separate Python environment because the dep stacks are incompatible (SAM 2.1 here vs. SAM 3D's custom CUDA wheels). `segment.py` hardcodes:

```python
SAM3D_PYTHON      = "/home/eyeballisticmissile/miniforge3/envs/sam3d-objects/bin/python"
SAM3D_LIFT_SCRIPT = "/home/eyeballisticmissile/sam-3d-objects/lift_3d.py"
```

Update those constants if your paths differ.

### One-time setup

1. Accept the license at `https://huggingface.co/facebook/sam-3d-objects` and run `hf auth login` in the `sam3d-objects` env
2. Download model weights (≈12 GB) — happens automatically on first lift
3. Install `nvdiffrast` (required for mesh postprocessing):

```bash
/path/to/envs/sam3d-objects/bin/pip install git+https://github.com/NVlabs/nvdiffrast --no-build-isolation
```

**Hardware:** NVIDIA GPU only, ≥32 GB VRAM (peaks at ~27 GiB). Wall-clock ~60 s per lift.

**Blackwell (sm_120 / RTX 5090):** the env uses torch 2.8.0+cu128, kaolin 0.18.0 from NVIDIA's `torch-2.8.0_cu128` wheel index, pytorch3d source-built targeting `compute_120,sm_120`, and spconv-cu126 (CUDA forward-compat).
