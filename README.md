# quick_sam

Interactive image segmentation using SAM 2.1. Click to add positive/negative points, see the mask update live, and save the result.

## Setup

```bash
cd quick_sam
uv sync
```

## Usage

```bash
uv run python segment.py <image_path> [--output-dir DIR] [--model MODEL_ID]
```

**Examples:**

```bash
# Basic usage
uv run python segment.py photo.jpg

# Custom output directory
uv run python segment.py photo.jpg --output-dir ./masks

# Use a smaller/faster model
uv run python segment.py photo.jpg --model facebook/sam2.1-hiera-small
```

## Controls

| Input | Action |
|-------|--------|
| Left-click | Add positive (foreground) point |
| Right-click | Add negative (background) point |
| `u` | Undo last point |
| `r` | Reset all points |
| `s` | Save mask + overlay and exit |
| `q` / `Esc` | Quit without saving |

## Output

Saving produces two files in the output directory:

- `<image_name>_mask.png` — Binary mask (white = foreground, black = background)
- `<image_name>_overlay.png` — Original image with mask overlay and point markers
