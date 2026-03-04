# stroboscopic_image_generator

A tool that tracks moving objects in a video with SAM2 and outputs a single stroboscopic composite image

## Demo

<video src="./demo.mp4" controls width="720"></video>

## Requirements

- `uv` installed

Install `uv`:
https://docs.astral.sh/uv/getting-started/installation/

## Setup

1. Set up the `uv` environment:

```bash
uv sync
```

2. Place your input video in the `video` directory.

## Example Command

```bash
uv run python scripts/stroboscopic_image_generator.py \
  --video video/example.MP4 \
  --out video/stroboscopic_example.png \
  --process-fps 15 \
  --interval-sec 1.5
```

## GUI Workflow

1. The first frame is displayed.
2. Left-click one or more points on the target object.
3. Press `Enter` or `Space` to start tracking.

Key bindings:

- Left click: add point
- `Backspace` / `Delete`: remove last point
- `R`: clear all points
- `Esc` / `Q`: cancel

## Model Selection

By default, the script uses the Hugging Face model ID:

- `facebook/sam2.1-hiera-small`

To use a local config + checkpoint:

```bash
uv run python scripts/stroboscopic_image_generator.py \
  --video video/example.MP4 \
  --out video/stroboscopic_example.png \
  --model-cfg configs/sam2.1/sam2.1_hiera_s.yaml \
  --checkpoint /path/to/sam2.1_hiera_small.pt
```

## Main Options

- `--interval-sec`: interval (seconds) between overlaid poses
- `--process-fps`: processing FPS (lower values reduce memory usage)
- `--alpha`: overlay opacity

## Notes

- The background image is always the first frame.
- For high-FPS or long videos, try lowering `--process-fps` first (for example, `15`) to avoid out of memory.
