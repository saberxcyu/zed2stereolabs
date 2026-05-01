# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python samples for Stereolabs ZED stereo cameras using the ZED SDK (`pyzed`). Two independent samples: body tracking and depth sensing.

## Environment Setup

- Python **3.10** exactly (pinned in `pyproject.toml`)
- Package manager: **uv** with a local `.venv`
- `pyzed` is installed from the local wheel `pyzed-5.2-cp310-cp310-win_amd64.whl`

Install dependencies:
```bash
uv sync
```

## Running the Samples

**Body tracking** (3D skeleton + 2D overlay):
```bash
cd body_tracking
python body_tracking.py
# or with options:
python body_tracking.py --input_svo_file <file.svo> --ip_address <a.b.c.d:port> --resolution HD1080
```
Controls: `q` to quit, `m` to pause/resume.

**Depth capture** (live RGB + depth view, records to disk):
```bash
cd depth_sensing
uv run capture.py
# or with options:
uv run capture.py --min-depth 0.5 --max-depth 3.0
uv run capture.py --input_svo_file <file.svo> --ip_address <a.b.c.d:port>
```
A resolution picker UI appears at startup (1-4 to select, `q` to cancel).
Controls: `s` to start/stop recording, `q` to quit without saving.
Saves a timestamped folder containing `video.mp4` and `depth.npz`.
Depth range defaults: 0.5 m (near) to 3.0 m (far).

**Depth analysis** (post-capture time-series viewer):
```bash
cd depth_sensing
uv run read.py <recording-folder>
```
Left-click a pixel or drag a region on the frame panel to plot depth over time.
Drag the slider to scrub through frames.

**Input modes** (shared by both samples):
- Default: live wired ZED camera
- `--input_svo_file`: replay a recorded `.svo` / `.svo2` file
- `--ip_address`: connect to a ZED streaming over the network (`a.b.c.d:port` or `a.b.c.d`)

## Architecture

### Body tracking viewer pattern
Body tracking combines two simultaneous views:
- **OpenGL viewer** (`body_tracking/ogl_viewer/viewer.py`): 3D GLUT window, mouse-navigable camera, GLSL shaders.
- **OpenCV viewer** (`body_tracking/cv_viewer/`): 2D skeleton overlay drawn on the left camera image via `cv2.imshow`.

### Depth sensing pipeline
`capture.py` records live depth data from the ZED camera:
- `sl.DEPTH_MODE.NEURAL` with `depth_stabilization=True` for temporal smoothing.
- `confidence_threshold=95` on `RuntimeParameters` (inverted scale: 0=strict, 100=permissive).
- Depth frames stored as float32 in memory, saved to `depth.npz` (compressed NumPy archive with `frames`, `timestamps`, `min_depth`, `max_depth` keys).
- Sentinel values from the SDK: `+inf` = TOO_FAR, `-inf` = TOO_CLOSE, `NaN` = occluded/no data. All are rendered black in the live colormap.

`read.py` loads a recorded folder and provides an interactive matplotlib viewer:
- Depth overlay uses `jet_r` colormap (red=near, blue=far) with RGBA blending over the RGB video.
- Non-finite values (`NaN`, `+/-inf`) are forced to opaque black after colormap application.
- Single-pixel click or drag-to-select-region both plot depth vs. time on the right panel.

### ZED SDK patterns
- `sl.Camera` is the central object; always opened with `InitParameters` and closed with `zed.close()`.
- Body tracking requires positional tracking enabled first (`enable_positional_tracking`) before `enable_body_tracking`.
- `sl.BODY_FORMAT.BODY_18/34/38` controls skeleton topology; `BODY_18_BONES` / `BODY_34_BONES` / `BODY_38_BONES` provide the bone connectivity for rendering.
- Memory type (`sl.MEM.CPU` vs `sl.MEM.GPU`) must match between `retrieve_measure` and how the data is read downstream.
