# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python application for Stereolabs ZED stereo cameras using the ZED SDK (`pyzed`). Functionality is split across `software/` modules: `software.py` is the entry point that dispatches to four independent mode modules (`_depth_record`, `_depth_analyze`, `_pose_record`, `_pose_analyze`), with shared code in `_common.py`.

## Environment Setup

- Python **3.10** exactly (pinned in `pyproject.toml`)
- Package manager: **uv** with a local `.venv`
- `pyzed` is installed from the local wheel `pyzed-5.2-cp310-cp310-win_amd64.whl`

Install dependencies:
```bash
uv sync
```

## Running the Software

```bash
cd software
uv run software.py
```
A 2x2 mode selection window appears first (Depth row / Pose row, each with Record and Analyze):
- **Depth > Record**: recording settings dialog (resolution left, depth range right; HD2K and 0.5/1.5 m defaults) -> live RGB+depth view. Press `s` to start/stop recording, `q` to return to the mode menu, or close [x] to exit. Saves `recordings/depth_<timestamp>/video.mp4` (left sensor rectified frames only) and `depth.npz`.
- **Depth > Analyze**: folder picker (opens to `recordings/`) -> loads `video.mp4` + `depth.npz` -> matplotlib viewer. Click or drag a region to plot depth over time. Slider scrubs frames, `[>]` plays. `q` or close [x] controls navigation.
- **Pose > Record**: pose settings dialog (resolution left, keypoint format BODY_18/34/38 right; HD2K and BODY_18 defaults) -> live side-by-side view (raw RGB left, skeleton overlay right). Press `s` to start/stop, `q` to return, [x] to exit. Saves `recordings/pose_<timestamp>/video.mp4` (left sensor rectified frames only) and `pose.npz`.
- **Pose > Analyze**: folder picker (opens to `recordings/`) -> loads `pose.npz` + `video.mp4` -> matplotlib viewer. Select a keypoint from radio buttons to plot its X, Y, Z position in meters over time; axis colors match the on-screen coordinate gizmo (red=X, green=Y, blue=Z). Left panel shows video with skeleton overlay and coordinate axes gizmo. Slider scrubs frames, `[>]` plays. `q` returns to menu, [x] exits.

## Directory Structure

```
ZED/
  software/
    software.py         # entry point: mode dialog + dispatch (~50 lines)
    _common.py          # shared imports, constants, and helper functions
    _depth_record.py    # depth_record_mode(): live depth recording
    _depth_analyze.py   # depth_analyze_mode(): depth analysis viewer
    _pose_record.py     # pose_record_mode(): live pose/skeleton recording
    _pose_analyze.py    # pose_analyze_mode(): pose trajectory analysis viewer
    recordings/         # all recordings land here (created automatically)
      depth_<timestamp>/
        video.mp4
        depth.npz
      pose_<timestamp>/
        video.mp4
        pose.npz
    tools/
      raspberry_pi/              # SDK-free pipeline for Raspberry Pi 4 data collection
        record.py                # Pi: capture raw stereo AVI via UVC/V4L2 (no SDK/CUDA)
        process.py               # Desktop: rectify raw AVI into left/right PNG sequences
        calibration/
          SN*.conf               # ZED calibration file (copy from ProgramData/Stereolabs/settings/)
        recordings/              # raw recordings land here
          raw_stereo_<timestamp>.avi
          raw_stereo_<timestamp>_extracted_<suffix>/
            left_rectified/
              frame_000000.png
              ...
            right_rectified/
              frame_000000.png
              ...
  .venv/
  pyproject.toml
  CLAUDE.md
```

## Architecture

### Software pipeline
`software.py` is a thin entry point (~50 lines) that shows the mode dialog and dispatches to four mode modules. All shared code (imports, constants, helpers, dialogs) lives in `_common.py`; each mode is self-contained in its own module. UI framework boundary:
- **tkinter**: all pre-capture dialogs (mode selection, resolution picker, depth range, folder picker)
- **OpenCV**: live capture window only (side-by-side RGB + depth or skeleton during recording)
- **matplotlib**: analysis viewer only (frame panel, plot, frame slider)

Recording output directory is always `software/recordings/`, defined as:
```python
RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
```

**Depth record mode** (`depth_record_mode` in `_depth_record.py`): records live depth data from the ZED camera:
- `sl.DEPTH_MODE.NEURAL` with `depth_stabilization=True` for temporal smoothing.
- `confidence_threshold=95` on `RuntimeParameters` (inverted scale: 0=strict, 100=permissive).
- Depth frames stored as float32 in memory, saved to `depth.npz` (compressed NumPy archive with `frames`, `timestamps`, `min_depth`, `max_depth` keys).
- Sentinel values from the SDK: `+inf` = TOO_FAR, `-inf` = TOO_CLOSE, `NaN` = occluded/no data. All are rendered black in the live colormap.

**Depth analyze mode** (`depth_analyze_mode` in `_depth_analyze.py`): loads a recorded folder and provides an interactive matplotlib viewer:
- Depth overlay uses `jet_r` colormap (red=near, blue=far) with RGBA blending over the RGB video.
- Non-finite values (`NaN`, `+/-inf`) are forced to opaque black after colormap application.
- Single-pixel click or drag-to-select-region both plot depth vs. time on the right panel.

**Pose record mode** (`pose_record_mode` in `_pose_record.py`): live body tracking with optional recording:
- Requires positional tracking enabled before body tracking (ZED SDK requirement).
- `sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE` used for best multi-person quality.
- `draw_skeleton_overlay` draws bones and joints directly on a BGR numpy frame; each tracked person gets a distinct color cycled from `BODY_COLORS` by `body.id % 10`.
- `pose.npz` format: fixed-shape arrays padded with `NaN`/`-1` for absent persons per frame.
  - `timestamps` (N,) float64 - seconds since recording start
  - `n_persons` (N,) int32 - detected persons per frame
  - `person_ids` (N, 10) int32 - tracking ID per slot; -1 = empty slot
  - `keypoints_3d` (N, 10, K, 3) float32 - XYZ in meters; NaN = absent (K = 18/34/38)
  - `keypoints_2d` (N, 10, K, 2) float32 - pixel coords at camera resolution; NaN = absent
  - `body_confidence` (N, 10) float32 - 0-100 confidence; NaN = absent
  - `body_format` scalar int32 - 18, 34, or 38
  - `image_scale` (2,) float64 - [sx, sy] scale factor from camera resolution to display resolution
- MAX_PERSONS=10 is an imposed array-sizing limit, not a ZED SDK model limit.

**Pose analyze mode** (`pose_analyze_mode` in `_pose_analyze.py`): offline trajectory viewer for pose.npz recordings:
- `keypoints_2d` is used only for the skeleton overlay on the video frame. All plot data comes from `keypoints_3d` (XYZ in meters).
- Select a keypoint from radio buttons -> plots X (red), Y (green), Z (blue) position in meters over time, with colors matching the coordinate axes gizmo drawn on the video frame. Reveals lateral drift, vertical travel, or unexpected depth motion.
- A red dashed vertical line tracks the current frame position on the chart as the slider is dragged.
- Coordinate axes gizmo drawn in the bottom-left corner of the video frame: X=right (red), Y=up (green), Z=into-screen (blue).

### Raspberry Pi pipeline

Two-step SDK-free workflow for collecting stereo data on a Raspberry Pi 4.

**Step 1 - `record.py` (runs on Pi):**
- Treats the ZED camera as a standard UVC device via V4L2 -- no ZED SDK or CUDA needed.
- ZED outputs a composite side-by-side frame at double width (left|right) over USB 3.0.
- Records raw stereo frames as HFYU lossless AVI -- no quality loss during capture.
- CLI: `python record.py [HD2K|HD1080|HD720|VGA]` -- defaults to HD2K (15 fps).
- Ctrl+C stops recording; the frame queue is drained before the file is finalized.
- AVI 2 GB limit applies at HD2K (~25 seconds); use HD1080 or HD720 for longer sessions.
- Saves to `raspberry_pi/recordings/raw_stereo_<timestamp>.avi`.

**Step 2 - `process.py` (runs on desktop/laptop):**
- Splits each composite frame into left and right halves, then applies stereo rectification
  using the ZED calibration parameters from `calibration/SN*.conf`.
- CLI: `python process.py <video_file> [HD2K|HD1080|HD720|VGA]` -- defaults to HD2K.
- Calibration file is auto-detected from `raspberry_pi/calibration/`. Copy it from:
  `C:\ProgramData\Stereolabs\settings\SN<serial>.conf` (requires ZED SDK installed on Windows).
- Outputs PNG sequences: `<stem>_extracted_<suffix>/left_rectified/` and `right_rectified/`.
- Rectified frames match the quality of `sl.VIEW.LEFT` / `sl.VIEW.RIGHT` from the SDK
  (same calibration math; HFYU intermediate is lossless so no pixel degradation).

### Video output convention
All `video.mp4` files saved by `software.py` (depth and pose modes) contain **left sensor rectified frames only** (`sl.VIEW.LEFT`). In pyzed 5.2, `sl.VIEW.LEFT` and `sl.VIEW.RIGHT` return rectified frames by default; the `_UNRECTIFIED` suffix opts out. If both left and right rectified frames are needed (e.g. for stereo reconstruction or disparity algorithms), use video.py inside software/tools.

### ZED SDK patterns
- `sl.Camera` is the central object; always opened with `InitParameters` and closed with `zed.close()`.
- Body tracking requires positional tracking enabled first (`enable_positional_tracking`) before `enable_body_tracking`.
- `sl.BODY_FORMAT.BODY_18/34/38` controls skeleton topology; `BODY_18_BONES` / `BODY_34_BONES` / `BODY_38_BONES` provide the bone connectivity for rendering.
- Memory type (`sl.MEM.CPU` vs `sl.MEM.GPU`) must match between `retrieve_measure` and how the data is read downstream.
