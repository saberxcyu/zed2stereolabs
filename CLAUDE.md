# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python application for Stereolabs ZED stereo cameras using the ZED SDK (`pyzed`). Functionality is split across `software/` modules: `software.py` is the entry point that dispatches to four independent mode modules (`_depth_record`, `_depth_analyze`, `_pose_record`, `_pose_analyze`), with shared code in `_common.py`.

## Environment Setup

- Python **3.10** exactly (pinned in `pyproject.toml`)
- Package manager: **uv** with a local `.venv`
- `pyzed` 5.3 is pulled from the Stereolabs CDN via `[tool.uv.sources]` in `pyproject.toml` — no local wheel file
- ZED SDK 5.3 must be installed at the system level (`/usr/local/zed/`) before `uv sync`

Install dependencies (after system-level prerequisites are met):
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
      video.py                   # stereo SVO recording via ZED SDK (requires pyzed)
      recordings/                # SVO recordings land here
        stereo_<timestamp>.svo2
      raspberry_pi/              # SDK-free pipeline for Raspberry Pi 4 data collection
        record.py                # Pi: capture raw stereo video via UVC/V4L2 (no SDK/CUDA)
        process.py               # Desktop: rectify raw video into left/right PNG sequences
        calibration/
          SN*.conf               # ZED calibration file (copy from /usr/local/zed/settings/)
        recordings/              # raw recordings land here
          raw_stereo_<timestamp>.mp4   # default (lossy)
          raw_stereo_<timestamp>.mkv   # lossless (pass 'mkv' argument)
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
- Default: MP4/mp4v lossy. Pass `mkv` argument for MKV/HFYU lossless (no pixel degradation).
  FFV1 was tried but is excluded from the custom FFmpeg build bundled with the OpenCV aarch64 wheel; HFYU is universally compiled in.
- CLI: `python record.py [HD2K|HD1080|HD720|VGA] [mkv]` -- defaults to HD2K MP4.
- Ctrl+C stops recording; the capture queue and write queue are both drained before the file
  is finalized.
- Saves to `raspberry_pi/recordings/raw_stereo_<timestamp>.mp4` (or `.mkv`).
- Uses a dedicated writer thread (`write_loop`) decoupled from the capture loop via a 16-frame
  `write_queue`. Disk writes never stall frame capture. If average write throughput falls below
  capture fps, the oldest queued frame is dropped at the queue boundary. VideoWriter fps is set
  to the same value as the capture fps, so the header is always correct.
- Prints `[Stats]` on stop: frames written, elapsed time, actual fps. If actual fps < 95% of
  target, a `[Warning]` is printed -- confirming whether the disk kept up.
- **Fps values are capped below native camera fps** to stay within the Pi 4's sustained disk
  write throughput (~44 M pixels/sec budget, derived from the observed HD2K write ceiling of
  ~12 fps with 30-40% thermal headroom). Thermal throttling (CPU drops from 1.5GHz to 600MHz
  above ~80C) can reduce throughput further during long sessions, which is why the budget is
  conservative:
  ```
  HD2K   7.5 fps  (native 15 fps; 8 not achievable via integer frame-skip -- 15/8
                   is not an integer; rate-limiter locks to 15/2 = 7.5)
  FHD   10.0 fps  (native 30 fps; exact -- 30/10 = 3)
  HD    15.0 fps  (native 30 fps; 20 not achievable -- 30/20 = 1.5 is not an integer;
                   rate-limiter locks to 30/2 = 15)
  VGA   30.0 fps  (native 30 fps; no cap needed)
  ```

**Step 2 - `process.py` (runs on desktop/laptop):**
- Splits each composite frame into left and right halves, then applies stereo rectification
  using the ZED calibration parameters from `calibration/SN*.conf`.
- CLI: `python process.py <video_file> [HD2K|HD1080|HD720|VGA]` -- defaults to HD2K.
- Accepts both `.mp4` and `.mkv` input.
- Calibration file is auto-detected from `raspberry_pi/calibration/`. Copy it from:
  - Windows: `C:\ProgramData\Stereolabs\settings\SN<serial>.conf`
  - Linux: `/usr/local/zed/settings/SN<serial>.conf`
  (requires ZED SDK installed on the desktop machine)
- Outputs PNG sequences: `<stem>_extracted_<suffix>/left_rectified/` and `right_rectified/`.
- Rectified frames match the quality of `sl.VIEW.LEFT` / `sl.VIEW.RIGHT` from the SDK
  (same calibration math; HFYU MKV intermediate is lossless so no pixel degradation).

### Stereo frame rate and write throughput

`zed.grab()` blocks until the next camera frame when the main loop is fast enough. When
per-iteration work (encode + write) exceeds the camera frame period (e.g. 66ms at HD2K
15fps), `grab()` returns a buffered SDK frame immediately without blocking. Actual write
rate = min(camera_fps, 1 / loop_time). If actual write rate < camera_fps while VideoWriter
fps is set to camera_fps, the saved video plays back faster than real time (sped up).

**Why `software.py` depth/pose modes are unaffected:** each mode writes only one video file
per frame (left rectified only). Depth and pose data accumulate in RAM and flush to NPZ on
stop. One small write per frame stays within the frame period, so `grab()` keeps blocking at
camera_fps.

**Why `video.py` previously had this problem:** writing two separate VideoWriter files per
frame (left + right) doubled per-iteration encode overhead, pushing the loop past the frame
period. Switching to ZED SDK SVO recording (`zed.enable_recording`) fixes this: the SDK
records in a separate internal thread and feeds at camera_fps regardless of Python loop speed.

**`record.py` (Pi) now uses a threaded writer:** `writer.write()` runs in a dedicated thread
decoupled from the main capture loop via a 16-frame write queue. Disk I/O can no longer slow
down capture. If average write throughput falls below camera_fps, the write queue fills and
drops the oldest frame -- actual written fps falls -- but this does not cause sped-up playback
because VideoWriter fps is always set to nominal camera_fps. The `[Stats]` print on stop
reports actual fps, confirming whether any drops occurred.

### If record.py write queue drops frames

If `[Stats]` reports actual fps significantly below target (e.g. 10fps at HD2K), the Pi disk
cannot sustain the write rate. Playback speed is correct (VideoWriter fps = nominal), but
frames are missing -- the recording is shorter than real time. Options:

- **Option B - Measured fps:** write N calibration frames to a temp VideoWriter, time the
  actual write overhead, use measured_fps for the real VideoWriter. Recording duration is
  accurate even when write rate < camera_fps, at the cost of discarding the first N frames.
- **Option C - Post-process remux:** track frame count and wall-clock duration, then fix the
  fps header after recording with `ffmpeg -r <actual_fps> -i input.mp4 -c copy output.mp4`.

### Video output convention
All `video.mp4` files saved by `software.py` (depth and pose modes) contain **left sensor rectified frames only** (`sl.VIEW.LEFT`). In pyzed 5.3, `sl.VIEW.LEFT` and `sl.VIEW.RIGHT` return rectified frames by default; the `_UNRECTIFIED` suffix opts out.

`software/tools/video.py` records full stereo as a single `stereo_<timestamp>.svo2` file using the ZED SDK's built-in SVO recorder (`zed.enable_recording`). SVO2 stores raw sensor data; left/right rectified views are reconstructed on playback through the SDK. Compression is selectable at launch: H264 lossy (default), H265 lossy, or Lossless.

### Linux-specific notes

- ZED SDK installs to `/usr/local/zed/`. Libraries are registered via `/etc/ld.so.conf.d/001-zed.conf`.
- CUDA toolkit installs to `/usr/local/cuda/` (standard Ubuntu path). `PATH` and `LD_LIBRARY_PATH` must include `/usr/local/cuda/bin` and `/usr/local/cuda/lib64`.
- tkinter dialogs use font `"DejaVu Sans"` (not `"Helvetica"`) for Linux compatibility. `"Helvetica"` does not exist on Linux; `ttf-dejavu` (Ubuntu: `fonts-dejavu`) must be installed.
- All dialog window titles are prefixed with `"ZED"` — required so window manager rules can match and float/center them.
- OpenCV live view windows are sized `1280x360` (not 1920x540) to fit within a single monitor on multi-monitor setups.
- The folder picker (`filedialog.askdirectory`) requires a **single click** to select a recording subfolder — double-clicking navigates into it rather than selecting it.
- Calibration file for the Pi pipeline is at `/usr/local/zed/settings/SN<serial>.conf` after the SDK is installed.

#### Arch Linux (current user setup)

The current user runs Arch Linux with Hyprland. Notes that differ from the Ubuntu baseline:

- CUDA installs to `/opt/cuda/` (not `/usr/local/cuda/`). Before installing the ZED SDK, symlink it: `sudo ln -s /opt/cuda /usr/local/cuda`. Also create `/usr/lib/x86_64-linux-gnu/` (Ubuntu-specific path the installer expects) and add it to ldconfig.
- Shell env (`~/.bashrc`): `PATH=/opt/cuda/bin:$PATH`, `LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH`, `CUDA_HOME=/opt/cuda`.
- ZED SDK installer will fail the `apt-get` step (expected on Arch — say `n` to system dependencies). Install missing libs manually: `yay -S openblas glew qt5-svg unzip python-pip python-setuptools`.
- Python 3.10 must come from the AUR `python310` package (`yay -S tk python310`) — **not** uv's bundled Python. uv's bundled Python ships its own Tcl/Tk without fontconfig, causing tkinter to render bitmap fonts. The AUR package links against the system `tk` (8.6) which has proper font rendering. Create the venv explicitly:
  ```bash
  uv venv --python /usr/bin/python3.10
  uv sync
  ```
- Hyprland window rules (Lua config) to float and center all ZED dialogs and OpenCV windows:
  ```lua
  windowrule = float, class:python3.10
  windowrule = center, class:python3.10
  windowrule = float, title:ZED.*
  windowrule = center, title:ZED.*
  ```
  Use `float = true` (not `floating = true`; the latter is rejected by this version of Hyprland). All tkinter dialog titles are prefixed `"ZED"` specifically to match this rule.

### ZED SDK patterns
- `sl.Camera` is the central object; always opened with `InitParameters` and closed with `zed.close()`.
- Body tracking requires positional tracking enabled first (`enable_positional_tracking`) before `enable_body_tracking`.
- `sl.BODY_FORMAT.BODY_18/34/38` controls skeleton topology; `BODY_18_BONES` / `BODY_34_BONES` / `BODY_38_BONES` provide the bone connectivity for rendering.
- Memory type (`sl.MEM.CPU` vs `sl.MEM.GPU`) must match between `retrieve_measure` and how the data is read downstream.
