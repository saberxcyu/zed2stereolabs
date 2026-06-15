# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python application for Stereolabs ZED stereo cameras using the ZED SDK (`pyzed`). Functionality is split across `software/` modules: `software.py` is the entry point that dispatches to six independent mode modules (`_depth_record`, `_depth_analyze`, `_pose_record`, `_pose_analyze`, `_pc_record`, `_pc_analyze`), with shared code in `_common.py`.

## Environment Setup

- Python **3.10** exactly (pinned in `pyproject.toml`)
- Package manager: **uv** with a local `.venv`
- `pyzed` 5.3 is pulled from the Stereolabs CDN via `[tool.uv.sources]` in `pyproject.toml` — no local wheel file
- ZED SDK 5.3 must be installed at the system level (`/usr/local/zed/`) before `uv sync`

Install dependencies (after system-level prerequisites are met):
```bash
cd software
uv sync
```

## Running the Software

```bash
cd software
uv run software.py
```
A 3×2 mode selection window appears first (Depth / Pose / Point Cloud rows, each with Record and Analyze):
- **Depth > Record**: recording settings dialog (resolution left, depth range right; HD2K and 0.5/1.5 m defaults) -> live RGB+depth view. Press `s` to start/stop recording, `q` to return to the mode menu, or close [x] to exit. Saves `recordings/depth_<timestamp>/video.mp4` (left sensor rectified frames only) and `depth.npz`.
- **Depth > Analyze**: folder picker (opens to `recordings/`) -> loads `video.mp4` + `depth.npz` -> matplotlib viewer. Click or drag a region to plot depth over time. Slider scrubs frames, `[>]` plays. `q` or close [x] controls navigation.
- **Pose > Record**: pose settings dialog (resolution left, keypoint format BODY_18/34/38 right; HD2K and BODY_18 defaults) -> live side-by-side view (raw RGB left, skeleton overlay right). Press `s` to start/stop, `q` to return, [x] to exit. Saves `recordings/pose_<timestamp>/video.mp4` (left sensor rectified frames only) and `pose.npz`.
- **Pose > Analyze**: folder picker (opens to `recordings/`) -> loads `pose.npz` + `video.mp4` -> matplotlib viewer. Select a keypoint from radio buttons to plot its X, Y, Z position in meters over time; axis colors match the on-screen coordinate gizmo (red=X, green=Y, blue=Z). Left panel shows video with skeleton overlay and coordinate axes gizmo. Slider scrubs frames, `[>]` plays. `q` returns to menu, [x] exits.
- **Point Cloud > Record**: PC settings dialog (resolution left, depth range in metres right; HD2K and 0.5/2.0 m defaults) -> live RGB+depth view (same as depth record). Press `s` to start/stop, `q` to return, [x] to exit. Saves `recordings/pc_<timestamp>/pc.npz` (per-pixel XYZRGB point clouds, no video file).
- **Point Cloud > Analyze**: folder picker -> loads `pc.npz` -> Open3D interactive 3D viewer. Points colored by depth using `jet_r` (red=near, blue=far). Space=play/pause, ←/→=step frames, Q=back to menu. Frame/time/point-count printed to terminal inline.

## Directory Structure

```
zed2stereolabs/
  software/                      # Part 1: ZED SDK pipeline (laptop / x86_64)
    pyproject.toml               # uv project (pyzed, opencv, matplotlib, ...)
    uv.lock
    .python-version              # pins Python 3.10
    software.py                  # entry point: mode dialog + dispatch (~90 lines)
    _common.py                   # shared imports, constants, and helper functions
    _depth_record.py             # depth_record_mode(): live depth recording
    _depth_analyze.py            # depth_analyze_mode(): depth analysis viewer
    _pose_record.py              # pose_record_mode(): live pose/skeleton recording
    _pose_analyze.py             # pose_analyze_mode(): pose trajectory analysis viewer
    _pc_record.py                # pc_record_mode(): live per-pixel point cloud recording
    _pc_analyze.py               # pc_analyze_mode(): Open3D point cloud clip viewer
    recordings/                  # all recordings land here (created automatically)
      depth_<timestamp>/
        video.mp4
        depth.npz
      pose_<timestamp>/
        video.mp4
        pose.npz
      pc_<timestamp>/
        pc.npz
    tools/
      video.py                   # stereo SVO recording via ZED SDK (requires pyzed)
      recordings/                # SVO recordings land here
        stereo_<timestamp>.svo2
      get_pc/
        get_pc.py                # standalone live point cloud viewer (ogl_viewer + PLY save)
        view_pc.py               # Open3D viewer for saved PLY files (file picker)
  raspberry_pi/                  # Part 2: SDK-free data collection (Jetson / Pi)
    pyproject.toml               # uv project (opencv-python, numpy)
    uv.lock
    record.py                    # capture raw stereo video via UVC/V4L2 (no SDK/CUDA)
    process.py                   # rectify raw video into left/right PNG sequences
    calibration/
      SN*.conf                   # ZED calibration file (copy from /usr/local/zed/settings/)
    recordings/                  # raw recordings land here
      raw_stereo_<timestamp>.mp4       # default (lossy)
      raw_stereo_<timestamp>.mkv       # lossless (pass 'mkv' argument)
      raw_stereo_<timestamp>_extracted_<suffix>/
        left_rectified/
          frame_000000.png
          ...
        right_rectified/
          frame_000000.png
          ...
  CLAUDE.md
  README.md
  .gitignore
```

## Architecture

### Software pipeline
`software.py` is a thin entry point (~90 lines) that shows the mode dialog and dispatches to six mode modules. All shared code (imports, constants, helpers, dialogs) lives in `_common.py`; each mode is self-contained in its own module. UI framework boundary:
- **tkinter**: all pre-capture dialogs (mode selection, resolution picker, depth range, folder picker)
- **OpenCV**: live capture window only (side-by-side RGB + depth or skeleton during recording; also the live display in pc_record)
- **matplotlib**: depth/pose analysis viewers (frame panel, plot, frame slider)
- **Open3D** (`VisualizerWithKeyCallback`): point cloud analysis viewer only. Must use `VisualizerWithKeyCallback` — the Open3D GUI module (Filament backend) hangs on Hyprland/XWayland and is not usable here.

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

**Point Cloud record mode** (`pc_record_mode` in `_pc_record.py`): records per-pixel XYZRGB point cloud clips:
- Uses `sl.MEASURE.XYZRGBA` — same NEURAL depth network as depth mode, different output format.
- `coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP`, `coordinate_units=sl.UNIT.METER`. Forward objects have **negative Z** in this coordinate system.
- Live display: same RGB + colorized depth side-by-side view as depth record mode (`normalize_depth_to_colormap`).
- Color unpacking from `MEASURE.XYZRGBA`: RGBA packed into float32 bits; R = bits 0–7, G = bits 8–15, B = bits 16–23 of the reinterpreted uint32. This is **not** BGR like OpenCV images.
- Depth filter uses negated thresholds: `xyz[:,2] <= -depth_min` and `xyz[:,2] >= -depth_max`.
- `pc.npz` schema — unorganized flat array (variable point count per frame):
  - `points`       (total_N, 6) float32 — XYZRGB; XYZ in metres, RGB normalized to [0,1]
  - `frame_counts` (T,) int32           — number of valid points per frame
  - `timestamps`   (T,) float64         — seconds since recording start
  - `depth_min`    scalar float32       — minimum depth used during recording (metres)
  - `depth_max`    scalar float32       — maximum depth used during recording (metres)
  - Frame i reconstruction: `pts = points[offsets[i]:offsets[i+1]]` where `offsets = cumsum(frame_counts)`.
- No video file is saved — only `pc.npz`. All point data accumulates in RAM and is flushed on stop.

**Point Cloud analyze mode** (`pc_analyze_mode` in `_pc_analyze.py`): Open3D clip viewer for pc.npz recordings:
- Uses `o3d.visualization.VisualizerWithKeyCallback` (GLFW + OpenGL). The Open3D GUI/Filament module (`open3d.visualization.gui`) hangs on Hyprland/XWayland and must not be used.
- Points are colored by depth using `jet_r` (red=near, blue=far); stored RGB channels are ignored. Color mapping: `t = clip((-z - depth_min) / (depth_max - depth_min), 0, 1)`, then `jet_r(t)`.
- `WAYLAND_DISPLAY` is unset and `DISPLAY=:0` is set before importing `open3d` (lazy import inside the function so it does not affect other modes at startup).
- Controls: Space=play/pause, ←=step back, →=step forward, Q=back to menu, [X]=quit.
- Playback speed derived from recorded `timestamps` differences, looping at end.
- Frame/time/point-count printed to terminal inline (`\r` overwrite) as frames advance.

### `_common.py` additions and fixes

**`show_mode_dialog()`** extended to a 3×2 grid (window height 280→360):
- Added a third `LabelFrame` row: **"Point Cloud"** with `pc_record` (dark green `#1a5c3a`) and `pc_analyze` (dark brown `#5c3a1a`) buttons.
- Returns `'pc_record'` or `'pc_analyze'` in addition to the existing return values.

**`show_pc_settings_dialog()`** (new): two-column dialog mirroring `show_record_settings_dialog`:
- Left column: resolution radio buttons (reuses `RESOLUTION_OPTIONS`).
- Right column: Min depth (m) and Max depth (m) entries, defaulting to 0.5 and 2.0.
- Returns `(resolution_enum, depth_min_m, depth_max_m)` or `None` if cancelled.
- Depth values are in **metres** (not mm) — `InitParameters` receives them directly.

**Qt warning suppression** (set before `import cv2`):
```python
os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')           # use X11; skip wayland plugin search
os.environ.setdefault('QT_QPA_FONTDIR', '/usr/share/fonts')  # system fonts; skip bundled dir
```
These mute the `qt.qpa.plugin: Could not find the Qt platform plugin "wayland"` and `QFontDatabase: Cannot find font directory` warnings that OpenCV's bundled Qt emits on startup.

**`normalize_depth_to_colormap()` fix**: `np.clip` passes `NaN`/`inf` through unchanged, causing a `RuntimeWarning: invalid value encountered in cast` when converting to `uint8`. Fixed by zeroing non-finite positions in `inverted` before the cast (they are overwritten black immediately after anyway):
```python
inverted[nan_mask] = 0.0
scaled = (inverted * 255).astype(np.uint8)
```

### `software.py` additions

Imports added: `show_pc_settings_dialog`, `pc_record_mode`, `pc_analyze_mode`.

Two new dispatch branches added (same pattern as existing modes):
```python
elif mode == 'pc_record':
    settings = show_pc_settings_dialog()
    ...
    result = pc_record_mode(resolution, depth_min, depth_max)

elif mode == 'pc_analyze':
    folder = filedialog.askdirectory(...)
    result = pc_analyze_mode(folder)
```

### `tools/get_pc/` — standalone point cloud tools

Two scripts in `software/tools/get_pc/` for ad-hoc point cloud quality assessment (not part of the main mode pipeline):

**`get_pc.py`**: live point cloud viewer using the ZED SDK's `ogl_viewer` (OpenGL). Press `s` to save a PLY snapshot to `tools/recordings/`. Requires the same `pyzed` venv as `software.py`. Run from the `software/` directory: `uv run tools/get_pc/get_pc.py`.

**`view_pc.py`**: Open3D viewer for the saved PLY files. Shows a file picker (opens to `tools/recordings/`), loads the selected PLY, and displays it in an Open3D window. Patches a ZED SDK PLY writer bug: the last vertex's color data is truncated, so the header vertex count is decremented by 1 before loading to suppress the RPly warning.

### Raspberry Pi / Jetson pipeline

Two-step SDK-free workflow for collecting stereo data on a Raspberry Pi or Jetson.

**Step 1 - `record.py`:**
- Treats the ZED camera as a standard UVC device via V4L2 -- no ZED SDK or CUDA needed.
- ZED outputs a composite side-by-side frame at double width (left|right) over USB 3.0.
- Always records MP4/mp4v via OpenCV VideoWriter — no external dependencies.
- CLI: `uv run record.py [HD2K|HD1080|HD720|VGA]` -- defaults to HD2K.
- Ctrl+C stops recording; the capture queue and write queue are both drained before the
  file is finalized.
- Saves to `raspberry_pi/recordings/raw_stereo_<timestamp>.mp4`.
- Uses a dedicated writer thread (`write_loop`) decoupled from the capture loop via a
  16-frame `write_queue`. Encoding never stalls frame capture. If encoding falls behind,
  the oldest queued frame is dropped. VideoWriter fps is set to the declared target fps,
  so playback speed is always correct regardless of actual write rate.
- Prints `[Stats]` on stop: frames written, elapsed time, actual fps. If actual fps < 95%
  of target, a `[Warning]` is printed.
- **Fps values in `RESOLUTION_OPTIONS`** are currently set for Jetson (mp4v encoder caps):
  HD2K=10, HD1080=12, HD720=30, VGA=30.
  When switching to Pi 5 (software encoding bottleneck), lower the caps:
  HD2K=7.5, HD1080=10, HD720=15, VGA=30.

**Step 2 - `process.py`:**
- Splits each composite frame into left and right halves, then applies stereo rectification
  using the ZED calibration parameters from `calibration/SN*.conf`.
- CLI: `uv run process.py <video_file> [HD2K|HD1080|HD720|VGA]` -- defaults to HD2K.
- Accepts `.mp4` input.
- Calibration file is auto-detected from `raspberry_pi/calibration/`. Copy it from:
  - Windows: `C:\ProgramData\Stereolabs\settings\SN<serial>.conf`
  - Linux: `/usr/local/zed/settings/SN<serial>.conf`
  (requires ZED SDK installed on the desktop machine)
- Outputs PNG sequences: `<stem>_extracted_<suffix>/left_rectified/` and `right_rectified/`.
- Rectified frames match the quality of `sl.VIEW.LEFT` / `sl.VIEW.RIGHT` from the SDK
  (same calibration math).

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

**`record.py` uses a threaded writer:** encoding runs in a dedicated thread decoupled from
the main capture loop via a 16-frame write queue. Encoding never stalls frame capture. If
encoding falls behind, the write queue fills and drops the oldest frame -- actual written
fps falls -- but this does not cause sped-up playback because VideoWriter fps is set to
the declared target at init time regardless of actual write rate. The `[Stats]`
print on stop reports actual fps, confirming whether any drops occurred.

### If record.py write queue drops frames

If `[Stats]` reports actual fps significantly below target, the encoder cannot sustain the
write rate. Playback speed is correct (VideoWriter fps = declared target), but frames
are missing -- the recording is shorter than real time. Lower the fps cap in
`RESOLUTION_OPTIONS` for the affected resolution until actual fps meets target.

### Encoding approaches tried and abandoned (record.py)

**GStreamer + nvjpegenc (Jetson hardware JPEG encoder) — abandoned 2026-06-13**

We attempted to replace OpenCV VideoWriter with a GStreamer pipeline piping BGR frames into
the Jetson hardware JPEG encoder (`nvjpegenc`) to write motion JPEG video. Summary of what
we learned and why it was dropped:

- `rawvideoparse` format values are **lowercase** (`i420`, not `I420`). Passing uppercase
  silently fails with "could not set property 'format'".
- `nvjpegenc` has two sink pad paths:
  - `video/x-raw(memory:NVMM)` — GPU memory path (fast, hardware). Requires frames already
    in NVMM memory (e.g. from `nvvideoconvert`).
  - `video/x-raw` — system RAM path. Only accepts `I420`, `YV12`, `GRAY8`. **Does not
    accept NV12 from system RAM.**
- `nvvideoconvert` (which bridges system RAM → NVMM) is **not installed** on this Jetson
  with JetPack 7 / `nvidia-l4t-gstreamer`. `gst-inspect-1.0 nvvideoconvert` returns
  "No such element".
- Even on the system RAM path (BGR → I420 → nvjpegenc), HD2K achieved ~14fps but HD1080
  consistently achieved only ~8.5fps. All individual components benchmarked fine in
  isolation (encoder: 41fps at HD2K, 53fps at HD1080 via `videotestsrc`; BGR→I420
  conversion: >1000fps). Root cause of HD1080 bottleneck was never identified — the
  GStSystemClock was observed to run "way slower" during HD1080 recording sessions, but
  why the pipeline scheduling was different at that resolution is unknown.
- **Decision:** reverted to `mp4v` via OpenCV VideoWriter. All four resolutions work
  correctly and the simpler code is more maintainable.

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
  cd software
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
