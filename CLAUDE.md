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
- **Depth > Analyze**: folder picker (opens to `recordings/`) -> loads `video.mp4` + `depth.npz` -> matplotlib viewer. Click or drag a region to add it as a named ROI (prompts for a name via a Tk dialog); each confirmed ROI gets its own colored marker/rect on the frame and its own line on the depth-over-time chart, so multiple ROIs can be compared at once. Slider scrubs frames, `[>]` plays. `q` or close [x] controls navigation.
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
      "data collection"/         # offline SVO record -> process -> analyze pipeline (space in dirname)
        record.py                # capture only: stereo SVO recording via ZED SDK
        process.py                # offline NEURAL depth replay of a recorded SVO -> organized XYZ point cloud + ZED body tracking
        select_roi.py              # interactive 2D ROI selection over pointcloud.bin -> CSV
        plot.py                     # CSV -> depth-over-time PNG
        segment.py                   # SAM3 person segmentation over video.mp4 -> mask.bin
        combine_data.py              # pointcloud.bin + mask.bin + pose.npz -> filtered combined_data.npz (for SMPLify-3D)
        visualize.py                  # interactive QC viewer for combined_data.npz (mask silhouette + skeleton overlay)
        models/
          sam3.pt                    # manually downloaded (gated, see below) -- not committed
        recordings/                # each recording's full pipeline output lands here
          svo_<timestamp>/
            recording.svo2         # record.py
            video.mp4               # process.py
            pointcloud.bin           # process.py (raw float32, appended, shape (N,H,W,3))
            pointcloud_meta.npz       # process.py (timestamps_ns, height/width, intrinsics, depth range)
            pose.npz                   # process.py (ZED body tracking; omit with --no-pose)
            mask.bin                    # segment.py (raw uint8, appended, shape (N,H,W))
            mask_meta.npz                 # segment.py
            combined_data.npz                    # combine_data.py (filtered points + joints, for SMPLify-3D)
            roi.csv                    # select_roi.py (frame,roi,x,y,z,timestamp)
            roi_depth_over_time.png    # plot.py
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
- Single-pixel click or drag-to-select-region each prompt for a name (`simpledialog.askstring`, via a hidden `tk.Tk()` root created/destroyed around the viewer session) and, once named, are added as a permanent ROI rather than replacing the previous selection -- empty or duplicate names are rejected and the selection is discarded.
- Each confirmed ROI gets a persistent colored marker/rectangle + text label on the frame panel, and its own line on the depth-over-time chart (`marker='o', markersize=1, linewidth=1`; y-axis inverted so decreasing depth reads as an upward peak, a chest-breathing visualization convention).
- ROI colors cycle through `ROI_COLORS`, matplotlib's default "tab10" cycle (`#1f77b4` blue, `#ff7f0e` orange, `#2ca02c` green, `#d62728` red, `#9467bd` purple, `#8c564b` brown, `#e377c2` pink, `#7f7f7f` gray, `#bcbd22` olive, `#17becf` cyan) -- 10 distinct colors, so an 11th ROI repeats the first. Matches the color `plot.py` assigns automatically (no explicit color set there, so matplotlib's own default cycle applies in ROI-confirmation order) and `select_roi.py`'s duplicated copy of the same palette, so a given ROI is the same color on the frame markers and on the depth-over-time chart across all three tools.
- The legend sits outside the plot axes on the right (`ax_plot` narrowed to leave room) and is re-sorted alphabetically by ROI name on every addition; color assignment stays tied to confirmation order regardless of the legend's sort order.
- Chart title is fixed to `'Average Depth Over Time'` (not per-selection); x/y limits auto-rescale (`rescale_plot()`) to the finite-data extent across all confirmed ROIs, with ~10% y-padding and ~2% x-padding, instead of the full sensor depth/time range.

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
- Uses `sl.MEASURE.DEPTH` during recording (same as depth mode) — avoids the overhead of `MEASURE.XYZRGBA` so the grab loop runs at the full camera fps (~15 fps at HD2K).
- `coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP`, `coordinate_units=sl.UNIT.METER`. Forward objects have **negative Z** in this coordinate system.
- Live display: same RGB + colorized depth side-by-side view as depth record mode (`normalize_depth_to_colormap`).
- XY coordinates are **not** computed on the fly. Raw depth frames accumulate in RAM during recording. When stop is pressed, XY is computed from camera intrinsics (`fx, fy, cx, cy` from `calibration_parameters.left_cam`) before saving:
  - `X = (u - cx) / fx * depth`
  - `Y = -(v - cy) / fy * depth`  (image Y inverted vs world Y)
  - `Z = -depth`  (forward objects are negative Z in RIGHT_HANDED_Y_UP)
- `pc.npz` schema — unorganized flat array (variable point count per frame):
  - `points`       (total_N, 6) float32 — XYZRGB; XYZ in metres, RGB normalised to [0,1]
  - `frame_counts` (T,) int32           — number of valid points per frame
  - `timestamps`   (T,) float64         — seconds since recording start
  - `depth_min`    scalar float32       — minimum depth used during recording (metres)
  - `depth_max`    scalar float32       — maximum depth used during recording (metres)
  - Frame i reconstruction: `pts = points[offsets[i]:offsets[i+1]]` where `offsets = cumsum(frame_counts)`.
- No video file is saved — only `pc.npz`. Raw depth frames accumulate in RAM and XY is computed in a batch on stop before flushing to disk.

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
`QT_QPA_PLATFORM=xcb` mutes `qt.qpa.plugin: Could not find the Qt platform plugin "wayland"`. `QT_QPA_FONTDIR` tells Qt where to find fonts for rendering, but does not suppress the `QFontDatabase: Cannot find font directory` warning — that warning fires because Qt also checks for a bundled font directory inside the cv2 wheel (`cv2/qt/fonts/`) which doesn't exist. The fix is to create that directory (empty) before importing cv2 using `importlib.util.find_spec('cv2')`; fontconfig then handles actual font rendering.

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

### Offline SVO data-collection pipeline (`software/tools/data collection/`)

A separate, terminal-first pipeline for long (1-20 minute) recordings, entirely decoupled from `software.py`'s tkinter mode menu and from the six modes documented above. Lives in its own `data collection/` subfolder under `software/tools/` (note the space in the directory name — always quote it on the command line, e.g. `uv run "tools/data collection/process.py" ...`), separate from `tools/get_pc/` and from `tools/recordings/` (which holds unrelated PLY snapshots from `get_pc.py`). Exists because `_depth_record.py`'s live-capture pattern (run NEURAL depth *and* buffer every frame in RAM simultaneously) becomes laggy and RAM-heavy well before 20 minutes. Six flat scripts, no shared module between them (small helpers like the Tk file/folder picker and the `ROI_COLORS` palette are deliberately duplicated per-script rather than factored out) — only genuinely-unchanged helpers from `_common.py` (`RESOLUTION_OPTIONS`, `_center_window`, `get_display_resolution`, `create_video_writer`, `draw_status`) are imported, via `sys.path.insert(0, .../'..', '..')` (two levels up from `data collection/` to reach `software/`). **Note the naming overlap with the unrelated `raspberry_pi/record.py` / `raspberry_pi/process.py` (different subproject, different purpose, no shared code) — always disambiguate by full path.**

Each stage writes into the same `software/tools/data collection/recordings/svo_<timestamp>/` folder as the previous one (see Directory Structure above for the exact file list). Invocation convention across `process.py`/`select_roi.py`/`plot.py`/`segment.py`/`combine_data.py`: optional positional path argument (a recording folder, or for `plot.py` a CSV file), falling back to a Tk picker if omitted. `record.py` takes `--resolution`/`--compression` flags instead of a path, since it has nothing to read yet.

This pipeline now has two independent output branches from the same `process.py` point-cloud pass: the original breathing/ROI branch (`select_roi.py` → `plot.py`, untouched by the below), and a newer branch (`segment.py` → `combine_data.py`) that produces a person-only filtered point cloud + joint dataset intended to feed **SMPLify-3D** for SMPL mesh fitting downstream (outside this repo).

**Key architectural decision — raw-appended binary, not `np.savez_compressed` or a pre-allocated memmap:** `pointcloud.bin` is opened once in `'wb'` mode and each frame's `.tobytes()` is appended directly to disk as it's produced — no Python-side frame buffering, no upfront full-size allocation. `np.savez_compressed` (today's `_depth_record.py`/`_pc_record.py` pattern) needs the entire array in RAM before writing, which is infeasible at these sizes (~591GB of point cloud data for 20 minutes at HD2K@15fps: `2208×1242×3×4 bytes × ~18,000 frames`). `np.load(path, mmap_mode='r')` on a *compressed* npz silently doesn't actually memory-map either — compression makes per-frame byte offsets unpredictable, so numpy decompresses the whole thing into RAM regardless of the flag (this is a pre-existing, previously-unnoticed characteristic of `_depth_analyze.py`'s own `depth.npz` loading). Plain append makes on-disk file size the ground truth for how many frames really got written (`n_frames = os.path.getsize(path) // (H*W*3*4)`), which matters if a long `process.py` run is interrupted — no separate counter can drift out of sync with what's actually on disk. Small scalar metadata (timestamps, camera intrinsics, depth range) still goes in a small **compressed** `.npz` per stage, since only the dense per-pixel array needs the raw-append treatment.

**`record.py`** (`record_mode`, minimal rename of the former `tools/video.py` / `stereo_video_mode`, since relocated into `tools/data collection/`): unchanged capture behavior — `depth_mode=sl.DEPTH_MODE.NONE` (no depth computed during capture), live left|right rectified OpenCV preview, `zed.enable_recording()`/`disable_recording()` writing directly to `svo_<timestamp>/recording.svo2` via the SDK's own internal encoding thread (zero Python-side buffering already, both before and after the rename). No settings dialog — the former tkinter resolution/compression picker was replaced with `--resolution {HD2K,HD1080,HD720,VGA}` (default `HD2K`) and `--compression {H264,H265,LOSSLESS}` (default `H264`) CLI flags, matching the terminal-first convention of the other scripts. `'s'` start/stop, `'q'`/`[x]` quit; discard-on-interrupt `shutil.rmtree`s the whole `svo_<timestamp>/` folder. While recording, prints a live elapsed-time clock to the terminal (`\r[Recording]   12.3s`, updated every frame) via `time.monotonic()` measured from when `'s'` was pressed.

**`process.py`**: replays a recorded `.svo2` **offline** (`init.set_from_svo_file(path)`, `svo_real_time_mode` left at its default `False` so replay runs as fast as NEURAL inference allows, not paced to the original capture rate) with `depth_mode=sl.DEPTH_MODE.NEURAL`, writing `video.mp4` (RGB, via `create_video_writer`) and `pointcloud.bin` frame-by-frame until `sl.ERROR_CODE.END_OF_SVOFILE_REACHED`. Point clouds are computed **directly inline** in the grab loop — there is no separate depth-only intermediate: `zed.retrieve_measure(..., sl.MEASURE.DEPTH, ...)` still retrieves single-channel depth (cheaper than `MEASURE.XYZRGBA`), and XYZ is derived from it per-frame using camera intrinsics fixed once after `zed.open()` (`X=(u-cx)/fx*depth`, `Y=-(v-cy)/fy*depth`, `Z=-depth`, same formula `_pc_record.py` uses, `RIGHT_HANDED_Y_UP`: forward objects are negative Z). This used to be a separate `convert_to_pc.py` pass reading back a `depth.bin` intermediate — folding it into `process.py`'s existing per-frame loop removes that intermediate file and its extra disk read+write pass entirely, since NEURAL inference (not the XYZ arithmetic) is already the loop's bottleneck. The `depth-only` intermediate is gone by design: this project no longer needs depth-only storage, so every recording pays the ~3x point-cloud storage cost against the ~1x-cheaper depth-only alternative that existed before. Output is an **organized** `(H, W, 3)`-per-frame point cloud — same per-pixel indexing as depth, so a 2D pixel selection maps directly to a 3D point at that index, no nearest-neighbor search needed. Per-frame timestamps use `zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()` — the camera's original hardware capture timestamp, preserved in the SVO regardless of replay speed — **not** `time.monotonic()` (which `_depth_record.py` uses correctly for *live* capture, but which would be meaningless here since replay speed has no fixed relationship to capture speed). CLI: `--depth-min`/`--depth-max` (defaults **0.25**/2m — deliberately narrower on the near end than `_depth_record.py`'s 0.5/1.5m default); resolution/dtype are not configurable (hardcoded float32, full capture resolution, matching today's precision) to keep the config surface minimal.

**`process.py` also runs ZED SDK body tracking, on by default** (`--no-pose` to skip it, `--body-format {BODY_18,BODY_34,BODY_38}` default `BODY_38`), sharing this same NEURAL-depth SVO replay loop rather than a second pass: body tracking requires depth internally anyway, so `zed.enable_positional_tracking()` + `zed.enable_body_tracking()` (called right after `zed.open()`, same order as `_pose_record.py`) and one `zed.retrieve_bodies(bodies, body_runtime)` per frame (alongside the existing `retrieve_image`/`retrieve_measure` calls) ride on the NEURAL inference already being computed for the point cloud — running it as a separate script would mean re-running NEURAL inference from scratch a second time. `body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE` and `body_runtime.detection_confidence_threshold = 40` match `_pose_record.py`'s live-capture parameters, but `body_param.enable_body_fitting = True` here (vs. `_pose_record.py`'s `False`) -- offline replay can afford the extra per-frame skeletal-fitting cost that live capture can't, for a smoother, more anatomically-constrained skeleton feeding the SMPL-dataset branch. Output `pose.npz` uses the same fixed-`MAX_PERSONS=10`-slot, `-1`/`NaN`-for-absent convention as `_pose_record.py`'s live `pose.npz` (`person_ids`, `keypoints_3d`, `body_confidence` — 2D keypoints are omitted here since there's no live overlay to draw offline), but differs in two ways: `timestamps_ns` (int64, the same per-grab `TIME_REFERENCE.IMAGE` timestamps already collected for `pointcloud_meta.npz`, guaranteeing frame-for-frame alignment) instead of `_pose_record.py`'s monotonic-seconds `timestamps` (meaningless for offline replay, same reasoning `pointcloud_meta.npz` already documents), and no `image_scale`, since `process.py` saves `video.mp4` at full native camera resolution rather than `_pose_record.py`'s live-preview-capped display. On the grab-failure branch that already writes NaN pointcloud/blank RGB frames, matching NaN/-1 pose entries are appended too, so `pose.npz`'s frame axis always stays in lockstep with `pointcloud.bin`/`video.mp4`.

**`select_roi.py`**: interactive matplotlib viewer, modeled on `_depth_analyze.py`'s proven point/region-click + `simpledialog`-naming interaction (same `ROI_COLORS` palette, same empty/duplicate-name rejection, same persistent colored marker/rectangle + label per confirmed ROI), simplified to a single frame panel + slider (no live chart — that's `plot.py`'s job now, run afterward from the CSV). Deliberately does **not** reuse `_common.py`'s `preload_rgb_frames()` (which decodes an entire video into a RAM list up front — exactly the problem this pipeline exists to avoid at 20-minute scale); instead keeps one `cv2.VideoCapture` open, doing a sequential `.read()` on forward steps and `CAP_PROP_POS_FRAMES` only on true jumps.

Takes a recording folder as its positional argument (containing `pointcloud.bin` + `pointcloud_meta.npz`), same convention as `process.py`/`plot.py`, falling back to a Tk folder picker if omitted. Each confirmed ROI stores flat pixel-index arrays (`rows`, `cols`) into the loaded point cloud — this *is* the mechanism that lets a 2D-space selection resolve to a specific pixel's 3D point later. Keys: `e` exports all confirmed ROIs to `roi.csv` (one row per valid point per frame per ROI, **not** pre-averaged — averaging is `plot.py`'s job) and exits; `q` or `[x]` cancels without writing anything (unlike `_depth_analyze.py`'s `'q'`, there's no menu to return to here). CSV schema is fixed: `frame,roi,x,y,z,timestamp`. `timestamp` is always raw Unix-epoch seconds (`timestamps_ns[frame] / 1e9`), not pre-zeroed — `plot.py` normalizes to a shared `t0` itself. Prints terminal confirmations at each stage (`[Loaded]` for the bin/meta/video files, per-ROI confirmations, `[Exporting]` progress, final `[Saved]`).

**`plot.py`**: loads `roi.csv` with stdlib `csv`+`collections.defaultdict` (no pandas, matching this repo's dependency set), groups by `(roi, frame)`, averages, and plots depth vs. elapsed time per ROI — directly modeled on the sibling LiDAR project's `plot.py` at `/home/saberxcyu/Projects/veloview-docker/plot.py` (grouping/averaging logic, `y`-axis inversion convention, and no-`plt.show()`/`savefig`-only batch behavior are all ported as-is). The point cloud's `Z = -depth`, so plotted values are `-avg_z` (positive distance-from-camera) before `ax.invert_yaxis()` — plotting raw `z` directly would read backwards. Output: `<csv_dir>/<csv_stem>_depth_over_time.png` (same folder as the input CSV, i.e. `roi_depth_over_time.png` for the standard `roi.csv` name), styled to match `_depth_analyze.py`'s existing chart conventions (`marker='o', markersize=1, linewidth=1`, inverted y-axis, `"Average Depth Over Time"` title).

**`segment.py`**: person segmentation for the SMPL-dataset branch, SAM3-only (`ultralytics.models.sam.SAM3VideoSemanticPredictor`). A YOLO-based backend (`ultralytics.YOLO(...).track()`, COCO class `0` "person" + ByteTrack) was tried first as the ungated default while SAM3 access was pending, but was dropped once SAM3 access was granted: YOLO's mask head predicts a small set of low-resolution prototype masks that are bilinearly upsampled to full frame size, producing visibly coarser silhouette boundaries than SAM3's per-pixel prediction — confirmed by reading `ultralytics/utils/ops.py`'s `process_mask_native()`, not just from the visual preview. SAM3 uses open-vocabulary **text/concept prompting** (`text=["person"]`) — it has no fixed label taxonomy, so the text prompt *is* the class filter, and it doesn't depend on first detecting a whole-body bounding box the way instance detectors do. This matters for the close-up torso recordings (`chest_svo`, `abdominal_svo`, etc.) an instance detector would otherwise struggle on. **Confirmed gated**: `sam3.pt` requires a manual Hugging Face access request/download at https://huggingface.co/facebook/sam3 (not auto-fetched by Ultralytics) — verified in practice, not just from docs. `--checkpoint` defaults to `software/tools/data collection/models/sam3.pt`; `segment.py` fails fast with the request-access instructions if it's missing.

Output: `mask.bin` (raw-appended per-frame `(H, W)` `uint8` foreground mask, same append-not-buffer convention as `pointcloud.bin`, just at 1 byte/pixel instead of 12) + `mask_meta.npz` (self-sufficient copy of `timestamps_ns`/`height`/`width`, plus `model`/`checkpoint`/`text_prompt` for provenance). Pure video + model script, no ZED SDK involvement at all, so unlike body tracking it runs as a fully separate, independently re-runnable pass over `process.py`'s already-saved `video.mp4` (decoupling its own PyTorch/CUDA context from the ZED SDK's CUDA context used during `process.py`, and avoiding re-running NEURAL inference just to re-tune segmentation).

`segment.py` also writes an annotated QC preview by default (`--no-preview` to skip) to `<outdir>/sam3_preview/`: passing `"save": True` + `"save_dir": <outdir>/sam3_preview` as `overrides` for the predictor needs no extra code, since Ultralytics' `write_results()` (which draws the mask overlay and saves it) runs inside the exact same per-frame `stream_inference` loop `segment.py` already iterates to build `mask.bin` — confirmed by reading `ultralytics/engine/predictor.py` directly. `save_dir` (rather than `project`/`name`) is set explicitly so the preview lands next to this recording's own files instead of Ultralytics' global auto-incrementing `runs/segment/predict/` tree. Note Ultralytics writes this preview as `.avi`, not `.mp4`.

**`segment.py`'s `--imgsz` (default `1246`)**: traced through the installed Ultralytics source (`ultralytics/models/sam/predict.py`) rather than assumed. `SAM3Predictor.stride = 14` (its ViT backbone uses patch size 14); Ultralytics' `check_imgsz` rounds any requested `imgsz` up to the nearest multiple of stride. Left unset, Ultralytics' global default (`cfg/default.yaml`'s `imgsz: 640`) rounds to `644` -- this is where SAM3's out-of-the-box `644x644` comes from, not a deliberately chosen SAM3 resolution, and it's too coarse for HD2K (`2208x1242`) silhouette edges. `1246` (89x14) was chosen to roughly match HD2K's `1242`px height, minimizing detail loss before the model's internal downsampling. Two hard constraints from the SDK, not settings: SAM-family predictors assert a **square** `imgsz` (`setup_source`: `imgsz[0] == imgsz[1]`), and `SAM3SemanticPredictor.pre_transform` always **stretches** (not letterbox-pads: `LetterBox(..., scale_fill=True)`) the frame to fill that square -- so HD2K's 16:9 frame always gets non-aspect-preserving-squished regardless of `imgsz`; only the amount of pre-squish detail loss is tunable. Compute cost scales roughly quadratically with `imgsz`, so this is deliberately exposed as a CLI flag (not hardcoded) to tune per-recording rather than commit to one value blind -- current plan is to try `1246` first and step down if throughput is unacceptable, per the unmeasured-SAM3-throughput risk already noted below.

**`retina_masks` does not help SAM3, confirmed by reading the source, not assumed:** the base `Predictor.__init__` (which `SAM3VideoSemanticPredictor` inherits through `SAM3SemanticPredictor` -> `SAM3Predictor` -> `SAM2Predictor`) unconditionally sets `self.args.retina_masks = True` after applying `overrides`, so it can't even be turned off via `overrides`. More importantly, `SAM3SemanticPredictor.postprocess` (inherited by the video variant, which defines no `postprocess` of its own) never reads `self.args.retina_masks` at all -- it unconditionally does `F.interpolate(masks, orig_img.shape[:2], mode="bilinear")`, upsampling straight to the original frame's full resolution regardless. `retina_masks` is a YOLO instance-segmentation concept irrelevant to SAM's custom postprocess path; `--imgsz` is the only real lever over SAM3 mask edge fidelity here.

**`combine_data.py`**: final packaging step for the SMPL-dataset branch, combining `pointcloud.bin` + `mask.bin` + (optionally) `pose.npz` into one `combined_data.npz`. `--mask-threshold` (default `0.5`) is applied here rather than baked into `mask.bin`, so re-thresholding doesn't require re-running SAM3. Per frame: threshold the mask, index the organized point cloud to a flat foreground-only `(N_i, 3)` array (plus the same `np.isfinite` validity check `select_roi.py` already uses), and — if `pose.npz` is present — select that recording's single persistent subject (same "most-frames" track-ID policy as `segment.py`) and pull their joints for that frame. `pose.npz` is optional (recording processed with `--no-pose`, or predates `segment.py`/`combine_data.py` existing) — if absent, `combine_data.py` warns and omits `joints_3d`/`joint_confidence`/`body_format` from the output entirely rather than filling placeholder NaNs. Output schema deliberately reuses `_pc_record.py`'s already-established flat/`frame_counts`/cumsum convention for `pc.npz` rather than inventing a new one: `points` (total_N, 3) float32 flat XYZ, `frame_counts` (T,) int32, `timestamps` (T,) float64, plus `joints_3d` (T, n_kp, 3) float32 (NaN where the subject wasn't detected that frame) and `joint_confidence` (T, n_kp) float32 when pose data exists. Frame `i` reconstruction: `pts_i = points[offsets[i]:offsets[i+1]]` where `offsets = cumsum(frame_counts)`, `joints_i = joints_3d[i]` — same access pattern as `pc.npz`. No point color by default (SMPLify-3D typically only needs geometry + joints).

Beyond picking the primary person by track ID, `combine_data.py` also masks each frame's joints individually against that frame's SAM3 foreground mask (`mask_joints_to_fg`/`project_to_pixel`, the exact inverse of `process.py`'s own depth->XYZ projection, run backwards): any joint whose projected pixel falls outside the mask (or out of frame bounds) is set to NaN alongside its `joint_confidence`, exactly like an undetected joint. This exists because ZED body tracking and SAM3 segmentation are independent systems with no cross-referencing between their own person-selection logic — the ID-persistent "primary" body ZED's tracker picks isn't guaranteed to be the same physical person SAM3's mask picked, so per-joint mask filtering (not the ID selection alone) is what actually keeps a frame's output joints consistent with its masked point cloud. (Explicitly considered and rejected: replacing ID-based primary-person selection with a per-frame "best mask overlap across all detected bodies" selector — judged unnecessary on top of the per-joint mask filter already in place.)

**`visualize.py`**: QC viewer for the SMPL-dataset branch's final `combined_data.npz`, so you don't have to trust `combine_data.py` blindly or fall back to the much heavier `pointcloud.bin`/`mask.bin`. Deliberately reads only `combined_data.npz` + the small `pointcloud_meta.npz` (never the raw bins) — it reconstructs the person silhouette by reprojecting `combined_data.npz`'s already-flattened, already-masked `points` back into pixel space with the exact inverse of `process.py`'s own depth->XYZ formula (`depth=-Z`, `col=cx+X*fx/depth`, `row=cy-Y*fy/depth`), which recovers the silhouette essentially pixel-perfectly since forward/inverse are exact algebraic inverses of the same math (rare single-pixel rounding gaps are the only artifact, harmless for QC). No RGB is shown — `combined_data.npz` never kept any.

Color: `t = (z - min_per_pixel) / (max_per_pixel - min_per_pixel)` is normalized **per reprojected pixel location across the whole clip** (not one global min/max over every point), so color reflects a spot's own relative motion (e.g. breathing amplitude at that point) rather than its absolute depth relative to the rest of the body. "Same point over time" == same reprojected `(row, col)` — there is no true 3D point tracking/correspondence available, same fixed-pixel convention `select_roi.py`/`plot.py` already use. Black (`t=0`, a pixel at its own farthest/exhale extreme) fades to blue (`t=1`, its own closest-to-camera/inhale extreme) via a plain linear RGB interpolation (`BLACK_BASE`/`BLUE_PEAK` constants), not a matplotlib colormap — chosen over an earlier `jet_r` red-blue scheme specifically so breathing reads as a one-directional "activation" signal (blue = breathing in) against a neutral base, rather than a diverging near/far gradient.

Skeleton overlay (only if `combined_data.npz` has `joints_3d`, i.e. the recording had `pose.npz` and at least one detection): joints and bones are reprojected with the identical inverse-projection math, then drawn via `pyzed.sl`'s own `BODY_18_BONES`/`BODY_34_BONES`/`BODY_38_BONES` connectivity tables and `BODY_18_PARTS`/`BODY_34_PARTS`/`BODY_38_PARTS` name enums (looked up by the `body_format` int stored in the npz) — matplotlib `Text` labels per joint, positioned/hidden per frame via `set_position`/`set_visible` rather than recreated, so scrubbing stays cheap. No confidence readout is shown (removed by request) — worth noting for any future re-add that `joint_confidence` in `combined_data.npz` is a **per-person**, not per-joint, value broadcast across all `n_kp` entries (`combine_data.py` assigns one `body_confidence[frame, slot]` scalar into a whole `(n_kp,)`-shaped row), so a per-joint display would have implied a precision the data doesn't actually have.

**Rewritten once for a real out-of-memory crash**: the first version projected/colorized the *entire* clip's `points` array in one vectorized numpy pass. That worked fine on a 137M-point test clip but OOM-killed the process on a real 518.6M-point recording (`off_wall_svo_20260727_093651`) — several full-clip-sized float/int temporaries (row, col, row_i, col_i, key, ...) multiplied into tens of GB. Fixed by processing one frame's point slice at a time everywhere (`project_frame_pixels`, `compute_pixel_ranges`, `render_frame_canvas`) — peak memory now stays close to just the size of the loaded `points` array itself (~6GB for that same 518M-point recording), regardless of clip length, and it's incidentally ~5x faster too (streaming avoids the huge temporaries entirely rather than just deferring them).

**Known open risks, not yet exercised against a real long recording**: total disk usage for a full 20-minute HD2K clip (`pointcloud.bin`) is on the order of ~591GB — validate on short (1-2 minute) clips first. NEURAL-mode SVO replay throughput on this GPU is unmeasured. `select_roi.py`'s `mp4v`/OpenCV random-seek frame accuracy on a long GOP-encoded video is unverified — if inaccurate, the fallback is forward-only sequential reads. `segment.py`'s per-frame throughput on a full 20-minute clip is unmeasured and likely far slower than a lighter instance detector would be (SAM3 is a large foundation model, not built for real-time use) — validate on a short clip first, same as the disk-usage risk above. ZED body tracking's detection confidence on these close-up torso-only recordings (as opposed to full-body views) is unvalidated — `process.py`'s NaN-for-absent convention degrades gracefully if detection fails, but hasn't been checked against a real close-up clip yet. `SAM3VideoSemanticPredictor`'s exact `Results` object attribute names for per-instance masks and track IDs are not fully documented by Ultralytics — `segment.py`'s implementation was worked out by reading the installed version's source rather than trusting the (thin) official docs.

**A real depth-range gotcha hit during testing, worth remembering**: `process.py`'s `--depth-max` default (1.5m) is tuned for the original close-up torso rig. A full-body recording taken at normal room distance (e.g. standing a few metres from the camera) will have the *entire subject* clipped to `+inf` (`TOO_FAR`) in `pointcloud.bin`, since depth outside `[--depth-min, --depth-max]` is non-finite by design (see `_common.py`'s sentinel-value convention). This silently produces an ROI/dataset with zero valid points — not a bug in `select_roi.py`'s pixel↔pointcloud indexing (verified: video/point-cloud frame counts and dimensions matched exactly, and the invalid region was stable across the whole clip). `select_roi.py`/`plot.py` currently give no feedback when a confirmed ROI has zero valid points until `plot.py` crashes on an empty CSV — reprocessing the same `recording.svo2` with a wider `--depth-max` (no need to re-record) is the fix when the subject is farther than the depth window; a friendlier empty-ROI error message in `plot.py` is still an open, not-yet-applied improvement.

### Raspberry Pi / Jetson pipeline

Two-step SDK-free workflow for collecting stereo data on a Raspberry Pi or Jetson.

**Step 1 - `raspberry_pi/record.py`:**
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

**Why `software/tools/record.py` (formerly `video.py`) previously had this problem:** writing two separate VideoWriter files per
frame (left + right) doubled per-iteration encode overhead, pushing the loop past the frame
period. Switching to ZED SDK SVO recording (`zed.enable_recording`) fixes this: the SDK
records in a separate internal thread and feeds at camera_fps regardless of Python loop speed.

**`raspberry_pi/record.py` uses a threaded writer:** encoding runs in a dedicated thread decoupled from
the main capture loop via a 16-frame write queue. Encoding never stalls frame capture. If
encoding falls behind, the write queue fills and drops the oldest frame -- actual written
fps falls -- but this does not cause sped-up playback because VideoWriter fps is set to
the declared target at init time regardless of actual write rate. The `[Stats]`
print on stop reports actual fps, confirming whether any drops occurred.

### If raspberry_pi/record.py write queue drops frames

If `[Stats]` reports actual fps significantly below target, the encoder cannot sustain the
write rate. Playback speed is correct (VideoWriter fps = declared target), but frames
are missing -- the recording is shorter than real time. Lower the fps cap in
`RESOLUTION_OPTIONS` for the affected resolution until actual fps meets target.

### Encoding approaches tried and abandoned (raspberry_pi/record.py)

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

`software/tools/data collection/record.py` records full stereo as a single `svo_<timestamp>/recording.svo2` file using the ZED SDK's built-in SVO recorder (`zed.enable_recording`). SVO2 stores raw sensor data; left/right rectified views are reconstructed on playback through the SDK. Compression is selectable at launch: H264 lossy (default), H265 lossy, or Lossless. See "Offline SVO data-collection pipeline" above for the full record -> process -> select_roi -> plot workflow built around this file.

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
