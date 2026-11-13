https://docs.opencv.org/3.4.17/df/d4a/tutorial_charuco_detection.html

# Calibrated world-space SMPL-B fitting

Last updated: 2026-11-11. Status: discussion documented; implementation has not started.

## Objective and agreed direction

Correct the camera mounting tilt in future captures using ZED SDK gravity alignment and floor initialization. Fit SMPL-B in a calibrated, Z-up world frame, with explicit camera intrinsics and extrinsics for image projection and visibility. Initially support one ZED 2 camera; later extend to four ZED X stereo cameras.

Keep the single-camera and four-camera approaches separately runnable for ablation studies. Prefer shared geometry/loss code with explicit configurations or entry points over duplicated implementations. Preserve the existing camera-space method as a reproducible baseline.

Existing recordings do not need to be corrected as part of this work. They remain useful for baseline comparisons and exploratory checks.

## Project context

This repository processes ZED recordings into person point clouds and WholeBody-133 2D landmarks, then fits SMPL with breathing deformation:

- `src/preprocess.py`: SAM3 person segmentation, RTMW landmark detection, extraction of foreground points.
- `src/manual_pointcloud_selector.py`: selection and automatic point-cloud cleaning.
- `src/registration.py`: staged optimization using image reprojection, Chamfer distance, shape/pose priors, and temporal regularization.
- `src/fit_config.py`: stages 1–4 primarily fit landmarks and priors; stage 5 adds full-body Chamfer fitting; stage 6 freezes shape/pose and fits breathing displacement, with translation still trainable under the current defaults.
- `src/losses.py`: projection, mesh visibility, Chamfer, and priors.
- `src/common.py`: camera-to-Z-up axis conversion and SMPL parameter conversion.
- `src/world2radar.py`: downstream radar-frame conversion; its assumptions need review when world calibration is introduced.

Collection repository: `/home/saberxcyu/Projects/zed2stereolabs/`.

Exact scripts used:

- `software/tools/data collection/record.py`: records SVO2.
- `software/tools/data collection/process.py`: replays SVO2, computes depth, reconstructs optical-camera XYZ, and exports `pointcloud.bin`, `pointcloud_meta.npz`, video, and optional ZED body poses.

## Observations motivating the work

Compared these stage-5 outputs for the same 158-frame crab-walking sequence:

- Earlier: `outputs/mesh 2026-10-20/absorber_saber_crab-walking_breathing_svo_20261016_125613.443237_combined_cleaned_registered_stage5.npz`.
- Newer: `outputs/absorber_saber_crab-walking_breathing_svo_20261016_125613.443237_combined_cleaned_cleaned_registered_chamfer_fit/absorber_saber_crab-walking_breathing_svo_20261016_125613.443237_combined_cleaned_cleaned_registered_stage5.npz`.

Inputs live under `data/absorber_saber_crab-walking_breathing_svo_20261016_125613.443237/`.

| Measurement | Earlier / once-cleaned | Newer / twice-cleaned |
| --- | ---: | ---: |
| Median cloud vertical extent | 1.919 m | 1.887 m |
| Reconstructed mesh rest-pose height | 1.964 m | 1.910 m |
| Median posed mesh vertical extent | 1.934 m | 1.891 m |
| Cloud-to-mesh-vertex RMS against the same twice-cleaned cloud | 2.50 cm | 2.28 cm |

RMS was measured every fifth frame against all reconstructed mesh vertices. It is a diagnostic nearest-vertex metric, not the exact visibility-filtered training loss. Rest-pose and posed vertical extents are different measurements; neither alone is a ground-truth stature measurement.

The second cleaning pass removed 649,472 points (4.05%). Metadata confirms the expanded `clip_rateofchange_sor` filter: median-centered clipping, image-grid discontinuity/component filtering, then statistical outlier removal. The earlier input used statistical outlier removal alone.

Current stage-5 culling excludes back-facing surface landmarks. It does not perform full self-occlusion testing, does not exclude every skeletal joint, and the validity mask is computed once at the start of a Chamfer-active stage. Saved runs support improvement but do not isolate the causal contribution of cleaning versus culling.

The subject approaches from about 3.15 m to 2.19 m. In the twice-cleaned sequence, the cloud lower edge varies with depth at approximately −3.56 cm/m (R² ≈ 0.89), consistent with roughly 2° of pitch over a level floor. Upper-edge estimates are less consistent. This is evidence to investigate tilt, not a calibrated angle: gait, segmentation, floor slope, and stereo error remain confounders. Do not hard-code 2°.

A rigid transform preserves distances and body dimensions. It fixes the physical reference for height and orientation; it does not remove depth bias, outliers, or an elongated fitted body. World-space fitting improves the model only when its observations or constraints provide additional information.

## Calibration distinctions established in discussion

### Optical calibration versus mounting calibration

`raspberry_pi/calibration/SN29311469.conf` contains lens intrinsics/distortion and the left-to-right stereo relationship. It does not contain the camera mounting pose relative to the room. The SDK-based exporter obtains calibrated rectified intrinsics from the SDK already.

Retain factory optical calibration and normal SDK self-calibration unless there is evidence of an optical problem. Changing the camera mount normally requires camera-to-world calibration, not re-estimation of stereo baseline or lens parameters.

The user's previous ceiling-camera/table/grid setup likely defined a tabletop reference, possibly through a 2D homography. A flat target defines its own plane; table height is unnecessary for tabletop-relative measurements. Relating that plane to gravity, the floor, or a robot requires a known additional relationship.

A rigid checkerboard/circle-grid target remains a valid way to establish full extrinsics. Freely holding the board establishes camera-to-board pose, not an independently known room frame. We agreed to start with SDK alignment instead.

### SDK initialization for one stationary camera

Relevant `sl.PositionalTrackingParameters` fields, subject to verification against the installed SDK version:

```python
tracking.enable_imu_fusion = True
tracking.set_gravity_as_origin = True
tracking.set_floor_as_origin = True
tracking.set_as_static = True  # only when stationary throughout that recording
```

- Gravity supplies pitch/roll alignment; gravity alone does not determine yaw or position.
- Floor initialization supplies a floor reference and camera height. A sloping floor and true gravity-horizontal are distinct; validate their relationship.
- Wait for successful initialization/tracking; floor search can remain in `SEARCHING_FLOOR_PLANE`. Check return codes and states rather than silently accepting identity transforms.
- `find_floor_plane()` and `reset_positional_tracking()` provide an explicit alternative to automatic floor initialization.
- Save the resulting pose/transform and provenance with the recording. Reinitialize after changing the mounting.

Do not assume a generic IMU orientation quaternion is already in the optical camera frame; inspect SDK conventions and the provided camera/IMU transform. Prefer the SDK's camera pose in its world frame.

### Why the existing exporter needs modification

`process.py` manually computes:

```python
x = (column - cx) / fx * depth
y = -(row - cy) / fy * depth
z = -depth
```

These remain optical-camera coordinates regardless of world initialization. Setting `runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD` affects supported SDK 3D retrieval, not these NumPy expressions. Either retrieve SDK world XYZ or explicitly transform the reconstructed camera XYZ.

Keep optical-camera data available for image-grid cleaning and projection. Rotating a cloud while keeping the old direct projection formulas produces incorrect image coordinates. Preserve pixel correspondence or perform image-dependent processing in camera coordinates before world conversion.

## Coordinate contract

Use explicit names and column-vector semantics in documentation:

```text
p_world = T_world_from_camera[i] @ p_camera[i]
p_camera[i] = inverse(T_world_from_camera[i]) @ p_world
```

Points above are homogeneous. NumPy row-vector implementations must implement the equivalent transpose convention consistently.

Canonical fitting/output world: metres, right-handed, Z-up aligned with gravity; document the horizontal heading and origin. Configure ZED consistently or explicitly convert its chosen world convention to this convention. Do not assume a function written for optical camera axes is automatically the correct SDK-world-to-Z-up conversion.

The existing `R_CAM_TO_ZUP` only permutes/sign-flips axes. Its current output is Z-up by convention but is not guaranteed gravity-aligned. Replace its role in calibrated exports with the correct composed transform; avoid double-transforming.

SMPL parameter conversion must update global root orientation and translation with the model's pelvis-pivot convention. Existing `cam_to_zup_smpl` handles a pelvis offset; generalize carefully to full rigid transforms and verify by reconstructing vertices. Local joint rotations and shape parameters should remain unchanged under a global frame change. Treat any exported displacement vectors according to the frame in which they are defined.

Radar conversion requires a calibrated relationship to the new shared world. Review existing radar assumptions rather than automatically reusing an offset/rotation defined for the old camera-derived frame.

## World-space optimizer design

One SMPL-B body per timestamp is represented in world coordinates. Each camera has intrinsics, world-to-camera extrinsics, observations, and confidence/validity information.

For each camera:

```text
world-space joints
    -> world-to-optical-camera transform
    -> existing camera-axis projection convention and intrinsics
    -> compare with that image's detected 2D joints
```

Sum normalized, confidence-weighted per-camera losses. The single-camera mode is the same mathematical formulation with one observation source, while remaining separately runnable for comparisons.

Required changes to current assumptions:

| Operation | Current assumption | World-space behavior |
| --- | --- | --- |
| `losses.cam2pix` / joint reprojection | Input already optical-camera XYZ; depth is −z | Transform world joints to each camera before projecting |
| `losses.get_chamfer_points` | Camera at origin; view vector is `-vertex` | Per-camera view vector from vertex to actual camera, or evaluate visibility in camera coordinates |
| `Registration._cull_backfacing_surface_joints` | Same camera-at-origin assumption | Compute separate validity masks per camera |
| `_snapshot` reprojection error | Direct projection of optimized joints | Use the same explicit camera transform as training |
| Initialization | Identity root orientation and cloud-centroid placement in camera coordinates | Initialize appropriately in world coordinates; test different mount orientations |
| Export bounds/parameters | Assumes fit is camera-space before axis conversion | Export calibrated world data directly with frame metadata |

Chamfer distances are frame-independent when mesh and cloud share a frame. Shape/local-pose priors and intrinsic mesh regularizers can be retained. Audit temporal terms: translation second differences have useful rigid-frame properties for a fixed transform, but global axis-angle subtraction is not a rotation-invariant distance. Determine whether it needs replacement or a documented limitation.

Intrinsics optimization does not force camera-space fitting: projection can differentiate through `K_i` after world-to-camera transformation. Initially keep SDK intrinsics and calibrated extrinsics fixed. In SMPLify-family code, camera optimization often refers to translation, not intrinsic calibration; SMPLify-X stores focal lengths as buffers in its standard camera implementation. If intrinsic refinement is later added, account for the fact that depth-derived clouds also depend on calibration, and constrain ambiguous camera/body tradeoffs.

## Four-camera extension

Independent IMU/floor initialization does not produce a shared room frame: horizontal translation and yaw remain unresolved. Use array extrinsic calibration.

Stereolabs provides ZED360 for stereo-camera arrays, including the intended ZED X stereo cameras (distinct from monocular ZED X One). Its documented calibration uses IMU pitch/roll and body-keypoint observations of one person walking slowly across overlapping views to estimate camera placement/yaw. Ankle visibility can establish floor level. Export the JSON configuration; recalibrate when the array geometry changes.

Use one `T_world_from_camera[i]` mapping each optical frame into the same shared world. If SDK outputs are already in some local world frame, compose the appropriate transforms instead of applying an optical-camera transform twice. Verify Fusion configuration pose semantics against actual SDK/sample code before integration.

Preserve per-camera clouds even when generating a combined visualization/export. Optimize against per-camera observations to support visibility, confidence, density balancing, and ablations. Use 2D landmarks from all available cameras; one-view landmarks plus all clouds is a possible ablation, not the default. One backface mask cannot represent visibility from all four viewpoints. Full occlusion testing is a separate future improvement.

Synchronize capture times. ZED X on the same compatible ZED Link Duo/Quad supports hardware synchronization; verify the actual capture-card and host setup before implementing recording. SDK Fusion offers supported fusion modules, but per-frame person-cloud assembly for this optimizer may be implemented directly using calibrated transforms. Do not equate accumulated spatial mapping with synchronized reconstruction of a moving person.

## Implementation phases

### 1. Verify SDK behavior and capture requirements

- Inspect installed SDK/Python API versions and hardware availability.
- Exercise gravity/floor initialization with the stationary ZED 2 at multiple mounting tilts.
- Confirm pose direction, units, optical axes, origin, and tracking readiness.
- Decide whether calibration is initialized live and saved beside SVO2 or recovered during replay. Ensure the chosen workflow preserves calibration observations and metadata; do not assume a saved matrix is automatically applied on replay.
- Validate repeated estimates and floor behavior before assigning numerical accuracy guarantees.

### 2. Add collection calibration and metadata

- Extend the collection tool with explicit initialization, success/failure reporting, and a preview of aligned geometry.
- Save camera serial, SDK version, resolution, actual rectified intrinsics, coordinate conventions, transform direction, transform matrix, initialization settings, timestamps, and calibration provenance.
- Preserve original camera-space observations; provide calibrated world coordinates or an unambiguous transform for downstream use.
- Define how initialization frames relate to exported video/depth timestamps.

### 3. Introduce explicit frames in this repository

- Establish the observation/calibration data contract and compatibility path for legacy inputs.
- Implement reusable differentiable rigid transforms and world-space camera projection.
- Update visibility, landmark culling, diagnostics, initialization, and exports.
- Generalize SMPL global-transform export and verify vertex reconstruction equivalence.
- Keep the existing camera-space baseline runnable and record configurations needed to reproduce results.

### 4. Validate one-camera world fitting

- Use known transforms to check camera/world projection equivalence and transform round trips.
- Compare visibility and loss values for equivalent camera/world geometry.
- Verify mesh parameters reconstruct the exported transformed vertices.
- Capture a stationary scene/person and walking sequence at different camera tilts; inspect floor residuals, vertical drift, image reprojection, cloud fit, and temporal behavior.
- Compare against camera-space fit followed by calibrated transformation. Expect near-equivalent geometric objectives, allowing for initialization, parameterization, and optimizer effects.

### 5. Extend capture and fitting to four cameras

- Establish synchronized recording and array calibration.
- Load per-camera observations into the shared world.
- Add normalized multi-view joint and per-camera cloud terms, visibility masks, and diagnostics.
- Validate extrinsics using overlapping surfaces and image overlays throughout the capture volume.
- Fit one shared body and export world/radar representations.

### 6. Run ablations

Keep experiments configurable and their data, calibration, code revision, and loss schedules recorded. Candidate comparisons:

- Legacy camera-space fitting versus world-space fitting, using equivalent calibration and observations.
- One camera versus four, on matched synchronized frames from the same session.
- One camera's 2D landmarks plus all clouds versus all cameras' landmarks and clouds.
- Once-cleaned versus expanded cleaning; landmark backface culling off/on.
- SDK gravity-only versus gravity/floor initialization where the comparison has a defined reference.
- Optional later floor-contact or gravity constraints off/on.

Evaluate image reprojection per camera, cloud-to-mesh and mesh-to-observed-surface distances with documented visibility, independent height/scale checks, floor/contact consistency, temporal smoothness, and runtime. Do not compare only each method's own training loss or mistake axis-aligned extent for true stature.

## Open decisions for the next session

- Which SDK version and hardware will run calibration now?
- Does each recording keep the camera stationary? Future camera movement requires per-frame poses rather than one fixed extrinsic.
- Where should world horizontal origin/heading be anchored, especially relative to radar?
- Which collection-time versus replay-time calibration workflow is practical?
- What measurement accuracy is required? For reference, angular error causes vertical drift `L * tan(angle)` across horizontal separation `L`: over 1 m, 2° ≈ 34.9 mm, 0.5° ≈ 8.7 mm, 0.1° ≈ 1.7 mm. These are geometry, not promised SDK accuracy.
- Which legacy configurations/results should be preserved as official ablation baselines?

Next practical step: verify SDK initialization and transform conventions on hardware, then agree on the saved calibration/observation format before changing the optimizer.

## Primary references

- [ZED camera calibration](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/camera-calibration)
- [Positional tracking settings](https://docs.stereolabs.com/docs/development/zed-sdk/modules/positional-tracking/settings/)
- [Plane detection and floor-aligned point-cloud example](https://docs.stereolabs.com/docs/development/zed-sdk/modules/spatial-mapping/plane-detection)
- [IMU API and camera/IMU relationship](https://docs.stereolabs.com/docs/development/zed-sdk/modules/sensors/imu)
- [ZED360 array calibration](https://docs.stereolabs.com/docs/development/zed-tools/zed-360)
- [Fusion API](https://docs.stereolabs.com/docs/development/zed-sdk/modules/fusion/)
- [Multi-camera setup and synchronization](https://docs.stereolabs.com/docs/development/zed-sdk/modules/camera/multi-camera)
- [SMPLify-X camera model](https://github.com/vchoutas/smplify-x/blob/master/smplifyx/camera.py)
- [SMPLify-X fitting](https://github.com/vchoutas/smplify-x/blob/master/smplifyx/fit_single_frame.py)
- [AMASS context: MoSh++ fitting to optical mocap markers](https://github.com/nghorbani/amass#body-models)

AMASS motivated the user's Z-up output convention. Its dataset/output convention does not require camera-based fitting to use any particular optimization frame. SMPL's local model coordinates, optimization frame, and exported world convention are separate choices.

## Live hardware check — 2026-11-11

Tested connected ZED 2 SN29311469, firmware 1523, SDK/Python API 5.3.0 using the collection repository's software virtual environment. Hardware access required execution outside the workspace sandbox. HD720/30 FPS, NEURAL depth, metres, RIGHT_HANDED_Y_UP.

Gravity/floor/static tracking initialization returned SUCCESS. A 25-second run reported 30 SEARCHING FLOOR PLANE frames followed by 715 OK frames. The returned camera-to-world pose had approximately 0.980 m vertical translation and 3.1° dominant rotation about camera X (current mounting, not the historical recording). Its rotation determinant was 1.00000023; numerical camera/world round-trip error was about 1.08 micrometres. Explicit find_floor_plane also returned SUCCESS.

These results verify API availability and successful automatic initialization, not absolute calibration accuracy. The independently requested plane was retrieved with CAMERA reference active: normal approximately [-0.01219, 0.99796, -0.06264], plane offset 1.00673 m. It is not exactly consistent with the initialization pose; investigate repeatability and transformed floor residuals before accepting a production calibration. No physical height/level ground truth was provided. SDK world XYZ versus manually transformed camera XYZ and image reprojection equivalence still need direct validation.

First open-only probe succeeded but segfaulted during shutdown after first-time neural-model optimization. The subsequent full tracking probe completed with exit code 0. Keep this shutdown issue in view if it recurs.

Temporary probe: /tmp/verify_zed_world.py. Temporary pose artifact: /tmp/zed_world_verification.npz. These temporary files are not durable calibration records. Collection/fitting code remains unchanged.

## Collection implementation — 2026-11-11

Implemented in the collection repository's `software/tools/data collection/`:

- `calibrate.py`: stationary gravity/floor SDK initialization, continuous OK readiness, timeout, serial selection, explicit camera-to-Z-up-world JSON and provenance. Refuses existing output paths.
- `calibration_io.py`: schema/frame/rigid-transform/serial validation and NPZ metadata helpers.
- `record.py --calibration FILE`: checks camera identity and snapshots calibration.json beside each SVO; omission retains the legacy baseline.
- `process.py`: automatically loads the recording calibration, checks serial, saves transform/provenance in pointcloud_meta.npz and pose.npz. Points remain optical-camera coordinates, explicitly labelled, for current consumers.
- `CALIBRATION.md`: commands and coordinate contract.

Validation: Python compilation, CLI help, valid/invalid rigid-transform and serial checks passed. New calibrate.py completed on the connected camera at HD720 with SEARCHING FLOOR PLANE -> OK and estimated height 0.999 m. Test output is `/tmp/zed_calibration_implemented_test.json`. Full GUI recording and SVO replay integration have not yet been exercised. Default HD2K calibration has not yet been hardware-tested. The earlier 0.980 m versus current 0.999 m estimates reinforce that independent accuracy/repeatability validation remains necessary.

Next: exercise calibrated recording/replay, propagate metadata through SMPL preprocessing, then implement world-space fitting. This does not yet transform SMPL inputs or implement the world-space fitter.

## Repeated IMU/floor experiment — 2026-11-11

Ran three fresh camera opens, 25 seconds each, HD2K/15 FPS, NEURAL, gravity/floor/static tracking. Collected 1,125 image-synchronized IMU readings with distinct timestamps within each run and 132/132 successful explicit floor detections (~2 Hz). Artifacts and scripts: `outputs/calibration_experiment/`; numerical summary: `comparison.txt`.

Startup heights were 0.99432, 0.99675, 0.99477 m (2.43 mm span); the earlier 45.5 mm startup range did not recur in these three runs. Fresh plane detections nevertheless showed occasional large excursions even after 10 seconds. After discarding the first 10 seconds, floor perpendicular-distance medians were 0.99178, 0.99000, 0.99072 m; within-run ranges were 5.79, 56.79, 19.48 mm. Plane pitch ranges were 4.021–4.160°, 2.498–4.151°, 3.760–4.211°. These distances are normalized plane offsets in camera coordinates, not independently measured physical heights.

Over those same windows, fused IMU vertical-direction deviation from each run's mean had p95 about 0.025–0.027° and maxima below 0.036°. These are temporal repeatability statistics in the IMU orientation convention, not absolute optical-camera attitude accuracy. No camera/IMU extrinsic correction was needed for the within-sensor variability comparison; such correction is required before constructing a camera calibration from these readings.

Evidence: fresh floor estimates contain outliers that SUCCESS and a longer wait do not reject; the observed IMU direction is much steadier. Candidate next improvement is robust consensus across fresh floor estimates, with spread/outlier reporting, and independently validated gravity orientation. The ~1.8 mm spread of floor medians is retrospective evidence of repeatability in this experiment, not a demonstrated production calibration fix or ground-truth accuracy. Saved production calibrations were not modified.

## Revised implementation checklist — consolidation and world export
This section supersedes earlier suggestions to postpone the point transformation until fitting. Agreed direction: `process.py` exports calibrated Z-up world coordinates; downstream processing explicitly handles camera projection. SDK automatic floor calibration remains an experimental baseline. The user intends to obtain a rigid calibration target because repeated floor estimates were inconsistent.

### Completed now

- Merged `calibration_io.py` into `calibrate.py` and removed the separate helper file. Shared constants, loading/validation, metadata, and tilt functions are importable from `calibrate`.
- Moved the pyzed import into `main()` so importing helpers does not initialize or require SDK bindings.
- Updated `record.py` and `process.py` imports. Recording does not perform calibration. Its optional `--calibration` only validates and snapshots the selected matrix; normal recording still works without it.
- Checked import without pyzed, existing calibration JSON compatibility, metadata creation, zero-tilt convention, and syntax. No new live camera run was necessary for this refactor.

### Current behavior (before the following steps)

`calibrate.py` -> matrix JSON; `record.py --calibration` -> recording-side snapshot; `process.py` -> camera-space points plus matrix metadata. SMPL preprocessing currently drops the calibration metadata; fitting remains camera-space. No world-coordinate exporter or world-space fitter has been implemented.

### Step 1 — Target calibration in the collection repository

Files: `software/tools/data collection/calibrate.py`, `CALIBRATION.md`.

- Add an explicit target-calibration mode while retaining SDK initialization as a labelled baseline.
- Finalize target type/layout, actual dimensions, coordinate origin, mounting orientation, and measured target-to-world relationship after the target is available. Do not assume a board is level/plumb or at floor height.
- Detect target features in the rectified left image using the corresponding SDK intrinsics; estimate target-to-camera pose, account for OpenCV versus SDK axes, and compose camera-to-target with target-to-world.
- Collect repeated observations, report reprojection and pose spread, handle ambiguous planar solutions, and preserve reference images/detections for inspection.
- Save the same explicit T_world_from_camera contract, target geometry/provenance and validation status. Retain pitch/roll reporting. Validate against independent measured references; repeated estimates alone establish precision, not accuracy.

### Step 2 — Explicit calibration selection at processing time

Files: collection `record.py`, `process.py`, `calibrate.py`.

- Add `process.py --calibration FILE`, falling back to the recording's calibration.json. An explicit processing override permits recording without any calibration attachment.
- Record the calibration actually used in processing metadata; distinguish it from the original recording attachment if overridden. Validate serial and transform contract before writing outputs.
- Keep recording limited to capture and optional provenance attachment; do not require a calibration to record.
- Add an explicit output frame option (`camera` baseline or `world`). World mode must require valid calibration; never silently substitute identity. Document defaults and output names so rerunning in another frame cannot silently mix/overwrite datasets.

### Step 3 — Apply world transformation during processing

File: collection `process.py`; shared math/metadata helpers in `calibrate.py` as needed.

- Reconstruct optical-camera XYZ as currently implemented, then apply T_world_from_camera once in world mode before writing organized float32 XYZ. Preserve H x W pixel organization and NaN invalid pixels.
- Transform ZED 3D body keypoints from CAMERA to the same world frame in world mode. Explicitly retain camera reference during SDK retrieval to prevent double transformation. Leave 2D image coordinates unchanged.
- Save pointcloud_frame, keypoint frame, units, serial, actual replay intrinsics, transform and inverse, calibration source, schema version and frame-applied status. Do not reuse metadata that always labels points camera-space.
- Verify world/camera point equivalence and pixel projection round trips. Keep legacy camera mode separately runnable.

### Step 4 — Preserve geometry and image correspondence in SMPL preprocessing

Files: `src/preprocess.py`, `src/common.py` (or a focused geometry module).

- Read frame metadata and propagate it to the combined NPZ rather than reconstructing a subset that drops calibration.
- SAM3 masks still index organized world points by the same image pixels.
- Audit depth-range filtering: use camera depth obtained through inverse extrinsics, not a world coordinate component. Inspect all uses of -z and camera projection.
- Keep RTMW 2D joints in image coordinates. Preserve calibration and timestamps through slicing/filtering and exported subsets.

### Step 5 — Update cleaning and inspection tools

Files: `src/manual_pointcloud_selector.py`, `src/select_roi.py`, `src/plot_roi.py`, `src/visualize.py`, `src/plot.py`, `src/common.py`; inspect actual call sites before editing.

- Make image projection transform world points back to optical-camera coordinates.
- Evaluate image-grid discontinuity/component logic and its depth thresholds in camera coordinates; use resulting masks to retain original world-space points.
- Make coordinate clipping/frame labels explicit. Preserve metadata for cleaned, selected, and sliced NPZ outputs.
- Ensure overlays, cameras and floor render in the correct frame without an extra historical axis conversion.

### Step 6 — World-space observations and optimizer

Files: `src/dataset.py`, `src/losses.py`, `src/registration.py`, `src/fit_config.py`, shared geometry helpers.

- Dataset: load explicit observation frames, fixed camera extrinsics/intrinsics, and world points. Give legacy inputs an explicit camera-baseline path, not inferred world calibration.
- Projection: map world joints through inverse extrinsics before applying the current optical-camera projection. Use the identical path for optimization, snapshots and reported joint error.
- Visibility: use the true camera position/view direction for Chamfer samples and surface-joint culling. Keep each camera's visibility/validity separate.
- Initialization: transform initial root orientation/translation correctly or initialize directly in the world frame. Account for SMPL's shape-dependent pelvis pivot.
- Review temporal global-rotation regularization and breathing-vector conventions; do not assume changing axes leaves axis-angle penalties identical.
- Initially retain current objectives and fixed calibration. Floor/contact constraints are a separate later feature so frame conversion can be validated independently.

### Step 7 — Export, radar and end-to-end validation

Files: `src/registration.py`, `src/common.py`, `src/world2radar.py`, visualization tools.

- Export world-space mesh parameters/bounds directly. Bypass old camera-to-Z-up conversion for already-world results.
- Verify reconstructed parameters reproduce transformed vertices; update root pose/translation with pelvis offset correctly. Transform any frame-dependent displacement vectors consistently.
- Define T_radar_from_world from actual calibration; do not reuse old camera-derived radar assumptions without checking them.
- Exercise record -> process -> preprocess -> clean -> fit -> visualize with a short calibrated recording and a camera baseline. Compare projected joints, visible vertices, rigid distances and reconstructed meshes for equivalent synthetic geometry.
- Check independent height/level references when available and record remaining calibration uncertainty.

### Step 8 — Four-camera extension and ablations

- Implement synchronized per-camera capture and array extrinsics in the collection tool. Independent floor/IMU calibrations do not determine relative yaw and horizontal positions.
- Use shared target observations across connected overlapping camera pairs, or a measured multi-face target if the eventual camera arrangement requires one. Jointly refine/check array geometry and establish one world anchor.
- Extend observations/loss aggregation to multiple cameras while keeping clouds identifiable per camera. Combine clouds for viewing/export, retain per-camera visibility and weights for fitting.
- Provide separately runnable one-camera and four-camera experiment configurations with shared internals. Preserve legacy fitting and record all calibration/configuration choices for ablations.

Implementation order: complete calibration selection and export contract, update all downstream frame assumptions, then enable world fitting end-to-end. A world-space point cloud alone must not be passed to the unchanged fitter.

## Latest workflow adjustment — independent calibration and recording

Supersedes previous recording-attachment instructions: `record.py` no longer imports calibration helpers, accepts --calibration, validates calibration, or copies calibration into recording folders. It records SVO as before.

`calibrate.py` now saves automatically beside the script under `calibration/YYYYMMDD_HHMMSS.ffffff.json`. The filename uses local time and the exact `%Y%m%d_%H%M%S.%f` format used for recordings; it represents calibration completion time, not the timestamp of a separate recording. The --output argument was removed. Exclusive file creation remains in place.

Processing-time selection (`process.py --calibration FILE`) is still pending and is the intended way to choose a specific saved calibration. Do not add recording attachment back during that work. Current process.py still supports an existing manually placed recording-side calibration.json for compatibility, but does not infer calibration from timestamps. Syntax and removed-option checks passed; no hardware recalibration was run for this CLI-only change.

## Capture migration reverted

Capture tools remain authoritative in `/home/saberxcyu/Projects/zed2stereolabs/software/tools/data collection/`, using that project's environment. The copied tools and calibration JSON were removed from SMPL-B; the originals were retained. No uv dependency files were changed.

Reason: SMPL-B pins NumPy 1.23.1, whereas installed pyzed 5.3 requires NumPy >=2,<3. Keep capture/calibration and fitting environments separate. Future capture changes in the checklist target the original collection repository; downstream preprocessing/fitting changes target SMPL-B. Exchange data through the documented files and explicit coordinate/calibration metadata.
