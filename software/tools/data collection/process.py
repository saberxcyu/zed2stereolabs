import sys
import os
import glob
import json
import argparse
import contextlib
from calibrate import load_calibration, calibration_metadata

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pyzed.sl as sl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _common import get_display_resolution, create_video_writer

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
CALIBRATION_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

SYNC_TOLERANCE_NS = 5_000_000  # 5 ms max delta for hardware-synced frame pairing


@contextlib.contextmanager
def _suppress_c_output():
    """Silence C-level stdout/stderr (e.g. Jetson NvMMLite decoder chatter) around open/close."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)

MAX_PERSONS = 10  # array-sizing limit, not a ZED SDK model limit -- matches _pose_record.py
BODY_FORMAT_OPTIONS = {
    "BODY_18": (sl.BODY_FORMAT.BODY_18, 18),
    "BODY_34": (sl.BODY_FORMAT.BODY_34, 34),
    "BODY_38": (sl.BODY_FORMAT.BODY_38, 38),
}

CAMERA_NAMES = {
    40932664: "phoenix",   # Reference camera
    41248751: "tortoise",  # Across from phoenix (-Z)
    48029520: "dragon",    # Left of phoenix (-X)
    44577743: "tiger",     # Right of phoenix (+X)
}


def camera_name(serial: int) -> str:
    return CAMERA_NAMES.get(int(serial), f"cam_{int(serial)}")


def camera_label(serial: int) -> str:
    return f"{camera_name(serial)} ({int(serial)})"


def _svo_stem_to_folder_name(stem: str) -> str:
    if stem.startswith("cam_") and stem[4:].isdigit():
        return camera_name(int(stem[4:]))
    if stem.isdigit():
        return camera_name(int(stem))
    return stem


def load_latest_zed360_calibration(calib_dir=CALIBRATION_DIR):
    """Load the latest zed360_*.json calibration file in RIGHT_HANDED_Y_UP (metres).
    Returns (poses_by_serial, override_gravity_by_serial, ref_serial, filename) or (None, None, None, None)."""
    candidates = sorted(glob.glob(os.path.join(calib_dir, "zed360_*.json")))
    if not candidates:
        return None, None, None, None

    latest_path = candidates[-1]
    configs = sl.read_fusion_configuration_file(
        latest_path, sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP, sl.UNIT.METER
    )
    if not configs:
        return None, None, None, None

    poses_by_serial = {}
    override_gravity_by_serial = {}
    for cfg in configs:
        sn = int(cfg.serial_number)
        poses_by_serial[sn] = np.array(cfg.pose.m, dtype=np.float64).reshape(4, 4)
        override_gravity_by_serial[sn] = bool(cfg.override_gravity)

    # Reference camera is the one whose rotation is closest to identity.
    ref_serial = min(
        poses_by_serial,
        key=lambda sn: float(np.linalg.norm(poses_by_serial[sn][:3, :3] - np.eye(3))),
    )
    return poses_by_serial, override_gravity_by_serial, ref_serial, os.path.basename(latest_path)


def resolve_svo_inputs(arg):
    """Accepts a folder of .svo2 file(s), a single .svo2 file, or None (Tk picker).
    Returns a list of (svo_path, outdir) pairs, or None if cancelled/invalid."""
    if arg is None:
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(
            title="Select recording folder (or Cancel to pick a single .svo2 file)",
            initialdir=TOOLS_RECORDING_DIR,
        )
        if folder:
            root.destroy()
            arg = folder
        else:
            path = filedialog.askopenfilename(
                title="Select a single .svo2 file",
                initialdir=TOOLS_RECORDING_DIR,
                filetypes=[("SVO2 files", "*.svo2")],
            )
            root.destroy()
            if not path:
                return None
            return [(path, os.path.dirname(os.path.abspath(path)))]

    if os.path.isdir(arg):
        svos = sorted(glob.glob(os.path.join(arg, "*.svo2")))
        if not svos:
            print(f"[Error] No .svo2 files found in {arg}")
            return None
        if len(svos) == 1:
            return [(svos[0], arg)]
        return [
            (p, os.path.join(arg, _svo_stem_to_folder_name(os.path.splitext(os.path.basename(p))[0])))
            for p in svos
        ]

    if os.path.isfile(arg):
        return [(arg, os.path.dirname(os.path.abspath(arg)))]

    print(f"[Error] Path not found: {arg}")
    return None


def read_svo_timestamps(svo_path):
    """Fast pass with DEPTH_MODE.NONE to extract (serial, timestamps_ns, rot_imu, imu_euler_deg) from an .svo2."""
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.sdk_verbose = 0

    zed = sl.Camera()
    with _suppress_c_output():
        status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Could not open {svo_path}: {repr(status)}")
        sys.exit(1)

    serial = int(zed.get_camera_information().serial_number)
    runtime = sl.RuntimeParameters()
    sensors_data = sl.SensorsData()
    imu_pose = sl.Transform()
    rot_imu = np.eye(4, dtype=np.float64)
    imu_euler_deg = np.zeros(3, dtype=np.float64)
    imu_sampled = False
    timestamps_ns = []
    try:
        while True:
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err == sl.ERROR_CODE.SUCCESS:
                timestamps_ns.append(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
                if not imu_sampled and len(timestamps_ns) >= 10:
                    if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
                        imu_data = sensors_data.get_imu_data()
                        if imu_data.is_available:
                            imu_data.get_pose(imu_pose)
                            rot_imu[:3, :3] = np.array(imu_pose.m, dtype=np.float64).reshape(4, 4)[:3, :3]
                            imu_euler_deg = np.array(imu_pose.get_euler_angles(radian=False), dtype=np.float64)
                            imu_sampled = True
            else:
                timestamps_ns.append(timestamps_ns[-1] if timestamps_ns else 0)
    finally:
        with _suppress_c_output():
            zed.close()

    return serial, np.array(timestamps_ns, dtype=np.int64), rot_imu, imu_euler_deg


def compute_synced_frame_indices(svo_paths, ref_serial=None, max_dt_ns=SYNC_TOLERANCE_NS):
    """Align N .svo2 files by nearest IMAGE timestamp within max_dt_ns.
    Returns ({svo_path: set_of_raw_frame_indices}, {serial: rot_imu_4x4})."""
    print(f"[Sync] Reading timestamps & IMU gravity pose from {len(svo_paths)} SVO2 files...")
    ts_by_svo = {}
    serial_by_svo = {}
    rot_imu_by_serial = {}
    for p in svo_paths:
        sn, ts, rot_imu, euler_deg = read_svo_timestamps(p)
        ts_by_svo[p] = ts
        serial_by_svo[p] = sn
        rot_imu_by_serial[sn] = rot_imu
        print(
            f"  {os.path.basename(p)} [{camera_label(sn)}]: {len(ts)} frames  "
            f"| IMU (deg): pitch={euler_deg[0]:+.2f}, yaw={euler_deg[1]:+.2f}, roll={euler_deg[2]:+.2f}"
        )

    ref_svo = next((p for p, sn in serial_by_svo.items() if sn == ref_serial), svo_paths[0])
    ts_ref = ts_by_svo[ref_svo]

    # For each other camera, find the nearest frame index to each ref frame.
    matched_idx_by_svo = {ref_svo: np.arange(len(ts_ref), dtype=np.int64)}
    dt_by_svo = {}
    valid_mask = np.ones(len(ts_ref), dtype=bool)

    for p in svo_paths:
        if p == ref_svo:
            continue
        ts_cam = ts_by_svo[p]
        if len(ts_cam) == 0:
            valid_mask[:] = False
            matched_idx_by_svo[p] = np.full(len(ts_ref), -1, dtype=np.int64)
            dt_by_svo[p] = np.full(len(ts_ref), max_dt_ns + 1, dtype=np.int64)
            continue

        pos = np.searchsorted(ts_cam, ts_ref)
        left = np.clip(pos - 1, 0, len(ts_cam) - 1)
        right = np.clip(pos, 0, len(ts_cam) - 1)
        dt_left = np.abs(ts_cam[left] - ts_ref)
        dt_right = np.abs(ts_cam[right] - ts_ref)
        use_right = dt_right < dt_left
        best_idx = np.where(use_right, right, left)
        best_dt = np.where(use_right, dt_right, dt_left)

        valid_mask &= (best_dt <= max_dt_ns)
        matched_idx_by_svo[p] = best_idx
        dt_by_svo[p] = best_dt

    kept_ref = np.where(valid_mask)[0]
    keep_sets = {p: set(matched_idx_by_svo[p][kept_ref].tolist()) for p in svo_paths}
    print(f"[Sync] Kept {len(kept_ref)}/{len(ts_ref)} synchronized frames (ref {camera_label(serial_by_svo[ref_svo])})")
    if len(kept_ref) > 0:
        for p in svo_paths:
            if p == ref_svo:
                continue
            dts_ms = dt_by_svo[p][kept_ref] / 1e6
            print(f"  {camera_label(serial_by_svo[p])} sync offset: mean={dts_ms.mean():.3f} ms, p95={np.percentile(dts_ms, 95):.3f} ms, max={dts_ms.max():.3f} ms")
    return keep_sets, rot_imu_by_serial


def process_svo(svo_path, outdir, depth_min, depth_max, enable_pose, body_format,
                zed360_calib=None, keep_indices=None):
    os.makedirs(outdir, exist_ok=True)
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.depth_mode             = sl.DEPTH_MODE.NEURAL_PLUS
    init.coordinate_units       = sl.UNIT.METER
    init.coordinate_system      = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_minimum_distance = depth_min
    init.depth_maximum_distance = depth_max
    init.depth_stabilization    = True
    init.sdk_verbose            = 0
    # svo_real_time_mode left at its default (False) -- replay as fast as NEURAL inference allows.

    zed = sl.Camera()
    with _suppress_c_output():
        status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Could not open {svo_path}: {repr(status)}")
        sys.exit(1)

    serial = int(zed.get_camera_information().serial_number)
    poses_by_serial, ref_serial, calib_filename = zed360_calib if zed360_calib else (None, None, None)

    if poses_by_serial is not None and serial in poses_by_serial:
        T_world_from_cam = poses_by_serial[serial]
        T_world_from_ref = poses_by_serial[ref_serial]
        T_ref_from_cam = np.linalg.inv(T_world_from_ref) @ T_world_from_cam
        frame_metadata = dict(
            pointcloud_frame="RIGHT_HANDED_Y_UP",
            coordinate_units="metres",
            calibration_available=True,
            zed360_calibration_available=True,
            calibration_file=calib_filename,
            camera_name=camera_name(serial),
            camera_serial=np.int64(serial),
            reference_name=camera_name(ref_serial),
            reference_serial=np.int64(ref_serial),
            T_world_from_camera=T_world_from_cam,
            T_ref_from_camera=T_ref_from_cam,
        )
        print(f"[Calibration] ZED360 ({calib_filename}) matched {camera_label(serial)} (ref {camera_label(ref_serial)})")
    else:
        calibration_path = os.path.join(outdir, "calibration.json")
        try:
            calibration = load_calibration(calibration_path, serial) if os.path.exists(calibration_path) else None
        except Exception:
            zed.close()
            raise
        frame_metadata = calibration_metadata(calibration)
        frame_metadata["camera_name"] = camera_name(serial)
        frame_metadata["camera_serial"] = np.int64(serial)
        print("[Calibration] " + ("Loaded; XYZ remains camera-space, world transform saved in metadata" if calibration else "Absent: camera-space baseline"))

    body_runtime = None
    n_kp = None
    if enable_pose:
        zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        body_param = sl.BodyTrackingParameters()
        body_param.enable_tracking     = True
        body_param.enable_body_fitting = True
        body_param.detection_model     = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        body_param.body_format, n_kp   = BODY_FORMAT_OPTIONS[body_format]
        zed.enable_body_tracking(body_param)
        body_runtime = sl.BodyTrackingRuntimeParameters()
        body_runtime.detection_confidence_threshold = 40

    display_res = get_display_resolution(zed)
    H, W = display_res.height, display_res.width

    camera_config = zed.get_camera_information().camera_configuration
    calib = camera_config.calibration_parameters.left_cam
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy
    camera_fps = camera_config.fps

    # Precomputed once: X/Y are fully determined by (row, col, depth, fx, fy, cx, cy),
    # so depth -> XYZ (RIGHT_HANDED_Y_UP: forward objects are negative Z) is folded
    # directly into this loop instead of a separate depth.bin -> pointcloud.bin pass.
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u_off = (cols - cx).astype(np.float32)
    v_off = (rows - cy).astype(np.float32)

    n_target = len(keep_indices) if keep_indices is not None else zed.get_svo_number_of_frames()
    print(f"[Input]      {svo_path}")
    print(f"[Resolution] {W} x {H}   [Depth range] {depth_min:.2f} - {depth_max:.2f} m")
    print(f"[Frames]     {n_target}")

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 95
    runtime.measure3D_reference_frame = sl.REFERENCE_FRAME.CAMERA

    skip_runtime = sl.RuntimeParameters()
    skip_runtime.enable_depth = False

    pc_path = os.path.join(outdir, "pointcloud.bin")
    pc_f    = open(pc_path, "wb")
    writer  = create_video_writer(display_res, outdir, camera_fps)

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    bodies    = sl.Bodies() if enable_pose else None
    timestamps_ns = []
    raw_idx = 0
    written_idx = 0
    xyz        = np.empty((H, W, 3), dtype=np.float32)
    nan_frame  = np.full((H, W, 3), np.nan, dtype=np.float32)
    blank_rgb  = np.zeros((H, W, 3), np.uint8)

    pose_ids, pose_kp3d, pose_conf, pose_np = [], [], [], []

    def append_empty_pose():
        pose_ids.append(np.full(MAX_PERSONS, -1, dtype=np.int32))
        pose_kp3d.append(np.full((MAX_PERSONS, n_kp, 3), np.nan, dtype=np.float32))
        pose_conf.append(np.full(MAX_PERSONS, np.nan, dtype=np.float32))
        pose_np.append(0)

    try:
        while True:
            if keep_indices is not None and written_idx >= n_target:
                break

            if keep_indices is not None and raw_idx not in keep_indices:
                err = zed.grab(skip_runtime)
                if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    break
                raw_idx += 1
                continue

            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break

            if err != sl.ERROR_CODE.SUCCESS:
                print(f"\n[Warning] grab() returned {repr(err)} at frame {raw_idx}, writing blank/NaN frame")
                pc_f.write(nan_frame.tobytes())
                if writer is not None:
                    writer.write(blank_rgb)
                timestamps_ns.append(timestamps_ns[-1] if timestamps_ns else 0)
                if enable_pose:
                    append_empty_pose()
                raw_idx += 1
                written_idx += 1
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, display_res)
            if enable_pose:
                zed.retrieve_bodies(bodies, body_runtime)

            rgb_bgr = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
            if writer is not None:
                writer.write(rgb_bgr)

            d = depth_mat.get_data().astype(np.float32)
            xyz[..., 0] = u_off / fx * d
            xyz[..., 1] = -(v_off / fy) * d
            xyz[..., 2] = -d
            pc_f.write(xyz.tobytes())
            timestamps_ns.append(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

            if enable_pose:
                ids  = np.full(MAX_PERSONS, -1, dtype=np.int32)
                kp3d = np.full((MAX_PERSONS, n_kp, 3), np.nan, dtype=np.float32)
                conf = np.full(MAX_PERSONS, np.nan, dtype=np.float32)
                for slot, body in enumerate(bodies.body_list[:MAX_PERSONS]):
                    ids[slot]  = body.id
                    kp3d[slot] = np.array(body.keypoint, dtype=np.float32)
                    conf[slot] = body.confidence
                pose_ids.append(ids)
                pose_kp3d.append(kp3d)
                pose_conf.append(conf)
                pose_np.append(len(bodies.body_list))

            raw_idx += 1
            written_idx += 1
            if written_idx % 30 == 0:
                print(f"\r[Processing] {written_idx}/{n_target} frames", end='', flush=True)

    finally:
        pc_f.close()
        if writer is not None:
            writer.release()
        image_mat.free(sl.MEM.CPU)
        depth_mat.free(sl.MEM.CPU)
        if enable_pose:
            zed.disable_body_tracking()
            zed.disable_positional_tracking()
        with _suppress_c_output():
            zed.close()

    print(f"\r[Processing] {written_idx}/{n_target} frames -- done")

    meta_path = os.path.join(outdir, "pointcloud_meta.npz")
    np.savez_compressed(
        meta_path,
        **frame_metadata,
        timestamps_ns = np.array(timestamps_ns, dtype=np.int64),
        height        = np.int32(H),
        width         = np.int32(W),
        fx            = np.float32(fx),
        fy            = np.float32(fy),
        cx            = np.float32(cx),
        cy            = np.float32(cy),
        depth_min     = np.float32(depth_min),
        depth_max     = np.float32(depth_max),
        svo_source    = os.path.basename(svo_path),
    )

    pc_mb = os.path.getsize(pc_path) / 1e6
    print(f"[Saved]      {pc_path}  ({pc_mb:.1f} MB, {written_idx} frames)")
    print(f"[Saved]      {meta_path}")

    if enable_pose:
        pose_path = os.path.join(outdir, "pose.npz")
        np.savez_compressed(
            pose_path,
            **frame_metadata,
            timestamps_ns   = np.array(timestamps_ns, dtype=np.int64),
            n_persons       = np.array(pose_np, dtype=np.int32),
            person_ids      = np.stack(pose_ids),
            keypoints_3d    = np.stack(pose_kp3d),
            body_confidence = np.stack(pose_conf),
            body_format     = np.int32(n_kp),
            svo_source      = os.path.basename(svo_path),
        )
        print(f"[Saved]      {pose_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded .svo2 offline through NEURAL depth mode, writing an organized per-pixel XYZ point cloud.")
    parser.add_argument('path', nargs='?', default=None,
                         help="svo_<ts>/ folder or recording.svo2 file. Omit for a file picker.")
    parser.add_argument('--depth-min', type=float, default=0.25, help="Minimum depth in metres (default 0.25)")
    parser.add_argument('--depth-max', type=float, default=5.0, help="Maximum depth in metres (default 3.0)")
    parser.add_argument('--pose', action='store_true',
                         help="Enable ZED body tracking / pose.npz output (off by default)")
    parser.add_argument('--body-format', choices=list(BODY_FORMAT_OPTIONS), default='BODY_38',
                         help="ZED body tracking keypoint format (default BODY_38)")
    args = parser.parse_args()

    inputs = resolve_svo_inputs(args.path)
    if not inputs:
        print("[Cancelled] No SVO selected.")
        sys.exit(1)

    poses_by_serial, override_gravity_by_serial, ref_serial, calib_filename = load_latest_zed360_calibration()

    if len(inputs) > 1:
        keep_sets, rot_imu_by_serial = compute_synced_frame_indices([p for p, _ in inputs], ref_serial=ref_serial)
        if poses_by_serial is not None:
            for sn, rot_imu in rot_imu_by_serial.items():
                if sn in poses_by_serial and not override_gravity_by_serial.get(sn, False):
                    # Per sl/Fusion.hpp: when override_gravity == false, Pose_abs = Pose_rel * Rot_IMU_camera
                    poses_by_serial[sn] = poses_by_serial[sn] @ rot_imu
    else:
        keep_sets = {inputs[0][0]: None}

    zed360_calib = (poses_by_serial, ref_serial, calib_filename) if poses_by_serial is not None else None

    for idx, (svo_path, outdir) in enumerate(inputs, 1):
        if len(inputs) > 1:
            print(f"\n=== Camera {idx}/{len(inputs)}: {os.path.basename(outdir)} ({os.path.basename(svo_path)}) ===")
        process_svo(
            svo_path, outdir, args.depth_min, args.depth_max,
            enable_pose=args.pose, body_format=args.body_format,
            zed360_calib=zed360_calib, keep_indices=keep_sets[svo_path],
        )


if __name__ == "__main__":
    main()
