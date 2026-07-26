import sys
import os
import argparse

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import pyzed.sl as sl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _common import get_display_resolution, create_video_writer

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)


def resolve_svo_path(arg):
    """Accepts a svo_<ts>/ folder, a recording.svo2 file, or None (Tk file picker).
    Returns (svo_path, outdir) or (None, None) if cancelled."""
    if arg is None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select recording.svo2",
            initialdir=TOOLS_RECORDING_DIR,
            filetypes=[("SVO2 files", "*.svo2")],
        )
        root.destroy()
        if not path:
            return None, None
        return path, os.path.dirname(path)

    if os.path.isdir(arg):
        svo_path = os.path.join(arg, "recording.svo2")
        if not os.path.exists(svo_path):
            print(f"[Error] No recording.svo2 in {arg}")
            return None, None
        return svo_path, arg

    if os.path.isfile(arg):
        return arg, os.path.dirname(os.path.abspath(arg))

    print(f"[Error] Path not found: {arg}")
    return None, None


def process_svo(svo_path, outdir, depth_min, depth_max):
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.depth_mode             = sl.DEPTH_MODE.NEURAL
    init.coordinate_units       = sl.UNIT.METER
    init.coordinate_system      = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_minimum_distance = depth_min
    init.depth_maximum_distance = depth_max
    init.depth_stabilization    = True
    # svo_real_time_mode left at its default (False) -- replay as fast as NEURAL inference allows.

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Could not open {svo_path}: {repr(status)}")
        sys.exit(1)

    display_res = get_display_resolution(zed)
    H, W = display_res.height, display_res.width

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    fx, fy, cx, cy = calib.fx, calib.fy, calib.cx, calib.cy

    n_expected = zed.get_svo_number_of_frames()
    print(f"[Input]      {svo_path}")
    print(f"[Resolution] {W} x {H}   [Depth range] {depth_min:.2f} - {depth_max:.2f} m")
    print(f"[Frames]     {n_expected} (expected)")

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 95

    depth_path = os.path.join(outdir, "depth.bin")
    depth_f    = open(depth_path, "wb")
    writer     = create_video_writer(display_res, outdir)

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    timestamps_ns = []
    frame_idx = 0
    nan_frame  = np.full((H, W), np.nan, dtype=np.float32)
    blank_rgb  = np.zeros((H, W, 3), np.uint8)

    try:
        while True:
            err = zed.grab(runtime)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"\n[Warning] grab() returned {repr(err)} at frame {frame_idx}, writing blank/NaN frame")
                depth_f.write(nan_frame.tobytes())
                if writer is not None:
                    writer.write(blank_rgb)
                timestamps_ns.append(timestamps_ns[-1] if timestamps_ns else 0)
                frame_idx += 1
                continue

            zed.retrieve_image(image_mat, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, display_res)

            rgb_bgr = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
            if writer is not None:
                writer.write(rgb_bgr)
            depth_f.write(depth_mat.get_data().astype(np.float32).tobytes())
            timestamps_ns.append(zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"\r[Processing] {frame_idx}/{n_expected} frames", end='', flush=True)

    finally:
        depth_f.close()
        if writer is not None:
            writer.release()
        image_mat.free(sl.MEM.CPU)
        depth_mat.free(sl.MEM.CPU)
        zed.close()

    print(f"\r[Processing] {frame_idx}/{n_expected} frames -- done")

    meta_path = os.path.join(outdir, "depth_meta.npz")
    np.savez_compressed(
        meta_path,
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

    depth_mb = os.path.getsize(depth_path) / 1e6
    print(f"[Saved]      {depth_path}  ({depth_mb:.1f} MB, {frame_idx} frames)")
    print(f"[Saved]      {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded .svo2 offline through NEURAL depth mode.")
    parser.add_argument('path', nargs='?', default=None,
                         help="svo_<ts>/ folder or recording.svo2 file. Omit for a file picker.")
    parser.add_argument('--depth-min', type=float, default=0.25, help="Minimum depth in metres (default 0.25)")
    parser.add_argument('--depth-max', type=float, default=1.5, help="Maximum depth in metres (default 1.5)")
    args = parser.parse_args()

    svo_path, outdir = resolve_svo_path(args.path)
    if svo_path is None:
        print("[Cancelled] No SVO selected.")
        sys.exit(1)

    process_svo(svo_path, outdir, args.depth_min, args.depth_max)


if __name__ == "__main__":
    main()
