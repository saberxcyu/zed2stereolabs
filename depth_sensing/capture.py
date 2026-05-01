import sys
import argparse
import os
import shutil
import time
from datetime import datetime

import cv2
import numpy as np
import pyzed.sl as sl

# -------------------------------- TO RUN --------------------------------
# Run as -> uv run capture.py --min-depth 0.5 --max-depth 0.8 
# adjust min and max depth range as needed
# ------------------------------------------------------------------------

WINDOW_NAME = "ZED Depth Viewer  |  RGB (left)  +  Depth (right)"

RESOLUTION_OPTIONS = [
    ("HD2K",   sl.RESOLUTION.HD2K,   2208, 1242, 15),
    ("HD1080", sl.RESOLUTION.HD1080, 1920, 1080, 30),
    ("HD720",  sl.RESOLUTION.HD720,  1280,  720, 30),
    ("VGA",    sl.RESOLUTION.VGA,     672,  376, 30),
]


def pick_resolution():
    """Show a static OpenCV menu; user presses 1-4 to select immediately."""
    WIN = "Select Resolution"
    W, H = 480, 190
    canvas = np.zeros((H, W, 3), np.uint8)
    cv2.putText(canvas, "Select resolution  (press 1-4):",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1, cv2.LINE_AA)
    for i, (name, _, w, h, fps) in enumerate(RESOLUTION_OPTIONS):
        line = f"[{i + 1}]  {name:<8} {w} x {h}  @ {fps} fps"
        cv2.putText(canvas, line, (30, 75 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.imshow(WIN, canvas)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('1'), ord('2'), ord('3'), ord('4')):
            idx = key - ord('1')
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    name, res, w, h, fps = RESOLUTION_OPTIONS[idx]
    print(f"[Resolution] {name}  ({w} x {h} @ {fps} fps)")
    return res


def parse_args(init, opt):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith((".svo", ".svo2")):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0:
        ip_str = opt.ip_address
        if ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4 and len(ip_str.split(':')) == 2:
            init.set_from_stream(ip_str.split(':')[0], int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ", ip_str)
        elif ip_str.replace(':', '').replace('.', '').isdigit() and len(ip_str.split('.')) == 4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ", ip_str)
        else:
            print("Unvalid IP format. Using live stream")


def get_display_resolution(zed):
    camera_info = zed.get_camera_information()
    return camera_info.camera_configuration.resolution


def create_video_writer(display_res, outdir):
    video_path = os.path.join(outdir, "video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (display_res.width, display_res.height), True)
    if not writer.isOpened():
        print(f"[Warning] Could not open VideoWriter for {video_path}. Video recording disabled.")
        return None
    return writer


def normalize_depth_to_colormap(depth_data, min_d, max_d):
    """
    - Valid depth in [min_d, max_d] -> JET colormap: near=red, far=blue
    - NaN, +inf, -inf (no data / out of range) -> solid black
    """
    nan_mask = ~np.isfinite(depth_data)  # catch NaN and +/-inf

    # Clip first: +/-inf -> boundary values, NaN passes through as-is
    working = np.clip(depth_data, min_d, max_d)

    range_span = max_d - min_d
    if range_span == 0.0:
        inverted = np.full_like(working, 0.5)
    else:
        normalized = (working - min_d) / range_span   # 0=near, 1=far
        inverted   = 1.0 - normalized                  # flip: 0=far->blue, 1=near->red in JET

    scaled    = (inverted * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    colorized[nan_mask] = [0, 0, 0]   # NaN and +/-inf -> black
    return colorized


def draw_status(frame, msg, color, recording):
    """Draws a semi-transparent status bar at the top of frame (in-place)."""
    bar_h = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, msg, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if recording:
        # Pulsing red dot to indicate active recording
        cv2.circle(frame, (frame.shape[1] - 22, 20), 8, (0, 0, 220), -1)


def main():
    parser = argparse.ArgumentParser(description="ZED depth recorder - saves RGB video + raw depth")
    parser.add_argument('--input_svo_file', type=str, default='',
                        help='Path to an .svo/.svo2 file for playback')
    parser.add_argument('--ip_address', type=str, default='',
                        help='IP address (a.b.c.d:port or a.b.c.d) for stream input')
    parser.add_argument('--mode', type=str, default='depth', choices=['depth', 'disparity'],
                        help='Retrieve depth (meters) or raw disparity (pixels). Default: depth')
    parser.add_argument('--min-depth', type=float, default=0.5,
                        help='Near clip for colormap normalization in meters. Default: 0.5')
    parser.add_argument('--max-depth', type=float, default=3.0,
                        help='Far clip for colormap normalization in meters. Default: 3.0')
    opt = parser.parse_args()

    if len(opt.input_svo_file) > 0 and len(opt.ip_address) > 0:
        print("Specify only --input_svo_file or --ip_address, not both. Exiting.")
        sys.exit(1)

    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    )
    init.depth_minimum_distance = opt.min_depth
    init.depth_maximum_distance = opt.max_depth
    
    # this applies a temporal filter that smoothes out high-frequency sensor noise to help 
    # capture low-frequency respiratory wave (need to be tested)
    init.depth_stabilization = True 

    parse_args(init, opt)
    init.camera_resolution = pick_resolution()


    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        sys.exit(1)

    display_res = get_display_resolution(zed)
    measure     = sl.MEASURE.DEPTH if opt.mode == "depth" else sl.MEASURE.DISPARITY

    print(f"[Mode] {opt.mode.capitalize()} | range [{opt.min_depth}, {opt.max_depth}] m")
    print("Press 's' to start recording | 'q' to quit without saving")

    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    # Recording state
    recording    = False
    writer       = None
    outdir       = None
    depth_frames = []
    timestamps   = []
    t_start      = None

    status_msg   = 'Press [s] to start recording - [q] to quit'
    status_color = (200, 200, 200)  # grey

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1920, 540)

    # adjust model confidence threshold if needed 
    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 95 # this conf_threshold is inverted, 0 is stricter, 100 is more forgiving

    try:
        while True:

            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_mat, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
                zed.retrieve_measure(depth_mat, measure, sl.MEM.CPU, display_res)

                # Prepare RGB frame
                rgb_bgra = image_mat.get_data()
                rgb_bgr  = cv2.resize(
                    cv2.cvtColor(rgb_bgra, cv2.COLOR_BGRA2BGR),
                    (display_res.width, display_res.height),
                    interpolation=cv2.INTER_LINEAR,
                )

                # Accumulate and write only while recording
                if recording:
                    if t_start is None:
                        t_start = time.monotonic()
                    timestamps.append(time.monotonic() - t_start)
                    depth_frames.append(depth_mat.get_data().copy()) # stay at float32
                    if writer is not None:
                        writer.write(rgb_bgr)

                # Build side-by-side live display
                colorized     = normalize_depth_to_colormap(depth_mat.get_data(), opt.min_depth, opt.max_depth)
                display_frame = np.concatenate([rgb_bgr, colorized], axis=1)
                draw_status(display_frame, status_msg, status_color, recording)
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF

                # Check AFTER waitKey - waitKey processes OS events including window close
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key == ord('s'):
                    if not recording:
                        # - Start recording ------------------------------------------
                        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outdir = f"depth_{ts}"
                        os.makedirs(outdir, exist_ok=True)
                        writer       = create_video_writer(display_res, outdir)
                        depth_frames = []
                        timestamps   = []
                        t_start      = None
                        recording    = True
                        status_msg   = 'Recording started - press [s] again to stop'
                        status_color = (0, 80, 255)   # orange-red
                        print(f"[Started]   Recording to {outdir}/")
                    else:
                        # -- Stop recording - show "saving" frame first, then block --
                        recording = False
                        if writer is not None:
                            writer.release()
                            writer = None

                        saving_frame = display_frame.copy()
                        draw_status(saving_frame,
                                    f'Saving {len(depth_frames)} frames to {outdir} ...',
                                    (0, 220, 255), False)
                        cv2.imshow(WINDOW_NAME, saving_frame)
                        cv2.waitKey(1)  # force display refresh before blocking save

                        # saves as npz and compress (works like a dictionary)
                        npz_path = os.path.join(outdir, "depth.npz")
                        np.savez_compressed(
                            npz_path,
                            frames     = np.stack(depth_frames), # 3D array (N, H, W)
                            timestamps = np.array(timestamps, dtype=np.float64), # 1D array (N,)
                            min_depth  = np.float32(opt.min_depth), # single value
                            max_depth  = np.float32(opt.max_depth), # single value
                        )
                        size_mb = os.path.getsize(npz_path) / 1e6
                        print(f"[Saved]     {outdir}/  ({len(depth_frames)} frames, {size_mb:.1f} MB)")
                        status_msg   = f'Saved: {outdir}  ({size_mb:.1f} MB) - press [s] to record again'
                        status_color = (0, 210, 0)   # green
                        depth_frames = []
                        timestamps   = []
                        t_start      = None

                elif key == ord('q'):
                    break


    finally:
        image_mat.free(sl.MEM.CPU)
        depth_mat.free(sl.MEM.CPU)
        if writer is not None:
            writer.release()
        # If 'q' was pressed mid-recording, discard the partial folder
        if recording and outdir is not None and os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
            print("[Discarded] Recording cancelled - no files saved.")
        zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
