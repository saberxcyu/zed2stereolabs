import sys
import os
import shutil
import time
import argparse
from datetime import datetime

import cv2
import numpy as np
import pyzed.sl as sl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _common import RESOLUTION_OPTIONS, get_display_resolution, draw_status

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

STEREO_WINDOW_NAME = "ZED Stereo Viewer  |  Left Rectified  +  Right Rectified"

SVO_COMPRESSION_OPTIONS = {
    "H264":     sl.SVO_COMPRESSION_MODE.H264,
    "H265":     sl.SVO_COMPRESSION_MODE.H265,
    "LOSSLESS": sl.SVO_COMPRESSION_MODE.LOSSLESS,
}


def record_mode(resolution_option, compression_type=sl.SVO_COMPRESSION_MODE.H264):
    """
    Open ZED, display side-by-side rectified frames, record SVO with 's'.
    Returns 'back' (q pressed) or 'quit' (window closed).
    """
    name, resolution, w, h, fps = resolution_option

    init = sl.InitParameters()
    init.camera_resolution = resolution
    init.depth_mode = sl.DEPTH_MODE.NONE

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        return 'back'

    display_res = get_display_resolution(zed)
    camera_fps  = float(zed.get_camera_information().camera_configuration.fps)
    print(f"[Resolution] {name}  ({display_res.width} x {display_res.height} @ {camera_fps:.0f} fps)")
    print("Press 's' to start/stop recording | 'q' to quit")

    left_mat  = sl.Mat()
    right_mat = sl.Mat()

    recording = False
    ts        = None
    outdir    = None
    svo_path  = None
    rec_start = None

    status_msg   = 'Press [s] to start recording - [q] to quit'
    status_color = (200, 200, 200)

    cv2.namedWindow(STEREO_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(STEREO_WINDOW_NAME, 1920, 540)

    runtime_params = sl.RuntimeParameters()
    exit_reason    = 'back'

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_mat,  sl.VIEW.LEFT,  sl.MEM.CPU, display_res)
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT, sl.MEM.CPU, display_res)

                left_bgr  = cv2.cvtColor(left_mat.get_data(),  cv2.COLOR_BGRA2BGR)
                right_bgr = cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2BGR)

                display_frame = np.concatenate([left_bgr, right_bgr], axis=1)
                draw_status(display_frame, status_msg, status_color, recording)
                display_small = cv2.resize(display_frame, (1920, 540), interpolation=cv2.INTER_AREA)
                cv2.imshow(STEREO_WINDOW_NAME, display_small)
                key = cv2.waitKey(1) & 0xFF

                if recording:
                    elapsed = time.monotonic() - rec_start
                    print(f"\r[Recording] {elapsed:6.1f}s", end='', flush=True)

                if cv2.getWindowProperty(STEREO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = 'quit'
                    break

                if key == ord('s'):
                    if not recording:
                        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outdir   = os.path.join(TOOLS_RECORDING_DIR, f"svo_{ts}")
                        os.makedirs(outdir, exist_ok=True)
                        svo_path = os.path.join(outdir, "recording.svo2")
                        rec_params = sl.RecordingParameters()
                        rec_params.video_filename   = svo_path
                        rec_params.compression_mode = compression_type
                        err = zed.enable_recording(rec_params)
                        if err != sl.ERROR_CODE.SUCCESS:
                            print(f"[Error]     Could not start SVO recording: {repr(err)}")
                            shutil.rmtree(outdir, ignore_errors=True)
                            outdir   = None
                            svo_path = None
                        else:
                            recording    = True
                            rec_start    = time.monotonic()
                            print(f"[Recording] {svo_path}")
                            status_msg   = 'Recording SVO - press [s] to stop'
                            status_color = (0, 80, 255)
                    else:
                        zed.disable_recording()
                        recording = False
                        elapsed   = time.monotonic() - rec_start
                        print()  # end the elapsed-time \r line
                        size_mb = os.path.getsize(svo_path) / 1e6
                        print(f"[Saved]     {size_mb:.1f} MB, {elapsed:.1f}s -> {svo_path}")
                        status_msg   = f'Saved: svo_{ts}/ - press [s] to record again'
                        status_color = (0, 210, 0)
                        outdir       = None
                        svo_path     = None
                        rec_start    = None

                elif key == ord('q'):
                    exit_reason = 'back'
                    break

    finally:
        if recording and outdir:
            zed.disable_recording()
            shutil.rmtree(outdir, ignore_errors=True)
            print("\n[Discarded] Recording cancelled - no file saved.")
        left_mat.free(sl.MEM.CPU)
        right_mat.free(sl.MEM.CPU)
        zed.close()
        cv2.destroyAllWindows()

    return exit_reason


def main():
    parser = argparse.ArgumentParser(description="Capture stereo video via the ZED SDK, recording to SVO2 on 's'.")
    parser.add_argument('--resolution', choices=[opt[0] for opt in RESOLUTION_OPTIONS], default='HD2K',
                         help="Camera resolution (default HD2K)")
    parser.add_argument('--compression', choices=list(SVO_COMPRESSION_OPTIONS.keys()), default='H264',
                         help="SVO compression mode (default H264)")
    args = parser.parse_args()

    res_option        = next(opt for opt in RESOLUTION_OPTIONS if opt[0] == args.resolution)
    compression_type  = SVO_COMPRESSION_OPTIONS[args.compression]

    record_mode(res_option, compression_type=compression_type)


if __name__ == "__main__":
    main()
