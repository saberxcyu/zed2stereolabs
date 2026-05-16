import sys
import os
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
import pyzed.sl as sl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _common import RESOLUTION_OPTIONS, _center_window, get_display_resolution, draw_status

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

STEREO_WINDOW_NAME = "ZED Stereo Viewer  |  Left Rectified  +  Right Rectified"

SVO_COMPRESSION_OPTIONS = [
    ("H264     lossy  (default)",       sl.SVO_COMPRESSION_MODE.H264),
    ("H265     lossy  (better ratio)",  sl.SVO_COMPRESSION_MODE.H265),
    ("Lossless",                        sl.SVO_COMPRESSION_MODE.LOSSLESS),
]


def show_stereo_settings_dialog():
    """Resolution and compression picker. Returns (resolution_option, sl.SVO_COMPRESSION_MODE) or None."""
    result = [None]

    root = tk.Tk()
    root.title("Stereo Video Settings")
    root.resizable(False, False)
    _center_window(root, 520, 280)

    tk.Label(root, text="Stereo Video Settings",
             font=("Helvetica", 14, "bold")).pack(pady=(16, 10))

    panels = tk.Frame(root)
    panels.pack(padx=20, fill='x')

    left = tk.LabelFrame(panels, text="Resolution",
                         font=("Helvetica", 10, "bold"), padx=10, pady=8)
    left.pack(side=tk.LEFT, fill='y')

    choice = tk.IntVar(value=0)  # default: HD2K
    for i, (name, _, w, h, fps) in enumerate(RESOLUTION_OPTIONS):
        tk.Radiobutton(left, text=f"{name}   {w} x {h}  @  {fps} fps",
                       variable=choice, value=i,
                       font=("Helvetica", 10)).pack(anchor='w')

    right = tk.LabelFrame(panels, text="Compression",
                          font=("Helvetica", 10, "bold"), padx=10, pady=8)
    right.pack(side=tk.LEFT, fill='y', padx=(12, 0))

    comp_choice = tk.IntVar(value=0)  # default: H264
    for i, (label, _) in enumerate(SVO_COMPRESSION_OPTIONS):
        tk.Radiobutton(right, text=label, variable=comp_choice, value=i,
                       font=("Helvetica", 10)).pack(anchor='w')

    def on_start():
        _, comp_type = SVO_COMPRESSION_OPTIONS[comp_choice.get()]
        result[0] = (RESOLUTION_OPTIONS[choice.get()], comp_type)
        root.destroy()

    tk.Button(root, text="Start Recording", width=16, font=("Helvetica", 11),
              command=on_start, bg="#2c5f2e", fg="white").pack(pady=14)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result[0]


def stereo_video_mode(resolution_option, compression_type=sl.SVO_COMPRESSION_MODE.H264):
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
    print("Press 's' to start/stop recording | 'q' to return to menu")

    left_mat  = sl.Mat()
    right_mat = sl.Mat()

    recording = False
    ts        = None
    svo_path  = None

    status_msg   = 'Press [s] to start recording - [q] to return to menu'
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

                if cv2.getWindowProperty(STEREO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = 'quit'
                    break

                if key == ord('s'):
                    if not recording:
                        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                        svo_path = os.path.join(TOOLS_RECORDING_DIR, f"stereo_{ts}.svo2")
                        rec_params = sl.RecordingParameters()
                        rec_params.video_filename   = svo_path
                        rec_params.compression_mode = compression_type
                        err = zed.enable_recording(rec_params)
                        if err != sl.ERROR_CODE.SUCCESS:
                            print(f"[Error]     Could not start SVO recording: {repr(err)}")
                            svo_path = None
                        else:
                            recording    = True
                            print(f"[Recording] {svo_path}")
                            status_msg   = 'Recording SVO - press [s] to stop'
                            status_color = (0, 80, 255)
                    else:
                        zed.disable_recording()
                        recording = False
                        size_mb   = os.path.getsize(svo_path) / 1e6
                        print(f"[Saved]     {size_mb:.1f} MB -> {svo_path}")
                        status_msg   = f'Saved: stereo_{ts}.svo2 - press [s] to record again'
                        status_color = (0, 210, 0)
                        svo_path     = None

                elif key == ord('q'):
                    exit_reason = 'back'
                    break

    finally:
        if recording and svo_path:
            zed.disable_recording()
            try: os.remove(svo_path)
            except OSError: pass
            print("[Discarded] Recording cancelled - no file saved.")
        left_mat.free(sl.MEM.CPU)
        right_mat.free(sl.MEM.CPU)
        zed.close()
        cv2.destroyAllWindows()

    return exit_reason


def main():
    while True:
        settings = show_stereo_settings_dialog()
        if settings is None:       # [x] on dialog -> exit
            break
        res_option, compression_type = settings
        result = stereo_video_mode(res_option, compression_type=compression_type)
        if result == 'quit':       # [x] on camera window -> exit
            break
        # result == 'back' ([q] pressed) -> loop back to dialog


if __name__ == "__main__":
    main()
