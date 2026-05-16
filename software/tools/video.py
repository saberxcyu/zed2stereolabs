import sys
import os
import shutil
from datetime import datetime
import time

import cv2
import numpy as np
import tkinter as tk
import pyzed.sl as sl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from _common import RESOLUTION_OPTIONS, _center_window, get_display_resolution, draw_status

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

STEREO_WINDOW_NAME = "ZED Stereo Viewer  |  Left Rectified  +  Right Rectified"


def show_stereo_settings_dialog():
    """Resolution picker for stereo recording. Returns (name, sl.RESOLUTION, w, h, fps) or None."""
    result = [None]

    root = tk.Tk()
    root.title("Stereo Video Settings")
    root.resizable(False, False)
    _center_window(root, 380, 280)

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

    def on_start():
        result[0] = RESOLUTION_OPTIONS[choice.get()]
        root.destroy()

    tk.Button(root, text="Start Recording", width=16, font=("Helvetica", 11),
              command=on_start, bg="#2c5f2e", fg="white").pack(pady=14)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result[0]


def stereo_video_mode(resolution_option):
    """
    Open ZED, display side-by-side rectified frames, record with 's'.
    Returns 'back' (q pressed) or 'quit' (window closed).
    """
    name, resolution, w, h, fps = resolution_option

    init = sl.InitParameters()
    init.camera_resolution = resolution
    init.depth_mode = sl.DEPTH_MODE.NONE

    zed = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        return 'back'

    display_res = get_display_resolution(zed)
    print(f"[Resolution] {name}  ({display_res.width} x {display_res.height} @ {fps} fps)")
    print("Press 's' to start/stop recording | 'q' to return to menu")

    left_mat  = sl.Mat()
    right_mat = sl.Mat()

    recording    = False
    left_writer  = None
    right_writer = None
    outdir       = None
    ts           = None

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

                if recording:
                    if left_writer  is not None: left_writer.write(left_bgr)
                    if right_writer is not None: right_writer.write(right_bgr)

                display_frame = np.concatenate([left_bgr, right_bgr], axis=1)
                draw_status(display_frame, status_msg, status_color, recording)
                cv2.imshow(STEREO_WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if cv2.getWindowProperty(STEREO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = 'quit'
                    break

                if key == ord('s'):
                    if not recording:
                        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outdir = os.path.join(TOOLS_RECORDING_DIR, f"stereo_{ts}")
                        os.makedirs(outdir, exist_ok=True)

                        fourcc       = cv2.VideoWriter_fourcc(*'mp4v')
                        frame_size   = (display_res.width, display_res.height)
                        left_writer  = cv2.VideoWriter(
                            os.path.join(outdir, "left.mp4"),  fourcc, float(fps), frame_size, True)
                        right_writer = cv2.VideoWriter(
                            os.path.join(outdir, "right.mp4"), fourcc, float(fps), frame_size, True)

                        if not left_writer.isOpened():
                            print("[Warning] Could not open left VideoWriter. Left video disabled.")
                            left_writer = None
                        if not right_writer.isOpened():
                            print("[Warning] Could not open right VideoWriter. Right video disabled.")
                            right_writer = None

                        recording    = True
                        status_msg   = 'Recording started - press [s] again to stop'
                        status_color = (0, 80, 255)
                        print(f"[Started]   Recording to {outdir}/")

                    else:
                        recording = False
                        if left_writer  is not None:
                            left_writer.release()
                            left_writer = None
                        if right_writer is not None:
                            right_writer.release()
                            right_writer = None

                        saving_frame = display_frame.copy()
                        draw_status(saving_frame, f'Saving to {outdir} ...', (0, 220, 255), False)
                        cv2.imshow(STEREO_WINDOW_NAME, saving_frame)
                        cv2.waitKey(1)

                        left_mb  = os.path.getsize(os.path.join(outdir, "left.mp4"))  / 1e6
                        right_mb = os.path.getsize(os.path.join(outdir, "right.mp4")) / 1e6
                        print(f"[Saved]     {outdir}/  ({left_mb:.1f} MB + {right_mb:.1f} MB)")
                        status_msg   = f'Saved: stereo_{ts} - press [s] to record again'
                        status_color = (0, 210, 0)

                elif key == ord('q'):
                    exit_reason = 'back'
                    break

    finally:
        left_mat.free(sl.MEM.CPU)
        right_mat.free(sl.MEM.CPU)
        if left_writer  is not None: left_writer.release()
        if right_writer is not None: right_writer.release()
        if recording and outdir is not None and os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
            print("[Discarded] Recording cancelled - no files saved.")
        zed.close()
        cv2.destroyAllWindows()

    return exit_reason


def main():
    while True:
        settings = show_stereo_settings_dialog()
        if settings is None:       # [x] on dialog -> exit
            break
        result = stereo_video_mode(settings)
        if result == 'quit':       # [x] on camera window -> exit
            break
        # result == 'back' ([q] pressed) -> loop back to dialog


if __name__ == "__main__":
    main()
