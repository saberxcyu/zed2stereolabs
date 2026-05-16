import os
import shutil
import math
import configparser
import glob
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
CALIB_DIR           = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration')
STEREO_WINDOW_NAME  = "ZED Stereo Viewer  |  Left Rectified  +  Right Rectified"

os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

# (name, per-eye width, per-eye height, fps, calibration section suffix)
# UVC combined width = per-eye width * 2 (left|right side-by-side)
# Calibration section suffix maps to LEFT_CAM_<suffix> / RIGHT_CAM_<suffix> in the .conf file
# and to <KEY>_<suffix> keys inside [STEREO]
RESOLUTION_OPTIONS = [
    ("HD2K",   2208, 1242, 15, "HD2K"),
    ("HD1080", 1920, 1080, 30, "HD"),
    ("HD720",  1280,  720, 30, "720"),
    ("VGA",     672,  376, 30, "VGA"),
]


# ---------------------------------------------------------------------------
# UI helpers (self-contained; _common.py cannot be imported - it pulls pyzed)
# ---------------------------------------------------------------------------

def _center_window(root, w, h):
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")


def draw_status(frame, msg, color, recording):
    bar_h   = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, msg, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 22, 20), 8, (0, 0, 220), -1)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def find_calibration_file():
    """Return path to first *.conf in CALIB_DIR, or None."""
    matches = glob.glob(os.path.join(CALIB_DIR, '*.conf'))
    if not matches:
        return None
    path = matches[0]
    print(f"[Calibration] Using {os.path.basename(path)}")
    return path


def load_calibration(conf_path, calib_suffix):
    """
    Parse a Stereolabs .conf calibration file and return OpenCV stereo parameters.

    Returns (K_l, D_l, K_r, D_r, R, T) where:
      K_l/K_r  - 3x3 camera matrices
      D_l/D_r  - distortion vectors [k1,k2,p1,p2,k3]
      R        - 3x3 rotation from left to right camera
      T        - 3x1 translation in mm (baseline along X)
    """
    cfg = configparser.ConfigParser()
    cfg.read(conf_path)

    def cam_params(section):
        s = cfg[section]
        K = np.array([
            [float(s['fx']), 0.0,            float(s['cx'])],
            [0.0,            float(s['fy']), float(s['cy'])],
            [0.0,            0.0,            1.0           ],
        ], dtype=np.float64)
        D = np.array([
            float(s.get('k1', 0)), float(s.get('k2', 0)),
            float(s.get('p1', 0)), float(s.get('p2', 0)),
            float(s.get('k3', 0)),
        ], dtype=np.float64)
        return K, D

    K_l, D_l = cam_params(f'LEFT_CAM_{calib_suffix}')
    K_r, D_r = cam_params(f'RIGHT_CAM_{calib_suffix}')

    st  = cfg['STEREO']
    suf = calib_suffix.lower()

    baseline = float(st['baseline'])                    # mm
    cv_ang   = float(st.get(f'cv_{suf}',   0.0))       # rotation around Y (radians)
    rx_ang   = float(st.get(f'rx_{suf}',   0.0))       # rotation around X (radians)
    rz_ang   = float(st.get(f'rz_{suf}',   0.0))       # rotation around Z (radians)
    ty       = float(st.get(f'ty_{suf}',   0.0))       # translation Y (mm)
    tz       = float(st.get(f'tz_{suf}',   0.0))       # translation Z (mm)

    # Build rotation matrix: Rz * Ry(convergence) * Rx
    def Rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)

    def Ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)

    def Rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

    R = Rz(rz_ang) @ Ry(cv_ang) @ Rx(rx_ang)
    T = np.array([[baseline], [ty], [tz]], dtype=np.float64)

    return K_l, D_l, K_r, D_r, R, T


def compute_rectify_maps(K_l, D_l, K_r, D_r, R, T, w, h):
    """
    Compute per-eye remap arrays for stereo rectification.
    alpha=0 crops output to valid pixels only (no black borders).
    Returns ((map1_l, map2_l), (map1_r, map2_r)).
    """
    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)
    return (map1_l, map2_l), (map1_r, map2_r)


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

def show_stereo_settings_dialog():
    """Resolution picker. Returns selected RESOLUTION_OPTIONS entry (5-tuple) or None."""
    result = [None]

    root = tk.Tk()
    root.title("Stereo Video Settings")
    root.resizable(False, False)
    _center_window(root, 380, 280)

    tk.Label(root, text="Stereo Video Settings",
             font=("Helvetica", 14, "bold")).pack(pady=(16, 10))

    panels = tk.Frame(root)
    panels.pack(padx=20, fill='x')

    frame_left = tk.LabelFrame(panels, text="Resolution",
                               font=("Helvetica", 10, "bold"), padx=10, pady=8)
    frame_left.pack(side=tk.LEFT, fill='y')

    choice = tk.IntVar(value=0)  # default: HD2K
    for i, (name, w, h, fps, _) in enumerate(RESOLUTION_OPTIONS):
        tk.Radiobutton(frame_left, text=f"{name}   {w} x {h}  @  {fps} fps",
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


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------

def stereo_video_mode(resolution_option, map_l, map_r):
    """
    Open the ZED as a UVC camera, display side-by-side rectified frames, record with 's'.
    Returns 'back' (q pressed) or 'quit' (window closed).
    """
    name, eye_w, eye_h, fps, _ = resolution_option

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[Error] Could not open /dev/video0. Check that the ZED is connected.")
        return 'back'

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  eye_w * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, eye_h)
    cap.set(cv2.CAP_PROP_FPS,          float(fps))

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    if actual_w != eye_w * 2 or actual_h != eye_h:
        print(f"[Warning] Requested {eye_w*2}x{eye_h} but camera returned {actual_w}x{actual_h}.")
        print("          Resolution mismatch - rectification maps will not match. Aborting.")
        cap.release()
        return 'back'

    print(f"[Resolution] {name}  ({eye_w} x {eye_h} @ {actual_fps:.0f} fps per eye)")
    print("Press 's' to start/stop recording | 'q' to return to menu")

    recording    = False
    left_writer  = None
    right_writer = None
    outdir       = None
    ts           = None

    status_msg   = 'Press [s] to start recording - [q] to return to menu'
    status_color = (200, 200, 200)

    cv2.namedWindow(STEREO_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(STEREO_WINDOW_NAME, 1920, 540)

    exit_reason = 'back'

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Warning] Failed to grab frame.")
                continue

            # Split combined side-by-side frame into left and right eyes
            left_raw  = frame[:, :eye_w]
            right_raw = frame[:, eye_w:]

            # Apply stereo rectification
            left_rect  = cv2.remap(left_raw,  map_l[0], map_l[1], cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_raw, map_r[0], map_r[1], cv2.INTER_LINEAR)

            if recording:
                if left_writer  is not None: left_writer.write(left_rect)
                if right_writer is not None: right_writer.write(right_rect)

            display_frame = np.concatenate([left_rect, right_rect], axis=1)
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
                    frame_size   = (eye_w, eye_h)
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
        cap.release()
        if left_writer  is not None: left_writer.release()
        if right_writer is not None: right_writer.release()
        if recording and outdir is not None and os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
            print("[Discarded] Recording cancelled - no files saved.")
        cv2.destroyAllWindows()

    return exit_reason


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    conf_path = find_calibration_file()
    if conf_path is None:
        print("[Error] No calibration file found.")
        print(f"        Place your camera's SN*.conf file in: {CALIB_DIR}/")
        print("        See calibration/README.txt for instructions.")
        return

    while True:
        settings = show_stereo_settings_dialog()
        if settings is None:       # [x] on dialog -> exit
            break

        _, eye_w, eye_h, _, calib_suffix = settings

        try:
            K_l, D_l, K_r, D_r, R, T = load_calibration(conf_path, calib_suffix)
            map_l, map_r = compute_rectify_maps(K_l, D_l, K_r, D_r, R, T, eye_w, eye_h)
        except (KeyError, ValueError) as e:
            print(f"[Error] Failed to load calibration for {calib_suffix}: {e}")
            break

        result = stereo_video_mode(settings, map_l, map_r)
        if result == 'quit':       # [x] on camera window -> exit
            break
        # result == 'back' ([q] pressed) -> loop back to dialog


if __name__ == "__main__":
    main()
