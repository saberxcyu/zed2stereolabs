import sys
import os
import shutil
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, RectangleSelector, Button
import pyzed.sl as sl

RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(RECORDING_DIR, exist_ok=True)

WINDOW_NAME      = "ZED Depth Viewer  |  RGB (left)  +  Depth (right)"
POSE_WINDOW_NAME = "ZED Pose Tracker  |  RGB (left)  +  Skeleton (right)"

RESOLUTION_OPTIONS = [
    ("HD2K",   sl.RESOLUTION.HD2K,   2208, 1242, 15),
    ("HD1080", sl.RESOLUTION.HD1080, 1920, 1080, 30),
    ("HD720",  sl.RESOLUTION.HD720,  1280,  720, 30),
    ("VGA",    sl.RESOLUTION.VGA,     672,  376, 30),
]

BODY_FORMAT_OPTIONS = [
    ("BODY_18  -  18 keypoints", sl.BODY_FORMAT.BODY_18, 18),
    ("BODY_34  -  34 keypoints", sl.BODY_FORMAT.BODY_34, 34),
    ("BODY_38  -  38 keypoints", sl.BODY_FORMAT.BODY_38, 38),
]

# One distinct BGR color per tracked person (cycled by person ID)
BODY_COLORS = [
    (0, 255, 0), (0, 128, 255), (255, 0, 128), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (128, 255, 0), (0, 0, 255),
    (255, 128, 0), (128, 0, 255),
]

# Same colors in RGB order (B,G,R -> R,G,B) normalized to [0,1] for matplotlib
BODY_COLORS_RGB = [(r / 255, g / 255, b / 255) for (b, g, r) in BODY_COLORS]

# Keypoint names per body format (index = keypoint enum value)
KEYPOINT_NAMES = {
    18: ['Nose', 'Neck', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
         'L_Shoulder', 'L_Elbow', 'L_Wrist',
         'R_Hip', 'R_Knee', 'R_Ankle',
         'L_Hip', 'L_Knee', 'L_Ankle',
         'R_Eye', 'L_Eye', 'R_Ear', 'L_Ear'],
    34: ['Pelvis', 'NavalSpine', 'ChestSpine', 'Neck',
         'L_Clavicle', 'L_Shoulder', 'L_Elbow', 'L_Wrist',
         'L_Hand', 'L_HandTip', 'L_Thumb',
         'R_Clavicle', 'R_Shoulder', 'R_Elbow', 'R_Wrist',
         'R_Hand', 'R_HandTip', 'R_Thumb',
         'L_Hip', 'L_Knee', 'L_Ankle', 'L_Foot',
         'R_Hip', 'R_Knee', 'R_Ankle', 'R_Foot',
         'Head', 'Nose', 'L_Eye', 'L_Ear', 'R_Eye', 'R_Ear',
         'L_Heel', 'R_Heel'],
    38: ['Pelvis', 'Spine1', 'Spine2', 'Spine3', 'Neck',
         'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear',
         'L_Clavicle', 'R_Clavicle',
         'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow',
         'L_Wrist', 'R_Wrist',
         'L_Hip', 'R_Hip', 'L_Knee', 'R_Knee',
         'L_Ankle', 'R_Ankle',
         'L_BigToe', 'R_BigToe', 'L_SmallToe', 'R_SmallToe',
         'L_Heel', 'R_Heel',
         'L_Thumb4', 'R_Thumb4', 'L_Index1', 'R_Index1',
         'L_Middle4', 'R_Middle4', 'L_Pinky1', 'R_Pinky1'],
}


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def normalize_depth_to_colormap(depth_data, min_d, max_d):
    """Valid depth in [min_d, max_d] -> JET colormap (near=red, far=blue). NaN/inf -> black."""
    nan_mask  = ~np.isfinite(depth_data)
    working   = np.clip(depth_data, min_d, max_d)
    range_span = max_d - min_d
    if range_span == 0.0:
        inverted = np.full_like(working, 0.5)
    else:
        normalized = (working - min_d) / range_span
        inverted   = 1.0 - normalized
    scaled    = (inverted * 255).astype(np.uint8)
    colorized = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    colorized[nan_mask] = [0, 0, 0]
    return colorized


# ---------------------------------------------------------------------------
# Startup dialogs (tkinter)
# ---------------------------------------------------------------------------

def _center_window(root, w, h):
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")


def show_mode_dialog():
    """
    2x2 mode selection dialog.
    Returns 'record', 'analyze', 'pose', 'pose_analyze', or None if closed.
    """
    result = [None]

    root = tk.Tk()
    root.title("ZED Tool")
    root.resizable(False, False)
    _center_window(root, 520, 280)

    tk.Label(root, text="Select Mode", font=("Helvetica", 16, "bold")).pack(pady=(16, 10))

    def _btn(parent, text, mode, bg):
        def cmd():
            result[0] = mode
            root.destroy()
        tk.Button(parent, text=text, width=14, height=2, font=("Helvetica", 12),
                  command=cmd, bg=bg, fg="white").pack(side=tk.LEFT, padx=14, pady=8)

    depth_frame = tk.LabelFrame(root, text="Depth", font=("Helvetica", 10, "bold"), padx=6)
    depth_frame.pack(padx=20, pady=4, fill='x')
    _btn(depth_frame, "Record",  'record',  "#2c5f2e")
    _btn(depth_frame, "Analyze", 'analyze', "#1a3a5c")

    pose_frame = tk.LabelFrame(root, text="Pose", font=("Helvetica", 10, "bold"), padx=6)
    pose_frame.pack(padx=20, pady=4, fill='x')
    _btn(pose_frame, "Record",  'pose',         "#5a2d82")
    _btn(pose_frame, "Analyze", 'pose_analyze', "#2d5a82")

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result[0]


# ---------------------------------------------------------------------------
# Capture-mode helpers
# ---------------------------------------------------------------------------

def show_record_settings_dialog():
    """Combined resolution + depth range dialog. Returns (sl.RESOLUTION, min_depth, max_depth) or None."""
    result = [None]

    root = tk.Tk()
    root.title("Recording Settings")
    root.resizable(False, False)
    _center_window(root, 600, 280)

    tk.Label(root, text="Recording Settings",
             font=("Helvetica", 14, "bold")).pack(pady=(16, 10))

    panels = tk.Frame(root)
    panels.pack(padx=20, fill='x')

    # Left: resolution radio buttons
    left = tk.LabelFrame(panels, text="Resolution", font=("Helvetica", 10, "bold"), padx=10, pady=8)
    left.pack(side=tk.LEFT, fill='y', padx=(0, 10))

    choice = tk.IntVar(value=0)  # default: HD2K
    for i, (name, _, w, h, fps) in enumerate(RESOLUTION_OPTIONS):
        tk.Radiobutton(left, text=f"{name}   {w} x {h}  @  {fps} fps",
                       variable=choice, value=i, font=("Helvetica", 10)).pack(anchor='w')

    # Right: depth range entries
    right = tk.LabelFrame(panels, text="Depth Range", font=("Helvetica", 10, "bold"), padx=14, pady=8)
    right.pack(side=tk.LEFT, fill='both', expand=True)

    tk.Label(right, text="Min depth (m):").grid(row=0, column=0, sticky='w', pady=6)
    min_var = tk.StringVar(value="0.5")
    tk.Entry(right, textvariable=min_var, width=8).grid(row=0, column=1, padx=8)

    tk.Label(right, text="Max depth (m):").grid(row=1, column=0, sticky='w', pady=6)
    max_var = tk.StringVar(value="1.5")
    tk.Entry(right, textvariable=max_var, width=8).grid(row=1, column=1, padx=8)

    err_label = tk.Label(right, text="", fg="red", font=("Helvetica", 9))
    err_label.grid(row=2, column=0, columnspan=2, pady=(4, 0))

    def on_start():
        try:
            mn = float(min_var.get())
            mx = float(max_var.get())
        except ValueError:
            err_label.config(text="Enter valid numbers.")
            return
        if mn <= 0 or mx <= 0:
            err_label.config(text="Values must be positive.")
            return
        if mn >= mx:
            err_label.config(text="Min must be less than Max.")
            return
        result[0] = (RESOLUTION_OPTIONS[choice.get()], mn, mx)
        root.destroy()

    tk.Button(root, text="Start Recording", width=16, font=("Helvetica", 11),
              command=on_start, bg="#2c5f2e", fg="white").pack(pady=14)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    if result[0] is None:
        return None
    (name, res, w, h, fps), mn, mx = result[0]
    print(f"[Resolution] {name}  ({w} x {h} @ {fps} fps)")
    print(f"[Depth] range [{mn}, {mx}] m")
    return res, mn, mx


def show_pose_settings_dialog():
    """Resolution + keypoint format dialog for pose mode. Returns (sl.RESOLUTION, sl.BODY_FORMAT) or None."""
    result = [None]

    root = tk.Tk()
    root.title("Pose Settings")
    root.resizable(False, False)
    _center_window(root, 580, 260)

    tk.Label(root, text="Pose Settings", font=("Helvetica", 14, "bold")).pack(pady=(16, 10))

    panels = tk.Frame(root)
    panels.pack(padx=20, fill='x')

    # Left: resolution radio buttons
    left = tk.LabelFrame(panels, text="Resolution", font=("Helvetica", 10, "bold"), padx=10, pady=8)
    left.pack(side=tk.LEFT, fill='y', padx=(0, 10))

    res_choice = tk.IntVar(value=0)  # default: HD2K
    for i, (name, _, w, h, fps) in enumerate(RESOLUTION_OPTIONS):
        tk.Radiobutton(left, text=f"{name}   {w} x {h}  @  {fps} fps",
                       variable=res_choice, value=i, font=("Helvetica", 10)).pack(anchor='w')

    # Right: keypoint format radio buttons
    right = tk.LabelFrame(panels, text="Keypoint Format", font=("Helvetica", 10, "bold"), padx=14, pady=8)
    right.pack(side=tk.LEFT, fill='both', expand=True)

    fmt_choice = tk.IntVar(value=0)  # default: BODY_18
    for i, (label, _, n_kp) in enumerate(BODY_FORMAT_OPTIONS):
        tk.Radiobutton(right, text=label, variable=fmt_choice, value=i,
                       font=("Helvetica", 10)).pack(anchor='w', pady=4)

    def on_start():
        _, res_enum, _, _, _ = RESOLUTION_OPTIONS[res_choice.get()]
        _, fmt_enum, _       = BODY_FORMAT_OPTIONS[fmt_choice.get()]
        result[0] = (res_enum, fmt_enum)
        root.destroy()

    tk.Button(root, text="Start Pose Tracking", width=18, font=("Helvetica", 11),
              command=on_start, bg="#5a2d82", fg="white").pack(pady=14)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result[0]


def get_display_resolution(zed):
    return zed.get_camera_information().camera_configuration.resolution


def create_video_writer(display_res, outdir):
    video_path = os.path.join(outdir, "video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_path, fourcc, 30.0, (display_res.width, display_res.height), True)
    if not writer.isOpened():
        print(f"[Warning] Could not open VideoWriter for {video_path}. Video recording disabled.")
        return None
    return writer


def draw_status(frame, msg, color, recording):
    bar_h   = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, msg, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    if recording:
        cv2.circle(frame, (frame.shape[1] - 22, 20), 8, (0, 0, 220), -1)


def draw_skeleton_overlay(frame, bodies, body_format, image_scale):
    """Draw skeleton bones and joints on a BGR frame in-place."""
    bones = {
        sl.BODY_FORMAT.BODY_18: sl.BODY_18_BONES,
        sl.BODY_FORMAT.BODY_34: sl.BODY_34_BONES,
        sl.BODY_FORMAT.BODY_38: sl.BODY_38_BONES,
    }[body_format]

    H, W = frame.shape[:2]
    for body in bodies.body_list:
        color = BODY_COLORS[body.id % len(BODY_COLORS)]
        kps   = body.keypoint_2d

        for bone in bones:
            a  = kps[bone[0].value]
            b  = kps[bone[1].value]
            ax = int(a[0] * image_scale[0])
            ay = int(a[1] * image_scale[1])
            bx = int(b[0] * image_scale[0])
            by = int(b[1] * image_scale[1])
            if 0 < ax < W and 0 < ay < H and 0 < bx < W and 0 < by < H:
                cv2.line(frame, (ax, ay), (bx, by), color, 2, cv2.LINE_AA)

        for kp in kps:
            px = int(kp[0] * image_scale[0])
            py = int(kp[1] * image_scale[1])
            if 0 < px < W and 0 < py < H:
                cv2.circle(frame, (px, py), 4, color, -1)

        label_x = int(kps[0][0] * image_scale[0])
        label_y = int(kps[0][1] * image_scale[1]) - 10
        cv2.putText(frame, f"ID {body.id}", (max(0, label_x), max(0, label_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_skeleton_from_saved(rgb_frame, kp2d_frame, person_ids_frame, body_format, image_scale):
    """Draw skeleton from saved keypoints_2d onto an RGB numpy frame. Returns new RGB array."""
    bones = {
        sl.BODY_FORMAT.BODY_18: sl.BODY_18_BONES,
        sl.BODY_FORMAT.BODY_34: sl.BODY_34_BONES,
        sl.BODY_FORMAT.BODY_38: sl.BODY_38_BONES,
    }[body_format]

    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    H, W  = frame.shape[:2]

    for slot, pid in enumerate(person_ids_frame):
        if pid < 0:
            break
        color = BODY_COLORS[int(pid) % len(BODY_COLORS)]
        kps   = kp2d_frame[slot]  # (n_kp, 2)

        for bone in bones:
            a = kps[bone[0].value]
            b = kps[bone[1].value]
            if not (np.isfinite(a[0]) and np.isfinite(b[0])):
                continue
            ax_ = int(a[0] * image_scale[0]); ay_ = int(a[1] * image_scale[1])
            bx_ = int(b[0] * image_scale[0]); by_ = int(b[1] * image_scale[1])
            if 0 < ax_ < W and 0 < ay_ < H and 0 < bx_ < W and 0 < by_ < H:
                cv2.line(frame, (ax_, ay_), (bx_, by_), color, 2, cv2.LINE_AA)

        for kp in kps:
            if not np.isfinite(kp[0]):
                continue
            px = int(kp[0] * image_scale[0]); py = int(kp[1] * image_scale[1])
            if 0 < px < W and 0 < py < H:
                cv2.circle(frame, (px, py), 4, color, -1)

        root_kp = kps[0]
        if np.isfinite(root_kp[0]):
            lx = max(0, int(root_kp[0] * image_scale[0]))
            ly = max(0, int(root_kp[1] * image_scale[1]) - 10)
            cv2.putText(frame, f"ID {pid}", (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def draw_axes_gizmo(rgb_frame):
    """Overlay XYZ coordinate axes in the bottom-left corner of an RGB frame."""
    frame = rgb_frame.copy()
    H, W  = frame.shape[:2]
    ox, oy = 55, H - 55
    alen   = 40
    dz     = int(alen * 0.65)

    # X -- right, red
    cv2.arrowedLine(frame, (ox, oy), (ox + alen, oy),
                    (255, 60, 60), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(frame, 'X', (ox + alen + 4, oy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 60), 2, cv2.LINE_AA)

    # Y -- up; world-up = decreasing pixel y, green
    cv2.arrowedLine(frame, (ox, oy), (ox, oy - alen),
                    (60, 220, 60), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(frame, 'Y', (ox - 16, oy - alen - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 2, cv2.LINE_AA)

    # Z -- into screen, blue; isometric convention: diagonal toward lower-left
    cv2.arrowedLine(frame, (ox, oy), (ox - dz, oy + dz),
                    (80, 150, 255), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(frame, 'Z', (ox - dz - 16, oy + dz + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 150, 255), 2, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Analyze-mode helpers
# ---------------------------------------------------------------------------

def preload_rgb_frames(cap, n, h, w):
    rgb_list = []
    for _ in range(n):
        ret, frame = cap.read()
        rgb_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else np.zeros((h, w, 3), np.uint8))
    return rgb_list


def build_depth_rgba(frame_data, min_depth, max_depth, cmap, valid_alpha=0.55, nan_alpha=0.80):
    """
    Build an (H, W, 4) float32 RGBA overlay.
    Valid depth -> jet_r colormap, semi-transparent. NaN/inf -> opaque black.
    """
    f         = frame_data.astype(np.float32)
    nan_mask  = ~np.isfinite(f)
    f_clipped = np.clip(f, min_depth, max_depth)
    range_span = max_depth - min_depth
    if range_span > 0:
        normalized = (f_clipped - min_depth) / range_span
    else:
        normalized = np.full_like(f_clipped, 0.5)
    rgba             = cmap(normalized).astype(np.float32)
    rgba[..., 3]     = valid_alpha
    rgba[nan_mask, :3] = 0.0
    rgba[nan_mask, 3]  = nan_alpha
    return rgba


def print_point_table(cx, cy, timestamps, depth_series):
    print(f"\nSelection: Point (x={cx}, y={cy})")
    print(f"{'Frame':>6}  {'Time(s)':>8}  {'Depth(m)':>9}")
    print(f"{'------':>6}  {'-------':>8}  {'--------':>9}")
    for i, (t, d) in enumerate(zip(timestamps, depth_series)):
        depth_str = f"{float(d):.3f}" if np.isfinite(d) else "  NaN  "
        print(f"{i:>6}  {t:>8.3f}  {depth_str:>9}")


def print_region_table(x1, y1, x2, y2, timestamps, region_frames):
    flat        = region_frames.reshape(len(region_frames), -1)
    mean_series = np.nanmean(flat, axis=1)
    std_series  = np.nanstd(flat, axis=1)
    print(f"\nSelection: Region x=[{x1},{x2}] y=[{y1},{y2}] - mean depth")
    print(f"{'Frame':>6}  {'Time(s)':>8}  {'Mean(m)':>8}  {'Std(m)':>7}")
    print(f"{'------':>6}  {'-------':>8}  {'-------':>8}  {'------':>7}")
    for i, (t, m, s) in enumerate(zip(timestamps, mean_series, std_series)):
        mean_str = f"{float(m):.3f}" if np.isfinite(m) else "  NaN  "
        std_str  = f"{float(s):.3f}" if np.isfinite(s) else "  NaN  "
        print(f"{i:>6}  {t:>8.3f}  {mean_str:>8}  {std_str:>7}")
