import sys
import os
import argparse

import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pyzed.sl as sl

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

BODY_FORMAT_ENUM = {18: sl.BODY_FORMAT.BODY_18, 34: sl.BODY_FORMAT.BODY_34, 38: sl.BODY_FORMAT.BODY_38}
BONES_BY_FORMAT = {
    sl.BODY_FORMAT.BODY_18: sl.BODY_18_BONES,
    sl.BODY_FORMAT.BODY_34: sl.BODY_34_BONES,
    sl.BODY_FORMAT.BODY_38: sl.BODY_38_BONES,
}
PARTS_BY_FORMAT = {
    sl.BODY_FORMAT.BODY_18: sl.BODY_18_PARTS,
    sl.BODY_FORMAT.BODY_34: sl.BODY_34_PARTS,
    sl.BODY_FORMAT.BODY_38: sl.BODY_38_PARTS,
}


def resolve_folder(arg):
    if arg is None:
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select recording folder", initialdir=TOOLS_RECORDING_DIR)
        root.destroy()
        return folder or None
    if os.path.isdir(arg):
        return arg
    print(f"[Error] Not a directory: {arg}")
    return None


def load_combined(outdir):
    combined_path = os.path.join(outdir, "combined_data.npz")
    meta_path = os.path.join(outdir, "pointcloud_meta.npz")
    if not os.path.exists(combined_path) or not os.path.exists(meta_path):
        print(f"[Error] {combined_path} / {meta_path} not found -- run combine_data.py first.")
        sys.exit(1)

    cd = np.load(combined_path)
    if cd['points'].shape[0] == 0 or cd['frame_counts'].shape[0] == 0:
        print(f"[Error] {combined_path} has zero points/frames.")
        sys.exit(1)

    meta_npz = np.load(meta_path)
    meta = {k: meta_npz[k].item() for k in ('height', 'width', 'fx', 'fy', 'cx', 'cy')}

    data = {
        'points':       cd['points'],
        'frame_counts': cd['frame_counts'],
        'timestamps':   cd['timestamps'],
        'meta':         meta,
    }
    if 'joints_3d' in cd.files:
        data['joints_3d']        = cd['joints_3d']
        data['joint_confidence'] = cd['joint_confidence']
        data['body_format']      = int(cd['body_format'])

    print(f"[Loaded] {combined_path}")
    print(f"[Loaded] {meta_path}")
    return data


def project_xyz(x, y, z, fx, fy, cx, cy):
    """Inverse of process.py's depth->XYZ projection (RIGHT_HANDED_Y_UP, forward = -Z):
    X=(col-cx)/fx*depth, Y=-(row-cy)/fy*depth, Z=-depth."""
    depth = -z
    with np.errstate(divide='ignore', invalid='ignore'):
        col = cx + x * fx / depth
        row = cy - y * fy / depth
    valid = np.isfinite(row) & np.isfinite(col) & (depth > 0)
    return row, col, valid


def project_frame_pixels(points, offsets, idx, meta):
    """Projects only one frame's slice of `points` (a few hundred thousand rows at
    most, not the whole clip) to integer (row, col) pixel coordinates, clipped to
    bounds. Returns (row_i, col_i, z) for the VALID points of that frame only."""
    H, W = meta['height'], meta['width']
    s, e = offsets[idx], offsets[idx + 1]
    if e <= s:
        empty_i = np.empty(0, dtype=np.int64)
        return empty_i, empty_i, np.empty(0, dtype=np.float32)

    pts = points[s:e]
    row, col, valid = project_xyz(pts[:, 0], pts[:, 1], pts[:, 2], meta['fx'], meta['fy'], meta['cx'], meta['cy'])
    row_i = np.round(row).astype(np.int64)
    col_i = np.round(col).astype(np.int64)
    valid = valid & (row_i >= 0) & (row_i < H) & (col_i >= 0) & (col_i < W)

    return row_i[valid], col_i[valid], pts[valid, 2]


def compute_pixel_ranges(points, offsets, meta):
    """Streaming, one-frame-at-a-time pass computing each reprojected pixel location's
    own Z min/max across the whole clip (relative local motion, e.g. breathing amplitude
    at a given spot, rather than one global min/max across every point in the clip).
    'Same point over time' == same reprojected (row, col) pixel location -- there is no
    true 3D point tracking/correspondence available here, just like select_roi.py/plot.py's
    fixed-pixel ROI convention.

    Deliberately frame-by-frame rather than projecting the whole (total_N, 3) `points`
    array at once: a real clip can be hundreds of millions of points (e.g. 518M seen in
    practice), and the several full-length float/int temporaries a whole-clip vectorized
    pass needs (row, col, row_i, col_i, key, ...) can multiply that into tens of GB and
    exhaust RAM. Per-frame slices are only a few hundred thousand points each, so peak
    memory here stays close to just the base `points` array already in RAM."""
    H, W = meta['height'], meta['width']
    n_keys = H * W
    min_per_pixel = np.full(n_keys, np.inf, dtype=np.float32)
    max_per_pixel = np.full(n_keys, -np.inf, dtype=np.float32)

    n_frames = len(offsets) - 1
    for idx in range(n_frames):
        row_i, col_i, z = project_frame_pixels(points, offsets, idx, meta)
        if row_i.shape[0] == 0:
            continue
        key = row_i * W + col_i
        np.minimum.at(min_per_pixel, key, z)
        np.maximum.at(max_per_pixel, key, z)
        if (idx + 1) % 100 == 0 or idx == n_frames - 1:
            print(f"\r[Building] pixel Z ranges {idx + 1}/{n_frames} frames", end='', flush=True)
    print()

    denom = max_per_pixel - min_per_pixel
    denom[denom < 1e-9] = 1e-9  # pixel seen in only one frame: no variation to show, falls back to t=0 (black)
    return min_per_pixel, denom


BLACK_BASE = np.array([0, 0, 0], dtype=np.float32)
BLUE_PEAK  = np.array([20, 20, 220], dtype=np.float32)


def t_to_color(t):
    """Black (t=0, pixel at its own exhale/farthest extreme -> Z minimum) fading to blue
    (t=1, pixel at its own inhale/closest-to-camera extreme -> Z maximum). Z = -depth in
    this RIGHT_HANDED_Y_UP recording, so breathing in (chest moving toward the camera,
    depth decreasing) means Z increasing toward that pixel's max -- i.e. blue marks
    breathe-in, black is the neutral/resting base."""
    return (BLACK_BASE + t[:, None] * (BLUE_PEAK - BLACK_BASE)).astype(np.uint8)


def render_frame_canvas(idx, points, offsets, meta, min_per_pixel, denom):
    H, W = meta['height'], meta['width']
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    row_i, col_i, z = project_frame_pixels(points, offsets, idx, meta)
    if row_i.shape[0] == 0:
        return canvas
    key = row_i * W + col_i
    t = np.clip((z - min_per_pixel[key]) / denom[key], 0.0, 1.0)
    colors = t_to_color(t)
    canvas[row_i, col_i] = colors
    return canvas


def get_bone_pairs(body_format_int):
    body_format = BODY_FORMAT_ENUM.get(body_format_int)
    if body_format is None:
        print(f"[Warning] Unrecognized body_format {body_format_int} -- skipping skeleton bones.")
        return []
    return [(bone[0].value, bone[1].value) for bone in BONES_BY_FORMAT[body_format]]


def get_joint_names(body_format_int, n_kp):
    body_format = BODY_FORMAT_ENUM.get(body_format_int)
    if body_format is None:
        return [str(i) for i in range(n_kp)]
    parts = PARTS_BY_FORMAT[body_format]
    return [parts(i).name for i in range(n_kp)]


def project_joints_frame(jf, meta):
    """Same projection as project_xyz, but keeps row/col as float NaN (not int-rounded)
    for absent joints -- matplotlib scatter/line artists skip NaN coordinates natively."""
    x, y, z = jf[:, 0], jf[:, 1], jf[:, 2]
    row, col, valid = project_xyz(x, y, z, meta['fx'], meta['fy'], meta['cx'], meta['cy'])
    row = np.where(valid, row, np.nan)
    col = np.where(valid, col, np.nan)
    return row, col


def build_viewer(data, outdir):
    meta = data['meta']
    H, W = meta['height'], meta['width']
    frame_counts = data['frame_counts']
    timestamps = data['timestamps']
    T = timestamps.shape[0]
    offsets = np.concatenate([[0], np.cumsum(frame_counts)])

    points = data['points']
    min_per_pixel, denom = compute_pixel_ranges(points, offsets, meta)

    has_pose = 'joints_3d' in data
    bones = get_bone_pairs(data['body_format']) if has_pose else []

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(f"Combined Data Viewer - {outdir}")
    plt.subplots_adjust(bottom=0.15)
    ax.set_axis_off()

    im = ax.imshow(render_frame_canvas(0, points, offsets, meta, min_per_pixel, denom))

    joint_scatter = ax.scatter([], [], s=30, facecolors='white', edgecolors='black',
                                linewidths=0.8, zorder=3)
    bone_lines = [ax.plot([], [], '-', color='white', linewidth=1.5, zorder=2)[0] for _ in bones]
    joint_names = get_joint_names(data['body_format'], data['joints_3d'].shape[1]) if has_pose else []
    joint_labels = [ax.text(0, 0, name, fontsize=6, color='black', ha='left', va='center',
                             zorder=4, visible=False,
                             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
                     for name in joint_names]

    ax_slider = fig.add_axes([0.10, 0.05, 0.72, 0.04])
    ax_play   = fig.add_axes([0.85, 0.035, 0.08, 0.06])
    slider    = Slider(ax_slider, '', 0, T - 1, valinit=0, valstep=1)
    play_btn  = Button(ax_play, '>')

    def update_overlay(idx):
        if not has_pose:
            return
        jf = data['joints_3d'][idx]
        row, col = project_joints_frame(jf, meta)
        joint_scatter.set_offsets(np.column_stack([col, row]))
        for line, (i, j) in zip(bone_lines, bones):
            line.set_data([col[i], col[j]], [row[i], row[j]])
        for k, label in enumerate(joint_labels):
            if np.isfinite(row[k]) and np.isfinite(col[k]):
                label.set_position((col[k] + 6, row[k]))
                label.set_visible(True)
            else:
                label.set_visible(False)

    def on_slider(val):
        idx = int(slider.val)
        im.set_data(render_frame_canvas(idx, points, offsets, meta, min_per_pixel, denom))
        update_overlay(idx)
        t_s = timestamps[idx] - timestamps[0]
        ax.set_title(f"Frame {idx}/{T - 1}   t = {t_s:.2f} s", fontsize=9)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    playing = [False]
    timer = fig.canvas.new_timer(interval=33)

    def tick():
        if not playing[0]:
            return
        idx = int(slider.val)
        if idx >= T - 1:
            playing[0] = False
            play_btn.label.set_text('>')
            fig.canvas.draw_idle()
            timer.stop()
            return
        slider.set_val(idx + 1)

    timer.add_callback(tick)

    def on_play(event):
        playing[0] = not playing[0]
        if playing[0]:
            if int(slider.val) >= T - 1:
                slider.set_val(0)
            play_btn.label.set_text('||')
            timer.start()
        else:
            play_btn.label.set_text('>')
        fig.canvas.draw_idle()

    play_btn.on_clicked(on_play)

    def on_key(event):
        if event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)

    on_slider(0)
    print("Drag the slider to navigate frames  |  '>' to play/pause  |  'q' to quit.\n")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive viewer for a recording's combined_data.npz (masked point cloud + pose).")
    parser.add_argument('path', nargs='?', default=None,
                         help="Recording folder (containing combined_data.npz). Omit for a folder picker.")
    args = parser.parse_args()

    outdir = resolve_folder(args.path)
    if outdir is None:
        print("[Cancelled] No folder selected.")
        sys.exit(1)

    data = load_combined(outdir)
    build_viewer(data, outdir)


if __name__ == "__main__":
    main()
