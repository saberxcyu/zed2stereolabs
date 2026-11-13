import sys
import os
import time
import argparse

import numpy as np
import tkinter as tk
from tkinter import filedialog

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)


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
    if not os.path.exists(combined_path):
        print(f"[Error] {combined_path} not found -- run combine_data.py first.")
        sys.exit(1)

    data = np.load(combined_path)
    points       = data['points']        # (total_N, 3) float32  XYZ, no color
    frame_counts = data['frame_counts']  # (T,) int32
    timestamps   = data['timestamps']    # (T,) float64
    if points.shape[0] == 0 or frame_counts.shape[0] == 0:
        print(f"[Error] {combined_path} has zero points/frames.")
        sys.exit(1)

    print(f"[Loaded] {combined_path}")
    return points, frame_counts, timestamps


def main():
    parser = argparse.ArgumentParser(
        description="Play back a recording's combined_data.npz filtered point cloud in 3D, "
                    "paced by its recorded timestamps.")
    parser.add_argument('path', nargs='?', default=None,
                         help="Recording folder (containing combined_data.npz). Omit for a folder picker.")
    args = parser.parse_args()

    outdir = resolve_folder(args.path)
    if outdir is None:
        print("[Cancelled] No folder selected.")
        sys.exit(1)

    points, frame_counts, timestamps = load_combined(outdir)

    # Unset WAYLAND_DISPLAY before importing open3d so GLFW uses X11/XWayland -- native
    # Wayland uses EGL, which GLEW (Open3D's GL loader) cannot initialize. Must happen
    # before the import, same convention as tools/get_pc/view_pc.py and _pc_analyze.py.
    os.environ.pop('WAYLAND_DISPLAY', None)
    os.environ.setdefault('DISPLAY', ':0')
    import open3d as o3d
    import matplotlib.cm as cm

    T       = len(frame_counts)
    offsets = np.concatenate([[0], np.cumsum(frame_counts.astype(np.int64))])

    # No RGB is stored in combined_data.npz (see visualize.py), so color by depth instead,
    # same jet_r (red=near, blue=far) convention _pc_analyze.py uses for pc.npz. Z = -depth
    # in this RIGHT_HANDED_Y_UP recording. Global min/max over the whole clip (not per-frame)
    # so color is directly comparable across frames.
    depth = -points[:, 2].astype(np.float64)
    depth_min, depth_max = float(depth.min()), float(depth.max())
    depth_range = max(depth_max - depth_min, 1e-9)

    print(f"[Loaded]   {T} frames  |  {len(points):,} total points")
    print(f"[Duration] {timestamps[-1] - timestamps[0]:.2f} s")
    print("Controls: Space=play/pause  <-/->=step frame  Q=quit")

    state = {'frame': 0, 'playing': False, 'last_frame_time': 0.0}
    pcd = o3d.geometry.PointCloud()

    def _load_frame(i):
        s, e = int(offsets[i]), int(offsets[i + 1])
        pts  = points[s:e]
        t    = np.clip((depth[s:e] - depth_min) / depth_range, 0.0, 1.0)
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(cm.jet_r(t)[:, :3])

    def _print_status(i):
        print(f"\rFrame {i + 1}/{T}  {timestamps[i] - timestamps[0]:.2f}s  pts={frame_counts[i]:,}    ",
              end='', flush=True)

    _load_frame(0)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"Combined Point Cloud  |  {T} frames  -  Space=play  <-/->=step  Q=quit",
        width=1280,
        height=720,
    )
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    vis.reset_view_point(True)

    def on_space(vis):
        state['playing'] = not state['playing']
        if state['playing']:
            state['last_frame_time'] = time.monotonic()
        return False

    def on_left(vis):
        state['playing'] = False
        i = max(0, state['frame'] - 1)
        state['frame'] = i
        _load_frame(i)
        vis.update_geometry(pcd)
        _print_status(i)
        return True

    def on_right(vis):
        state['playing'] = False
        i = min(T - 1, state['frame'] + 1)
        state['frame'] = i
        _load_frame(i)
        vis.update_geometry(pcd)
        _print_status(i)
        return True

    def on_q(vis):
        vis.close()
        return False

    vis.register_key_callback(32,       on_space)
    vis.register_key_callback(263,      on_left)
    vis.register_key_callback(262,      on_right)
    vis.register_key_callback(ord('Q'), on_q)
    vis.register_key_callback(ord('q'), on_q)

    def animation_cb(vis):
        if not state['playing']:
            return False

        now = time.monotonic()
        i   = state['frame']

        if i + 1 < T:
            target_dt = float(timestamps[i + 1] - timestamps[i])
        else:
            target_dt = float(timestamps[-1] - timestamps[-2]) if T > 1 else 1.0 / 15.0

        if now - state['last_frame_time'] < target_dt:
            return False

        next_i = (i + 1) % T
        state['frame']           = next_i
        state['last_frame_time'] = now
        _load_frame(next_i)
        vis.update_geometry(pcd)
        _print_status(next_i)
        return False

    vis.register_animation_callback(animation_cb)
    vis.run()
    vis.destroy_window()
    print()


if __name__ == "__main__":
    main()
