import os
import time
import numpy as np
import matplotlib.cm as _cm


def pc_analyze_mode(folder):
    # Unset WAYLAND_DISPLAY before importing open3d so GLFW uses X11/XWayland.
    os.environ.pop('WAYLAND_DISPLAY', None)
    os.environ.setdefault('DISPLAY', ':0')
    import open3d as o3d

    npz_path = os.path.join(folder, 'pc.npz')
    if not os.path.exists(npz_path):
        print(f"[Error] No pc.npz found in {folder}")
        return 'back'

    data         = np.load(npz_path)
    points       = data['points']        # (total_N, 6) float32
    frame_counts = data['frame_counts']  # (T,) int32
    timestamps   = data['timestamps']    # (T,) float64
    depth_min_m  = float(data['depth_min']) if 'depth_min' in data else 0.3
    depth_max_m  = float(data['depth_max']) if 'depth_max' in data else 3.0

    T       = len(frame_counts)
    offsets = np.concatenate([[0], np.cumsum(frame_counts.astype(np.int64))])

    print(f"[Loaded]   {T} frames  |  {len(points):,} total points")
    print(f"[Duration] {timestamps[-1]:.2f} s")
    print("Controls: Space=play/pause  ←/→=step frame  Q=back to menu  [X]=quit")

    def _depth_colors(pts):
        z = pts[:, 2].astype(np.float64)
        t = np.clip((-z - depth_min_m) / (depth_max_m - depth_min_m), 0.0, 1.0)
        return _cm.jet_r(t)[:, :3]

    state = {
        'frame':           0,
        'playing':         False,
        'last_frame_time': 0.0,
        'exit':            'quit',
    }

    pcd = o3d.geometry.PointCloud()

    def _load_frame(i):
        s, e = int(offsets[i]), int(offsets[i + 1])
        pts  = points[s:e]
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(_depth_colors(pts))

    def _print_status(i):
        print(f"\rFrame {i + 1}/{T}  {timestamps[i]:.2f}s  pts={frame_counts[i]:,}    ",
              end='', flush=True)

    _load_frame(0)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"ZED Point Cloud  |  {T} frames  —  Space=play  ←/→=step  Q=back",
        width=1280,
        height=720,
    )
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size       = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    vis.reset_view_point(True)

    # ------------------------------------------------------------------ keys
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
        state['exit'] = 'back'
        vis.close()
        return False

    vis.register_key_callback(32,       on_space)
    vis.register_key_callback(263,      on_left)
    vis.register_key_callback(262,      on_right)
    vis.register_key_callback(ord('Q'), on_q)
    vis.register_key_callback(ord('q'), on_q)

    # --------------------------------------------------------- animation loop
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

    return state['exit']
