import sys
import os
import re
import tempfile

# Force GLFW to use X11/XWayland instead of native Wayland.
# GLEW (used by Open3D) requires a GLX context; native Wayland uses EGL
# which GLEW cannot initialize. Must be set before open3d is imported.
os.environ.pop('WAYLAND_DISPLAY', None)
os.environ.setdefault('DISPLAY', ':0')

import tkinter as tk
from tkinter import filedialog

import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _common import _center_window

RECORDINGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'recordings'
)


def _load_ply(path):
    """Load a PLY file, stripping the last vertex to work around the ZED SDK
    PLY writer bug that truncates the final vertex's color data."""
    with open(path, 'rb') as f:
        data = f.read()

    end_marker = b'end_header\n'
    idx = data.find(end_marker)
    if idx == -1:
        return o3d.io.read_point_cloud(path)

    header = data[:idx].decode('ascii')
    body = data[idx + len(end_marker):]
    patched = re.sub(
        r'element vertex (\d+)',
        lambda m: f'element vertex {int(m.group(1)) - 1}',
        header,
    )

    tmp = tempfile.NamedTemporaryFile(suffix='.ply', delete=False)
    try:
        tmp.write(patched.encode('ascii'))
        tmp.write(end_marker)
        tmp.write(body)
        tmp.close()
        pcd = o3d.io.read_point_cloud(tmp.name)
    finally:
        os.unlink(tmp.name)

    return pcd


def pick_ply_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select Point Cloud",
        initialdir=RECORDINGS_DIR,
        filetypes=[("PLY files", "*.ply"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


def main():
    path = pick_ply_file()
    if path is None:
        return

    print(f"[Loading]  {path}")
    pcd = _load_ply(path)

    n_points = len(pcd.points)
    has_color = pcd.has_colors()
    print(f"[Points]   {n_points:,}")
    print(f"[Colors]   {'yes' if has_color else 'no'}")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=os.path.basename(path),
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
