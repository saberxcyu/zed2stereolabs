import sys
import os
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


def convert(outdir):
    depth_path = os.path.join(outdir, "depth.bin")
    meta_path  = os.path.join(outdir, "depth_meta.npz")
    if not os.path.exists(depth_path) or not os.path.exists(meta_path):
        print(f"[Error] {depth_path} / {meta_path} not found -- run process.py first.")
        sys.exit(1)

    meta = np.load(meta_path)
    H, W = int(meta['height']), int(meta['width'])
    fx, fy, cx, cy = (float(meta[k]) for k in ('fx', 'fy', 'cx', 'cy'))
    frame_bytes = H * W * 4
    n_frames = os.path.getsize(depth_path) // frame_bytes
    if n_frames == 0:
        print("[Error] depth.bin has zero complete frames.")
        sys.exit(1)

    print(f"[Input]  {depth_path}  ({n_frames} frames, {W}x{H})")

    depth_mm = np.memmap(depth_path, dtype=np.float32, mode='r', shape=(n_frames, H, W))

    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u_off = (cols - cx).astype(np.float32)
    v_off = (rows - cy).astype(np.float32)

    pc_path = os.path.join(outdir, "pointcloud.bin")
    pc_f = open(pc_path, "wb")
    xyz  = np.empty((H, W, 3), dtype=np.float32)

    try:
        for i in range(n_frames):
            d = depth_mm[i]
            # RIGHT_HANDED_Y_UP: X=right, Y=up, Z=backward (forward objects are negative Z)
            xyz[..., 0] = u_off / fx * d
            xyz[..., 1] = -(v_off / fy) * d
            xyz[..., 2] = -d
            pc_f.write(xyz.tobytes())
            if (i + 1) % 30 == 0 or i == n_frames - 1:
                print(f"\r[Converting] {i + 1}/{n_frames} frames", end='', flush=True)
    finally:
        pc_f.close()

    print()

    pc_meta_path = os.path.join(outdir, "pointcloud_meta.npz")
    np.savez_compressed(
        pc_meta_path,
        timestamps_ns = meta['timestamps_ns'],
        height        = np.int32(H),
        width         = np.int32(W),
        fx=np.float32(fx), fy=np.float32(fy), cx=np.float32(cx), cy=np.float32(cy),
        depth_min = meta['depth_min'],
        depth_max = meta['depth_max'],
        depth_source = "depth.bin",
        svo_source   = meta['svo_source'],
    )

    pc_mb = os.path.getsize(pc_path) / 1e6
    print(f"[Saved]  {pc_path}  ({pc_mb:.1f} MB)")
    print(f"[Saved]  {pc_meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert depth.bin -> organized per-pixel XYZ point cloud.")
    parser.add_argument('path', nargs='?', default=None,
                         help="Recording folder (containing depth.bin). Omit for a folder picker.")
    args = parser.parse_args()

    outdir = resolve_folder(args.path)
    if outdir is None:
        print("[Cancelled] No folder selected.")
        sys.exit(1)

    convert(outdir)


if __name__ == "__main__":
    main()
