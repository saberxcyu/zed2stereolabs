import sys
import os
import argparse
from collections import Counter

import numpy as np
import tkinter as tk
from tkinter import filedialog

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

PC_CHANNELS = 3  # pointcloud.bin: XYZ per pixel


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


def select_primary_person(person_ids):
    """Pick the track ID present in the most frames (the persistent subject),
    rather than re-selecting independently per frame -- avoids flickering
    between the subject and a briefly-detected bystander."""
    counts = Counter(int(pid) for pid in person_ids.ravel() if pid != -1)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def project_to_pixel(x, y, z, fx, fy, cx, cy):
    """Inverse of process.py's depth->XYZ projection (RIGHT_HANDED_Y_UP, forward = -Z):
    X=(col-cx)/fx*depth, Y=-(row-cy)/fy*depth, Z=-depth."""
    depth = -z
    with np.errstate(divide='ignore', invalid='ignore'):
        col = cx + x * fx / depth
        row = cy - y * fy / depth
    valid = np.isfinite(row) & np.isfinite(col) & (depth > 0)
    return row, col, valid


def mask_joints_to_fg(jf, jconf, fg, H, W, fx, fy, cx, cy):
    """NaN-out any joint whose projected pixel location falls outside this frame's
    foreground mask (or out of bounds) -- ZED body tracking runs independently of
    SAM3 and can track a bystander, or place a joint just outside the segmented
    silhouette, either of which would otherwise leak a non-person point into the
    SMPLify-3D dataset."""
    row, col, valid = project_to_pixel(jf[:, 0], jf[:, 1], jf[:, 2], fx, fy, cx, cy)
    row_i = np.round(row).astype(np.int64)
    col_i = np.round(col).astype(np.int64)
    in_bounds = valid & (row_i >= 0) & (row_i < H) & (col_i >= 0) & (col_i < W)

    in_mask = np.zeros(jf.shape[0], dtype=bool)
    in_mask[in_bounds] = fg[row_i[in_bounds], col_i[in_bounds]]

    jf = jf.copy()
    jf[~in_mask] = np.nan
    jconf = np.where(in_mask, jconf, np.nan)
    return jf, jconf


def combine_data(outdir, mask_threshold):
    pc_path   = os.path.join(outdir, "pointcloud.bin")
    pc_meta_path = os.path.join(outdir, "pointcloud_meta.npz")
    mask_path = os.path.join(outdir, "mask.bin")
    mask_meta_path = os.path.join(outdir, "mask_meta.npz")
    if not os.path.exists(pc_path) or not os.path.exists(pc_meta_path):
        print(f"[Error] {pc_path} / {pc_meta_path} not found -- run process.py first.")
        sys.exit(1)
    if not os.path.exists(mask_path) or not os.path.exists(mask_meta_path):
        print(f"[Error] {mask_path} / {mask_meta_path} not found -- run segment.py first.")
        sys.exit(1)

    pc_meta = np.load(pc_meta_path)
    H, W = int(pc_meta['height']), int(pc_meta['width'])
    timestamps_ns = pc_meta['timestamps_ns']
    fx, fy, cx, cy = (float(pc_meta[k]) for k in ('fx', 'fy', 'cx', 'cy'))

    pc_frame_bytes = H * W * PC_CHANNELS * 4
    n_frames = os.path.getsize(pc_path) // pc_frame_bytes
    if n_frames == 0:
        print("[Error] pointcloud.bin has zero complete frames.")
        sys.exit(1)
    points = np.memmap(pc_path, dtype=np.float32, mode='r', shape=(n_frames, H, W, PC_CHANNELS))

    mask_frame_bytes = H * W
    n_mask_frames = os.path.getsize(mask_path) // mask_frame_bytes
    if n_mask_frames != n_frames:
        print(f"[Error] Frame count mismatch: pointcloud.bin has {n_frames} frames, "
              f"mask.bin has {n_mask_frames} frames.")
        sys.exit(1)
    masks = np.memmap(mask_path, dtype=np.uint8, mode='r', shape=(n_frames, H, W))

    print(f"[Loaded] {pc_path}  ({n_frames} frames, {W}x{H})")
    print(f"[Loaded] {mask_path}")

    pose_path = os.path.join(outdir, "pose.npz")
    has_pose = os.path.exists(pose_path)
    primary_id = None
    if has_pose:
        pose = np.load(pose_path)
        if int(pose['timestamps_ns'].shape[0]) != n_frames:
            print(f"[Error] Frame count mismatch: pointcloud.bin has {n_frames} frames, "
                  f"pose.npz has {pose['timestamps_ns'].shape[0]} frames.")
            sys.exit(1)
        primary_id = select_primary_person(pose['person_ids'])
        if primary_id is None:
            print("[Warning] pose.npz has no detected persons -- omitting joints from dataset.")
            has_pose = False
        else:
            print(f"[Loaded] {pose_path}  (primary person id={primary_id})")
    else:
        print(f"[Warning] {pose_path} not found -- dataset will have no joints.")

    thresh_u8 = int(round(mask_threshold * 255))

    all_points = []
    frame_counts = np.empty(n_frames, dtype=np.int32)
    if has_pose:
        n_kp = pose['keypoints_3d'].shape[2]
        joints_3d = np.full((n_frames, n_kp, 3), np.nan, dtype=np.float32)
        joint_confidence = np.full((n_frames, n_kp), np.nan, dtype=np.float32)

    n_nonfinite = 0  # masked-in pixels dropped for non-finite depth (NaN/+-inf sentinel)

    for frame in range(n_frames):
        fg = masks[frame] > thresh_u8
        pts_masked = points[frame][fg]
        valid = np.all(np.isfinite(pts_masked), axis=1)
        pts = pts_masked[valid]
        all_points.append(pts)
        frame_counts[frame] = pts.shape[0]

        n_nonfinite += pts_masked.shape[0] - pts.shape[0]

        if has_pose:
            slot = np.where(pose['person_ids'][frame] == primary_id)[0]
            if slot.size > 0:
                jf    = pose['keypoints_3d'][frame, slot[0]]
                jconf = np.full(n_kp, pose['body_confidence'][frame, slot[0]], dtype=np.float32)
                jf, jconf = mask_joints_to_fg(jf, jconf, fg, H, W, fx, fy, cx, cy)
                joints_3d[frame]        = jf
                joint_confidence[frame] = jconf

        if (frame + 1) % 30 == 0 or frame == n_frames - 1:
            print(f"\r[Building] {frame + 1}/{n_frames} frames", end='', flush=True)
    print()

    points_flat = np.concatenate(all_points, axis=0) if all_points else np.empty((0, 3), dtype=np.float32)
    timestamps = timestamps_ns.astype(np.float64) / 1e9

    n_kept        = points_flat.shape[0]
    n_masked_in   = n_kept + n_nonfinite  # pixels that passed the SAM3 mask, before the finite check
    nonfinite_pct = 100 * n_nonfinite / n_masked_in if n_masked_in else 0.0

    print(f"[Filtered] non-finite depth (inside mask): {n_nonfinite:,} / {n_masked_in:,} masked-in points ({nonfinite_pct:.1f}%)")

    combined_data_path = os.path.join(outdir, "combined_data.npz")
    save_kwargs = dict(
        points       = points_flat,
        frame_counts = frame_counts,
        timestamps   = timestamps,
        svo_source   = str(pc_meta['svo_source']),
        # Left-camera intrinsics (already loaded above for mask_joints_to_fg's projection) --
        # needed downstream for reprojecting 3D joints/points back to pixel space (e.g. a 2D
        # keypoint reprojection loss), not just for the foreground-mask check done here.
        height       = np.int32(H),
        width        = np.int32(W),
        fx           = np.float32(fx),
        fy           = np.float32(fy),
        cx           = np.float32(cx),
        cy           = np.float32(cy),
    )
    if has_pose:
        save_kwargs.update(
            joints_3d        = joints_3d,
            joint_confidence = joint_confidence,
            body_format       = np.int32(pose['body_format']),
        )
    np.savez_compressed(combined_data_path, **save_kwargs)

    dataset_mb = os.path.getsize(combined_data_path) / 1e6
    print(f"[Saved] {combined_data_path}  ({dataset_mb:.1f} MB, {points_flat.shape[0]} total points, "
          f"{n_frames} frames, joints={'yes' if has_pose else 'no'})")


def main():
    parser = argparse.ArgumentParser(
        description="Combine pointcloud.bin + mask.bin + pose.npz into a filtered, SMPLify-3D-ready combined_data.npz.")
    parser.add_argument('path', nargs='?', default=None,
                         help="Recording folder (containing pointcloud.bin, mask.bin, pose.npz). Omit for a folder picker.")
    parser.add_argument('--mask-threshold', type=float, default=0.5,
                         help="Foreground mask threshold in [0,1] (default 0.5)")
    args = parser.parse_args()

    outdir = resolve_folder(args.path)
    if outdir is None:
        print("[Cancelled] No folder selected.")
        sys.exit(1)

    combine_data(outdir, args.mask_threshold)


if __name__ == "__main__":
    main()
