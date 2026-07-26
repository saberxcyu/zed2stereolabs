import sys
import os
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')


def choose_csv_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select ROI CSV to plot",
        initialdir=TOOLS_RECORDING_DIR,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return path


def load_roi_csv(path):
    """Parse a select_roi.py CSV into:
    - rois: ROI names in first-seen order
    - series: {roi: (frames, avg_time, avg_value)}, one point per frame, averaged
      over every point in that ROI/frame (multiple points per pixel-region
      selection collapse to a single value per frame here)
    - t0: earliest average timestamp across all ROIs, used to normalize every
      series to a shared zero
    - value_col: 'z' (pointcloud.bin mode -- select_roi.py wrote x,y,z) or
      'depth' (depth.bin mode -- select_roi.py wrote depth directly), detected
      from the CSV header so this script works with either schema.
    """
    val_by_roi_frame = defaultdict(lambda: defaultdict(list))
    t_by_roi_frame   = defaultdict(lambda: defaultdict(list))
    rois = []

    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {name: i for i, name in enumerate(header)}
        value_col = 'z' if 'z' in col else 'depth'
        roi_i, frame_i, val_i, ts_i = col['roi'], col['frame'], col[value_col], col['timestamp']

        for row in reader:
            roi = row[roi_i]
            frame_map = val_by_roi_frame[roi]
            if not frame_map and roi not in rois:
                rois.append(roi)
            frame = int(row[frame_i])
            frame_map[frame].append(float(row[val_i]))
            t_by_roi_frame[roi][frame].append(float(row[ts_i]))

    series = {}
    all_ts = []
    for roi in rois:
        val_map = val_by_roi_frame[roi]
        t_map   = t_by_roi_frame[roi]
        frames  = sorted(val_map.keys())
        avg_t   = [sum(t_map[fr]) / len(t_map[fr]) for fr in frames]
        avg_val = [sum(val_map[fr]) / len(val_map[fr]) for fr in frames]
        series[roi] = (frames, avg_t, avg_val)
        all_ts.extend(avg_t)

    t0 = min(all_ts)
    return rois, series, t0, value_col


def main():
    parser = argparse.ArgumentParser(description="Plot average depth vs. time per ROI from a select_roi.py CSV.")
    parser.add_argument('path', nargs='?', default=None,
                         help="roi.csv path. Omit for a file picker.")
    args = parser.parse_args()

    csv_path = args.path or choose_csv_file()
    if not csv_path:
        print("[Cancelled] No CSV selected.")
        sys.exit(1)

    rois, series, t0, value_col = load_roi_csv(csv_path)
    print(f"[Input] {csv_path}  (value column: {value_col})")

    fig, ax = plt.subplots(figsize=(9, 5))
    for roi in rois:
        frames, avg_t, avg_val = series[roi]
        avg_t_s = [t - t0 for t in avg_t]
        if value_col == 'z':
            # point cloud Z = -depth (RIGHT_HANDED_Y_UP: forward objects are negative Z);
            # flip back to positive distance-from-camera before plotting/inverting.
            avg_depth = [-z for z in avg_val]
        else:
            # depth.bin values are already positive distance-from-camera.
            avg_depth = avg_val
        for fr, t, d in zip(frames, avg_t_s, avg_depth):
            print(f"roi={roi:<12} frame={fr:>5}  time={t:.3f}s  avg_depth={d:.4f} m")
        ax.plot(avg_t_s, avg_depth, linewidth=1, label=roi)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (m) (inverted depth)")
    ax.invert_yaxis()
    ax.set_title("Average Depth Over Time")
    ax.grid(True, alpha=0.3)
    if len(rois) > 1:
        ax.legend(fontsize=7, ncol=1, loc='center left', bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    csv_dir  = os.path.dirname(os.path.abspath(csv_path))
    out_path = os.path.join(csv_dir, f"{Path(csv_path).stem}_depth_over_time.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
