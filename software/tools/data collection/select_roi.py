import sys
import os
import argparse

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RectangleSelector, Button

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

# matplotlib's default "tab10" color cycle, same as _depth_analyze.py's ROI_COLORS and
# what plot.py's ROI lines get automatically -- keeps ROI colors consistent across tools.
ROI_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

CHANNELS = 3  # pointcloud.bin: XYZ per pixel


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


def select_roi(outdir):
    bin_path  = os.path.join(outdir, "pointcloud.bin")
    meta_path = os.path.join(outdir, "pointcloud_meta.npz")
    if not os.path.exists(bin_path) or not os.path.exists(meta_path):
        print(f"[Error] {bin_path} / {meta_path} not found -- run process.py first.")
        sys.exit(1)

    meta = np.load(meta_path)
    H, W = int(meta['height']), int(meta['width'])
    timestamps_ns = meta['timestamps_ns']
    frame_bytes = H * W * CHANNELS * 4
    n_frames = os.path.getsize(bin_path) // frame_bytes
    if n_frames == 0:
        print("[Error] pointcloud.bin has zero complete frames.")
        sys.exit(1)

    data = np.memmap(bin_path, dtype=np.float32, mode='r', shape=(n_frames, H, W, CHANNELS))
    print(f"[Loaded] {bin_path}  ({n_frames} frames, {W}x{H})")
    print(f"[Loaded] {meta_path}")

    video_path = os.path.join(outdir, "video.mp4")
    has_video = os.path.exists(video_path)
    cap = cv2.VideoCapture(video_path) if has_video else None
    if has_video and not (cap and cap.isOpened()):
        print(f"[Warning] Could not open {video_path}. Showing blank frames.")
        has_video = False
    if has_video:
        print(f"[Loaded] {video_path}")
    else:
        print(f"[Warning] {video_path} not found -- showing blank frames.")

    last_idx = [-1]
    blank_rgb = np.zeros((H, W, 3), np.uint8)

    def get_rgb(idx):
        if not has_video:
            return blank_rgb
        if idx != last_idx[0] + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        last_idx[0] = idx
        if not ret:
            return blank_rgb
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print("Left-click to select point  |  Left-click+drag to select region")
    print("Each selection prompts for a name and is added as its own ROI.")
    print("Drag the slider to navigate frames.")
    print("Press 'e' to export confirmed ROIs to roi.csv and exit  |  'q' to cancel and exit.\n")

    tk_root = tk.Tk()
    tk_root.withdraw()
    rois = []
    exported = [False]

    fig = plt.figure(figsize=(10, 8))
    fig.canvas.manager.set_window_title(f"Select ROIs - {outdir}")

    ax_frame  = fig.add_axes([0.08, 0.15, 0.86, 0.78])
    ax_play   = fig.add_axes([0.02, 0.035, 0.06, 0.065])
    ax_slider = fig.add_axes([0.10, 0.05, 0.85, 0.04])

    rgb_im = ax_frame.imshow(get_rgb(0), aspect='equal', interpolation='nearest')
    ax_frame.set_title(f"Frame 0  |  t = 0.000 s", fontsize=9)
    ax_frame.set_xlabel("x (px)")
    ax_frame.set_ylabel("y (px)")

    point_marker, = ax_frame.plot([], [], '+', color='white', ms=14, mew=2.5, zorder=6)
    region_rect   = plt.Rectangle((0, 0), 0, 0, edgecolor='white', facecolor='none',
                                   lw=2, zorder=6, visible=False)
    ax_frame.add_patch(region_rect)

    slider   = Slider(ax_slider, '', 0, n_frames - 1, valinit=0, valstep=1)
    play_btn = Button(ax_play, '>')

    playing = [False]
    timer   = fig.canvas.new_timer(interval=33)  # ~30 fps

    def tick():
        if not playing[0]:
            return
        idx = int(slider.val)
        if idx >= n_frames - 1:
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
            if int(slider.val) >= n_frames - 1:
                slider.set_val(0)
            play_btn.label.set_text('||')
            timer.start()
        else:
            play_btn.label.set_text('>')
        fig.canvas.draw_idle()

    play_btn.on_clicked(on_play)

    def on_slider(val):
        idx = int(slider.val)
        rgb_im.set_data(get_rgb(idx))
        t_s = (int(timestamps_ns[idx]) - int(timestamps_ns[0])) / 1e9
        ax_frame.set_title(f"Frame {idx}  |  t = {t_s:.3f} s", fontsize=9)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    def confirm_roi(rows_sel, cols_sel, add_frozen_artist):
        name = simpledialog.askstring("Name ROI", "Enter a name for this ROI:", parent=tk_root)
        if name is None:
            return False
        name = name.strip()
        if not name:
            print("ROI name cannot be empty -- selection not added, try again")
            return False
        if any(roi['name'] == name for roi in rois):
            print(f"ROI name '{name}' is already used -- pick a different name")
            return False

        color = ROI_COLORS[len(rois) % len(ROI_COLORS)]
        add_frozen_artist(color, name)
        rois.append({'name': name, 'rows': rows_sel, 'cols': cols_sel, 'color': color})
        print(f"ROI '{name}' added ({len(rois)} total, {len(rows_sel)} px)")
        return True

    _selector_fired = [False]

    def on_select(eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        _selector_fired[0] = True
        x1 = int(round(min(eclick.xdata, erelease.xdata)))
        y1 = int(round(min(eclick.ydata, erelease.ydata)))
        x2 = int(round(max(eclick.xdata, erelease.xdata)))
        y2 = int(round(max(eclick.ydata, erelease.ydata)))
        x1, x2 = max(0, x1), min(W - 1, x2)
        y1, y2 = max(0, y1), min(H - 1, y2)

        rr, cc = np.meshgrid(np.arange(y1, y2 + 1), np.arange(x1, x2 + 1), indexing='ij')
        rows_sel, cols_sel = rr.ravel(), cc.ravel()

        region_rect.set_xy((x1, y1))
        region_rect.set_width(x2 - x1)
        region_rect.set_height(y2 - y1)
        region_rect.set_visible(True)
        fig.canvas.draw_idle()

        def add_frozen_artist(color, name):
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor=color,
                                  facecolor='none', lw=2, zorder=5)
            ax_frame.add_patch(rect)
            ax_frame.text(x1, max(0, y1 - 3), name, color=color, fontsize=8, zorder=6)

        confirm_roi(rows_sel, cols_sel, add_frozen_artist)
        region_rect.set_visible(False)
        fig.canvas.draw_idle()

    def on_release(event):
        if event.inaxes != ax_frame or event.button != 1:
            return
        if _selector_fired[0]:
            _selector_fired[0] = False
            return
        if event.xdata is None or event.ydata is None:
            return
        cx = max(0, min(W - 1, int(round(event.xdata))))
        cy = max(0, min(H - 1, int(round(event.ydata))))
        point_marker.set_data([cx], [cy])
        fig.canvas.draw_idle()

        def add_frozen_artist(color, name):
            ax_frame.plot([cx], [cy], '+', color=color, ms=14, mew=2.5, zorder=6)
            ax_frame.text(cx + 4, max(0, cy - 4), name, color=color, fontsize=8, zorder=6)

        confirm_roi(np.array([cy]), np.array([cx]), add_frozen_artist)
        point_marker.set_data([], [])
        fig.canvas.draw_idle()

    selector = RectangleSelector(
        ax_frame, on_select,
        useblit=True, button=[1],
        minspanx=5, minspany=5,
        spancoords='pixels', interactive=False,
    )
    fig.canvas.mpl_connect('button_release_event', on_release)

    def export_and_close():
        if not rois:
            print("[Warning] No ROIs confirmed -- select at least one before exporting.")
            return
        csv_path = os.path.join(outdir, "roi.csv")
        total_rows = 0
        print(f"[Exporting] -> {csv_path}")
        with open(csv_path, 'w', newline='') as f:
            f.write("frame,roi,x,y,z,timestamp\n")
            for frame in range(n_frames):
                frame_data = data[frame]
                ts_s = timestamps_ns[frame] / 1e9   # raw Unix-epoch seconds; plot.py normalizes to t0
                for roi in rois:
                    pts   = frame_data[roi['rows'], roi['cols'], :]
                    valid = np.all(np.isfinite(pts), axis=1)
                    for x, y, z in pts[valid]:
                        f.write(f"{frame},{roi['name']},{x:.5f},{y:.5f},{z:.5f},{ts_s:.6f}\n")
                        total_rows += 1
                if (frame + 1) % 30 == 0 or frame == n_frames - 1:
                    print(f"\r[Exporting] {frame + 1}/{n_frames} frames", end='', flush=True)
        print()
        print(f"[Saved] {csv_path}  ({total_rows} rows, {len(rois)} ROIs)")
        exported[0] = True
        plt.close(fig)

    def on_key(event):
        if event.key == 'e':
            export_and_close()
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    tk_root.destroy()
    if cap is not None:
        cap.release()

    if not exported[0]:
        print("[Cancelled] No ROIs exported.")


def main():
    parser = argparse.ArgumentParser(description="Interactive 2D ROI selection over a recording's pointcloud.bin, exporting to CSV.")
    parser.add_argument('path', nargs='?', default=None,
                         help="Recording folder (containing pointcloud.bin). Omit for a folder picker.")
    args = parser.parse_args()

    outdir = resolve_folder(args.path)
    if outdir is None:
        print("[Cancelled] No folder selected.")
        sys.exit(1)

    select_roi(outdir)


if __name__ == "__main__":
    main()
