import sys
import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, RectangleSelector

# -------------------------------- TO RUN --------------------------------
# Run as -> uv run read.py "Path to the recording folder"
# Use capture.py to create the recording folder
# ------------------------------------------------------------------------

def preload_rgb_frames(cap, n, h, w):
    """Read all n frames sequentially (no seeking) into a list of RGB arrays."""
    rgb_list = []
    for _ in range(n):
        ret, frame = cap.read()
        rgb_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else np.zeros((h, w, 3), np.uint8))
    return rgb_list

def build_depth_rgba(frame_data, min_depth, max_depth, cmap,
                     valid_alpha=0.55, nan_alpha=0.80):
    """
    Build an (H, W, 4) float32 RGBA overlay array.

    - Valid depth in [min_depth, max_depth] -> jet_r colormap, semi-transparent
    - +inf (TOO_FAR)   -> saturated deep blue (far boundary), semi-transparent
    - -inf (TOO_CLOSE) -> saturated deep red  (near boundary), semi-transparent
    - NaN  (no data)   -> solid black, more opaque - clearly marks missing data
    """
    f = frame_data.astype(np.float32)
    nan_mask = ~np.isfinite(f)  # Catch NaN and Inf

    # Create a copy for calculation and CLIP FIRST
    # Do NOT replace NaNs with min_depth here
    f_clipped = np.clip(f, min_depth, max_depth)

    range_span = max_depth - min_depth
    if range_span > 0:
        normalized = (f_clipped - min_depth) / range_span
    else:
        normalized = np.full_like(f_clipped, 0.5)

    # Get colors from colormap
    rgba = cmap(normalized).astype(np.float32)

    # Apply transparency to valid pixels
    rgba[..., 3] = valid_alpha

    # FORCE Black and Opacity for NaNs
    # This must happen LAST to override the cmap colors
    rgba[nan_mask, 0:3] = 0.0  # RGB = 0 (Black)
    rgba[nan_mask, 3]   = nan_alpha
    
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


def main(folder_path):
    npz_path   = os.path.join(folder_path, "depth.npz")
    video_path = os.path.join(folder_path, "video.mp4")

    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        sys.exit(1)

    data       = np.load(npz_path, mmap_mode='r')
    frames     = data['frames']       # (N, H, W) float16, memory-mapped
    timestamps = data['timestamps']   # (N,) float64
    min_depth  = float(data['min_depth'])
    max_depth  = float(data['max_depth'])
    N, H, W    = frames.shape

    rgb_list  = None
    has_video = False
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            est_mb = N * H * W * 3 / 1e6
            print(f"[Loading] Reading {N} video frames (~{est_mb:.0f} MB)...", end='', flush=True)
            rgb_list  = preload_rgb_frames(cap, N, H, W)
            has_video = True
            cap.release()
            print(" done.")
        else:
            print(f"[Warning] Could not open {video_path}. Showing depth only.")
            cap.release()
    else:
        print(f"[Warning] {video_path} not found. Showing depth only.")

    print(f"Loaded: {N} frames, {W}x{H}, depth range [{min_depth}, {max_depth}] m")
    print("Left-click to select point  |  Left-click+drag to select region")
    print("Drag the slider to navigate frames.\n")

    # Colormap: jet_r maps 0->red (near), 1->blue (far)
    cmap = mcm.get_cmap('jet_r').copy()

    # -- Figure layout --------------------------------------------------------
    fig = plt.figure(figsize=(18, 7))
    fig.canvas.manager.set_window_title(f"Reading depth data - {folder_path}")

    ax_frame  = fig.add_axes([0.05, 0.18, 0.44, 0.74])
    ax_plot   = fig.add_axes([0.57, 0.18, 0.40, 0.74])
    ax_slider = fig.add_axes([0.10, 0.05, 0.80, 0.04])

    # -- Frame panel - RGB background + depth RGBA overlay -------------------
    depth_rgba0 = build_depth_rgba(frames[0], min_depth, max_depth, cmap)

    if has_video:
        rgb_im = ax_frame.imshow(rgb_list[0], aspect='equal', interpolation='nearest')

    depth_im = ax_frame.imshow(depth_rgba0, aspect='equal', interpolation='nearest')

    ax_frame.set_title("Frame 0  |  t = 0.000 s", fontsize=9)
    ax_frame.set_xlabel("x (px)")
    ax_frame.set_ylabel("y (px)")

    # Colorbar via a standalone ScalarMappable (not tied to depth_im's RGBA data)
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=min_depth, vmax=max_depth))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_frame, label='Depth (m)', fraction=0.03, pad=0.02)

    # Selection markers (drawn above both layers)
    point_marker, = ax_frame.plot([], [], '+', color='white', ms=14, mew=2.5, zorder=6)
    region_rect   = plt.Rectangle((0, 0), 0, 0, edgecolor='white', facecolor='none',
                                   lw=2, zorder=6, visible=False)
    ax_frame.add_patch(region_rect)

    # -- Plot panel -----------------------------------------------------------
    ax_plot.set_xlabel('Time (s)')
    ax_plot.set_ylabel('Depth (m)')
    ax_plot.set_xlim(timestamps[0], timestamps[-1])
    ax_plot.set_ylim(min_depth, max_depth)
    ax_plot.set_title('Click or drag on the frame to select a point / region', fontsize=9)
    ax_plot.grid(True, alpha=0.3)
    plot_line, = ax_plot.plot([], [], lw=1.5, color='steelblue')
    vline = ax_plot.axvline(timestamps[0], color='red', lw=1, ls='--', alpha=0.7)

    # -- Frame slider ---------------------------------------------------------
    slider = Slider(ax_slider, 'Frame', 0, N - 1, valinit=0, valstep=1)

    def on_slider(val):
        idx = int(slider.val)
        if has_video:
            rgb_im.set_data(rgb_list[idx])
        depth_im.set_data(build_depth_rgba(frames[idx], min_depth, max_depth, cmap))
        ax_frame.set_title(f"Frame {idx}  |  t = {timestamps[idx]:.3f} s", fontsize=9)
        vline.set_xdata([timestamps[idx]])
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    # -- Selection (click -> point, drag -> region) ---------------------------
    def update_plot(ts, depth_series, label):

        # Create a masked array where non-finite values are hidden
        masked_depth = np.ma.masked_where(~np.isfinite(depth_series), depth_series)
        
        # Pass the full timestamps and the masked depth
        plot_line.set_xdata(ts)
        plot_line.set_ydata(masked_depth)
        
        ax_plot.set_title(label, fontsize=9)
        ax_plot.set_xlim(ts[0], ts[-1])
        ax_plot.set_ylim(min_depth, max_depth)
        vline.set_xdata([ts[int(slider.val)]])
        fig.canvas.draw_idle()

    _selector_fired = [False]   # mutable flag shared between on_select / on_release

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

        region       = frames[:, y1:y2 + 1, x1:x2 + 1].astype(np.float32)
        depth_series = np.nanmean(region.reshape(N, -1), axis=1)
        label        = f"Region x=[{x1},{x2}] y=[{y1},{y2}] - mean"
        point_marker.set_data([], [])
        region_rect.set_xy((x1, y1))
        region_rect.set_width(x2 - x1)
        region_rect.set_height(y2 - y1)
        region_rect.set_visible(True)
        print_region_table(x1, y1, x2, y2, timestamps, region)
        update_plot(timestamps, depth_series, label)

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
        depth_series = frames[:, cy, cx].astype(np.float32)
        label        = f"Point ({cx}, {cy})"
        point_marker.set_data([cx], [cy])
        region_rect.set_visible(False)
        print_point_table(cx, cy, timestamps, depth_series)
        update_plot(timestamps, depth_series, label)

    selector = RectangleSelector(
        ax_frame, on_select,
        useblit=True, button=[1],
        minspanx=5, minspany=5,
        spancoords='pixels', interactive=False,
    )
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query depth over time from a folder recorded by capture.py"
    )
    parser.add_argument('folder', type=str,
                        help='Path to the output folder produced by capture.py '
                             '(must contain video.mp4 and depth.npz)')
    opt = parser.parse_args()
    main(opt.folder)
