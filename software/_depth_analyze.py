from _common import *


def depth_analyze_mode(folder):
    npz_path   = os.path.join(folder, "depth.npz")
    video_path = os.path.join(folder, "video.mp4")

    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        sys.exit(1)

    data       = np.load(npz_path, mmap_mode='r')
    frames     = data['frames']
    timestamps = data['timestamps']
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

    cmap = mcm.get_cmap('jet_r').copy()

    fig = plt.figure(figsize=(18, 7))
    fig.canvas.manager.set_window_title(f"Reading depth data - {folder}")

    ax_frame  = fig.add_axes([0.05, 0.18, 0.44, 0.74])
    ax_plot   = fig.add_axes([0.57, 0.18, 0.40, 0.74])
    ax_play   = fig.add_axes([0.02, 0.035, 0.06, 0.065])
    ax_slider = fig.add_axes([0.10, 0.05, 0.85, 0.04])

    depth_rgba0 = build_depth_rgba(frames[0], min_depth, max_depth, cmap)
    if has_video:
        rgb_im = ax_frame.imshow(rgb_list[0], aspect='equal', interpolation='nearest')
    depth_im = ax_frame.imshow(depth_rgba0, aspect='equal', interpolation='nearest')

    ax_frame.set_title("Frame 0  |  t = 0.000 s", fontsize=9)
    ax_frame.set_xlabel("x (px)")
    ax_frame.set_ylabel("y (px)")

    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=min_depth, vmax=max_depth))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_frame, label='Depth (m)', fraction=0.03, pad=0.02)

    point_marker, = ax_frame.plot([], [], '+', color='white', ms=14, mew=2.5, zorder=6)
    region_rect   = plt.Rectangle((0, 0), 0, 0, edgecolor='white', facecolor='none',
                                   lw=2, zorder=6, visible=False)
    ax_frame.add_patch(region_rect)

    ax_plot.set_xlabel('Time (s)')
    ax_plot.set_ylabel('Depth (m)')
    ax_plot.set_xlim(timestamps[0], timestamps[-1])
    ax_plot.set_ylim(min_depth, max_depth)
    ax_plot.set_title('Click or drag on the frame to select a point / region', fontsize=9)
    ax_plot.grid(True, alpha=0.3)
    plot_line, = ax_plot.plot([], [], lw=1.5, color='steelblue')
    vline = ax_plot.axvline(timestamps[0], color='red', lw=1, ls='--', alpha=0.7)

    slider   = Slider(ax_slider, '', 0, N - 1, valinit=0, valstep=1)
    play_btn = Button(ax_play, '>')

    playing = [False]
    timer   = fig.canvas.new_timer(interval=33)  # ~30 fps

    def tick():
        if not playing[0]:
            return
        idx = int(slider.val)
        if idx >= N - 1:
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
            if int(slider.val) >= N - 1:
                slider.set_val(0)
            play_btn.label.set_text('||')
            timer.start()
        else:
            play_btn.label.set_text('>')
        fig.canvas.draw_idle()

    play_btn.on_clicked(on_play)

    def on_slider(val):
        idx = int(slider.val)
        if has_video:
            rgb_im.set_data(rgb_list[idx])
        depth_im.set_data(build_depth_rgba(frames[idx], min_depth, max_depth, cmap))
        ax_frame.set_title(f"Frame {idx}  |  t = {timestamps[idx]:.3f} s", fontsize=9)
        vline.set_xdata([timestamps[idx]])
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    def update_plot(ts, depth_series, label):
        masked_depth = np.ma.masked_where(~np.isfinite(depth_series), depth_series)
        plot_line.set_xdata(ts)
        plot_line.set_ydata(masked_depth)
        ax_plot.set_title(label, fontsize=9)
        ax_plot.set_xlim(ts[0], ts[-1])
        ax_plot.set_ylim(min_depth, max_depth)
        vline.set_xdata([ts[int(slider.val)]])
        fig.canvas.draw_idle()

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
        cx           = max(0, min(W - 1, int(round(event.xdata))))
        cy           = max(0, min(H - 1, int(round(event.ydata))))
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

    closed_by_q = [False]

    def on_key(event):
        if event.key == 'q':
            closed_by_q[0] = True
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return 'back' if closed_by_q[0] else 'quit'
