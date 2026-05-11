from _common import *


def pose_analyze_mode(folder):
    """Interactive pose trajectory viewer. Returns 'back' or 'quit'."""
    npz_path   = os.path.join(folder, "pose.npz")
    video_path = os.path.join(folder, "video.mp4")

    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found.")
        return 'back'

    data          = np.load(npz_path)
    timestamps    = data['timestamps']
    person_ids    = data['person_ids']
    keypoints_3d  = data['keypoints_3d']
    keypoints_2d  = data['keypoints_2d']
    body_format_i = int(data['body_format'])
    image_scale   = list(data['image_scale'])
    body_format   = {18: sl.BODY_FORMAT.BODY_18,
                     34: sl.BODY_FORMAT.BODY_34,
                     38: sl.BODY_FORMAT.BODY_38}[body_format_i]
    kp_names      = KEYPOINT_NAMES[body_format_i]

    N = len(timestamps)
    MAX_PERSONS = person_ids.shape[1]

    # Preload video frames as RGB
    rgb_list  = []
    has_video = False
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            H_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            W_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            print(f"[Loading] Reading {N} video frames...", end='', flush=True)
            rgb_list  = preload_rgb_frames(cap, N, H_v, W_v)
            has_video = True
            cap.release()
            print(" done.")
        else:
            cap.release()
    if not has_video:
        print(f"[Warning] {video_path} not found or unreadable. Skeleton only.")
        rgb_list = [np.zeros((480, 640, 3), dtype=np.uint8)] * N

    print(f"Loaded: {N} frames, body_format={body_format_i}")
    print("Select a keypoint from the panel to plot its X/Y/Z trajectory over time.")

    # -----------------------------------------------------------------------
    # Matplotlib figure
    # -----------------------------------------------------------------------
    from matplotlib.widgets import RadioButtons as _RB

    _AXIS_COLORS  = [(1.0, 60/255, 60/255), (60/255, 220/255, 60/255), (80/255, 150/255, 1.0)]
    _AXIS_LABELS  = ['X (m)', 'Y (m)', 'Z (m)']
    _PERSON_STYLES = ['-', '--', ':', '-.']

    fig = plt.figure(figsize=(20, 8))
    fig.canvas.manager.set_window_title(f"Pose Analysis - {folder}")

    ax_frame  = fig.add_axes([0.04, 0.15, 0.40, 0.78])
    ax_radio  = fig.add_axes([0.47, 0.15, 0.10, 0.78])
    ax_plot   = fig.add_axes([0.65, 0.15, 0.32, 0.78])
    ax_play   = fig.add_axes([0.02, 0.04, 0.05, 0.065])
    ax_slider = fig.add_axes([0.09, 0.05, 0.87, 0.04])

    first_skel = draw_axes_gizmo(draw_skeleton_from_saved(
        rgb_list[0], keypoints_2d[0], person_ids[0], body_format, image_scale))
    frame_im = ax_frame.imshow(first_skel, aspect='equal', interpolation='nearest')
    ax_frame.set_title("Frame 0  |  t = 0.000 s", fontsize=9)
    ax_frame.axis('off')

    slider   = Slider(ax_slider, '', 0, N - 1, valinit=0, valstep=1)
    play_btn = Button(ax_play, '>')

    vline_ref = [None]

    # ---- helpers ----------------------------------------------------------

    def _majority_pid(slot):
        ids   = person_ids[:, slot]
        valid = ids[ids >= 0]
        if len(valid) == 0:
            return slot
        return int(np.bincount(valid).argmax())

    def _plot_trajectory(kp_name):
        ax_plot.cla()
        kp_idx = kp_names.index(kp_name)
        ax_plot.set_xlabel('Time (s)')
        ax_plot.set_ylabel('Position (m)')
        ax_plot.set_title(kp_name, fontsize=9)
        ax_plot.set_xlim(timestamps[0], timestamps[-1])
        ax_plot.grid(True, alpha=0.3)

        person_slot = -1
        for slot in range(MAX_PERSONS):
            xyz   = keypoints_3d[:, slot, kp_idx, :]
            if not np.any(np.isfinite(xyz)):
                continue
            person_slot += 1
            ls  = _PERSON_STYLES[person_slot % len(_PERSON_STYLES)]
            pid = _majority_pid(slot)

            for comp, (axis_color, axis_lbl) in enumerate(zip(_AXIS_COLORS, _AXIS_LABELS)):
                comp_valid = np.isfinite(xyz[:, comp])
                if not np.any(comp_valid):
                    continue
                ts_v  = timestamps[comp_valid]
                val_v = xyz[comp_valid, comp]
                ax_plot.plot(ts_v, val_v, color=axis_color, lw=1.5, ls=ls,
                             label=f"ID {pid} {axis_lbl}" if person_slot == 0 else None)
        ax_plot.legend(fontsize=8, ncol=1)
        vline_ref[0] = ax_plot.axvline(timestamps[int(slider.val)],
                                       color='red', lw=1, ls='--', alpha=0.7)
        fig.canvas.draw_idle()

    radio = _RB(ax_radio, kp_names)
    radio.on_clicked(_plot_trajectory)
    _plot_trajectory(kp_names[0])

    # ---- slider / play ----------------------------------------------------

    def on_slider(val):
        idx  = int(slider.val)
        skel = draw_axes_gizmo(draw_skeleton_from_saved(
            rgb_list[idx], keypoints_2d[idx], person_ids[idx], body_format, image_scale))
        frame_im.set_data(skel)
        ax_frame.set_title(f"Frame {idx}  |  t = {timestamps[idx]:.3f} s", fontsize=9)
        if vline_ref[0] is not None:
            vline_ref[0].set_xdata([timestamps[idx]])
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    playing = [False]
    timer   = fig.canvas.new_timer(interval=33)

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

    # ---- key / close ------------------------------------------------------

    closed_by_q = [False]

    def on_key(event):
        if event.key == 'q':
            closed_by_q[0] = True
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()
    return 'back' if closed_by_q[0] else 'quit'
