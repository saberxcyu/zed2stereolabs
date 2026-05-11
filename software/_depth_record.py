from _common import *


def depth_record_mode(min_depth, max_depth, resolution):
    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
    )
    init.depth_minimum_distance = min_depth
    init.depth_maximum_distance = max_depth
    init.depth_stabilization    = True
    init.camera_resolution      = resolution

    zed    = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        sys.exit(1)

    display_res = get_display_resolution(zed)
    measure     = sl.MEASURE.DEPTH

    print(f"[Depth] range [{min_depth}, {max_depth}] m")
    print("Press 's' to start/stop recording | 'q' to return to menu")

    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    recording    = False
    writer       = None
    outdir       = None
    depth_frames = []
    timestamps   = []
    t_start      = None

    status_msg   = 'Press [s] to start recording - [q] to return to menu'
    status_color = (200, 200, 200)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1920, 540)

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 95

    exit_reason = 'back'

    try:
        while True:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_mat, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
                zed.retrieve_measure(depth_mat, measure, sl.MEM.CPU, display_res)

                rgb_bgra = image_mat.get_data()
                rgb_bgr  = cv2.resize(
                    cv2.cvtColor(rgb_bgra, cv2.COLOR_BGRA2BGR),
                    (display_res.width, display_res.height),
                    interpolation=cv2.INTER_LINEAR,
                )

                if recording:
                    if t_start is None:
                        t_start = time.monotonic()
                    timestamps.append(time.monotonic() - t_start)
                    depth_frames.append(depth_mat.get_data().copy())
                    if writer is not None:
                        writer.write(rgb_bgr)

                colorized     = normalize_depth_to_colormap(depth_mat.get_data(), min_depth, max_depth)
                display_frame = np.concatenate([rgb_bgr, colorized], axis=1)
                draw_status(display_frame, status_msg, status_color, recording)
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = 'quit'
                    break

                if key == ord('s'):
                    if not recording:
                        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outdir = os.path.join(RECORDING_DIR, f"depth_{ts}")
                        os.makedirs(outdir, exist_ok=True)
                        writer       = create_video_writer(display_res, outdir)
                        depth_frames = []
                        timestamps   = []
                        t_start      = None
                        recording    = True
                        status_msg   = 'Recording started - press [s] again to stop'
                        status_color = (0, 80, 255)
                        print(f"[Started]   Recording to {outdir}/")
                    else:
                        recording = False
                        if writer is not None:
                            writer.release()
                            writer = None

                        saving_frame = display_frame.copy()
                        draw_status(saving_frame,
                                    f'Saving {len(depth_frames)} frames to {outdir} ...',
                                    (0, 220, 255), False)
                        cv2.imshow(WINDOW_NAME, saving_frame)
                        cv2.waitKey(1)

                        npz_path = os.path.join(outdir, "depth.npz")
                        np.savez_compressed(
                            npz_path,
                            frames     = np.stack(depth_frames),
                            timestamps = np.array(timestamps, dtype=np.float64),
                            min_depth  = np.float32(min_depth),
                            max_depth  = np.float32(max_depth),
                        )
                        size_mb = os.path.getsize(npz_path) / 1e6
                        print(f"[Saved]     {outdir}/  ({len(depth_frames)} frames, {size_mb:.1f} MB)")
                        status_msg   = f'Saved: {outdir}  ({size_mb:.1f} MB) - press [s] to record again'
                        status_color = (0, 210, 0)
                        depth_frames = []
                        timestamps   = []
                        t_start      = None

                elif key == ord('q'):
                    exit_reason = 'back'
                    break

    finally:
        image_mat.free(sl.MEM.CPU)
        depth_mat.free(sl.MEM.CPU)
        if writer is not None:
            writer.release()
        if recording and outdir is not None and os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
            print("[Discarded] Recording cancelled - no files saved.")
        zed.close()
        cv2.destroyAllWindows()

    return exit_reason
