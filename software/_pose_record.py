from _common import *


def pose_record_mode(resolution, body_format):
    """Live skeleton overlay with optional recording. Returns 'back' or 'quit'."""
    _, _, n_kp = next(opt for opt in BODY_FORMAT_OPTIONS if opt[1] == body_format)
    bf_int     = n_kp
    MAX_PERSONS = 10

    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
    )
    init.camera_resolution = resolution

    zed    = sl.Camera()
    status = zed.open(init)
    if status > sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        sys.exit(1)

    zed.enable_positional_tracking(sl.PositionalTrackingParameters())

    body_param = sl.BodyTrackingParameters()
    body_param.enable_tracking     = True
    body_param.enable_body_fitting = False
    body_param.detection_model     = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    body_param.body_format         = body_format
    zed.enable_body_tracking(body_param)

    body_runtime = sl.BodyTrackingRuntimeParameters()
    body_runtime.detection_confidence_threshold = 40

    camera_info = zed.get_camera_information()
    camera_res  = camera_info.camera_configuration.resolution
    display_res = sl.Resolution(
        min(camera_res.width, 1280),
        min(camera_res.height, 720),
    )
    image_scale = [display_res.width  / camera_res.width,
                   display_res.height / camera_res.height]

    print("Press 's' to start/stop recording | 'q' to return to menu")

    image_mat = sl.Mat()
    bodies    = sl.Bodies()

    recording          = False
    writer             = None
    outdir             = None
    pose_frames_ids    = []
    pose_frames_kp3d   = []
    pose_frames_kp2d   = []
    pose_frames_conf   = []
    pose_frames_np     = []
    timestamps         = []
    t_start            = None

    status_msg   = 'Press [s] to start recording - [q] to return to menu'
    status_color = (200, 200, 200)

    cv2.namedWindow(POSE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(POSE_WINDOW_NAME, 1920, 540)

    exit_reason = 'back'

    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_mat, sl.VIEW.LEFT, sl.MEM.CPU, display_res)
                zed.retrieve_bodies(bodies, body_runtime)

                rgb_bgra = image_mat.get_data()
                rgb_bgr  = cv2.cvtColor(rgb_bgra, cv2.COLOR_BGRA2BGR)

                overlay = rgb_bgr.copy()
                draw_skeleton_overlay(overlay, bodies, body_format, image_scale)

                if recording:
                    if t_start is None:
                        t_start = time.monotonic()
                    timestamps.append(time.monotonic() - t_start)

                    if writer is not None:
                        writer.write(rgb_bgr)

                    ids  = np.full(MAX_PERSONS, -1, dtype=np.int32)
                    kp3d = np.full((MAX_PERSONS, n_kp, 3), np.nan, dtype=np.float32)
                    kp2d = np.full((MAX_PERSONS, n_kp, 2), np.nan, dtype=np.float32)
                    conf = np.full(MAX_PERSONS, np.nan, dtype=np.float32)
                    for slot, body in enumerate(bodies.body_list[:MAX_PERSONS]):
                        ids[slot]  = body.id
                        kp3d[slot] = np.array(body.keypoint,    dtype=np.float32)
                        kp2d[slot] = np.array(body.keypoint_2d, dtype=np.float32)
                        conf[slot] = body.confidence
                    pose_frames_ids.append(ids)
                    pose_frames_kp3d.append(kp3d)
                    pose_frames_kp2d.append(kp2d)
                    pose_frames_conf.append(conf)
                    pose_frames_np.append(len(bodies.body_list))

                display_frame = np.concatenate([rgb_bgr, overlay], axis=1)
                draw_status(display_frame, status_msg, status_color, recording)
                cv2.imshow(POSE_WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF

                if cv2.getWindowProperty(POSE_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = 'quit'
                    break

                if key == ord('s'):
                    if not recording:
                        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
                        outdir = os.path.join(RECORDING_DIR, f"pose_{ts}")
                        os.makedirs(outdir, exist_ok=True)
                        writer             = create_video_writer(display_res, outdir)
                        pose_frames_ids    = []
                        pose_frames_kp3d   = []
                        pose_frames_conf   = []
                        pose_frames_np     = []
                        timestamps         = []
                        t_start            = None
                        recording          = True
                        status_msg         = 'Recording started - press [s] again to stop'
                        status_color       = (0, 80, 255)
                        print(f"[Started]   Recording to {outdir}/")
                    else:
                        recording = False
                        if writer is not None:
                            writer.release()
                            writer = None

                        saving_frame = display_frame.copy()
                        draw_status(saving_frame,
                                    f'Saving {len(timestamps)} frames to {outdir} ...',
                                    (0, 220, 255), False)
                        cv2.imshow(POSE_WINDOW_NAME, saving_frame)
                        cv2.waitKey(1)

                        npz_path = os.path.join(outdir, "pose.npz")
                        np.savez_compressed(
                            npz_path,
                            timestamps      = np.array(timestamps,       dtype=np.float64),
                            n_persons       = np.array(pose_frames_np,   dtype=np.int32),
                            person_ids      = np.stack(pose_frames_ids),
                            keypoints_3d    = np.stack(pose_frames_kp3d),
                            keypoints_2d    = np.stack(pose_frames_kp2d),
                            body_confidence = np.stack(pose_frames_conf),
                            body_format     = np.int32(bf_int),
                            image_scale     = np.array(image_scale, dtype=np.float64),
                        )
                        size_mb = os.path.getsize(npz_path) / 1e6
                        print(f"[Saved]     {outdir}/  ({len(timestamps)} frames, {size_mb:.1f} MB)")
                        status_msg   = f'Saved: {outdir}  ({size_mb:.1f} MB) - press [s] to record again'
                        status_color = (0, 210, 0)
                        pose_frames_ids  = []
                        pose_frames_kp3d = []
                        pose_frames_kp2d = []
                        pose_frames_conf = []
                        pose_frames_np   = []
                        timestamps       = []
                        t_start          = None

                elif key == ord('q'):
                    exit_reason = 'back'
                    break

    finally:
        image_mat.free(sl.MEM.CPU)
        if writer is not None:
            writer.release()
        if recording and outdir is not None and os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
            print("[Discarded] Recording cancelled - no files saved.")
        zed.disable_body_tracking()
        zed.disable_positional_tracking()
        zed.close()
        cv2.destroyAllWindows()

    return exit_reason
