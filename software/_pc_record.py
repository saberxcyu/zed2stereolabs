from _common import *

PC_WINDOW_NAME = "ZED Point Cloud Recorder  |  RGB (left) + Depth (right)"


def pc_record_mode(resolution, depth_min, depth_max):
    """
    resolution:  sl.RESOLUTION enum
    depth_min, depth_max: float, metres
    Records per-pixel XYZ point clouds at the selected camera resolution.
    XY coordinates are computed from depth + camera intrinsics after recording stops.
    """
    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
    )
    init.depth_minimum_distance = depth_min
    init.depth_maximum_distance = depth_max
    init.depth_stabilization    = True
    init.camera_resolution      = resolution

    zed    = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Camera open failed: {repr(status)}")
        return 'back'

    rec_res = zed.get_camera_information().camera_configuration.resolution

    calib = zed.get_camera_information().camera_configuration.calibration_parameters.left_cam
    fx, fy = calib.fx, calib.fy
    cx, cy = calib.cx, calib.cy

    print(f"[Depth range] {depth_min:.1f} – {depth_max:.1f} m")
    print(f"[Record res]  {rec_res.width} x {rec_res.height} per-pixel")
    print("Press 's' to start/stop recording | 'q' to return to menu")

    image_mat = sl.Mat()
    depth_mat = sl.Mat()

    recording   = False
    outdir      = None
    depth_frames = []
    rgb_frames   = []
    timestamps   = []
    t_start      = None

    status_msg   = 'Press [s] to start recording - [q] to return to menu'
    status_color = (200, 200, 200)

    cv2.namedWindow(PC_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PC_WINDOW_NAME, 1280, 360)

    runtime_params = sl.RuntimeParameters()
    runtime_params.confidence_threshold = 95

    exit_reason = 'back'

    try:
        while True:
            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image_mat,   sl.VIEW.LEFT,    sl.MEM.CPU, rec_res)
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH, sl.MEM.CPU, rec_res)

            rgb_bgr   = cv2.cvtColor(image_mat.get_data(), cv2.COLOR_BGRA2BGR)
            colorized = normalize_depth_to_colormap(depth_mat.get_data(), depth_min, depth_max)

            if recording:
                if t_start is None:
                    t_start = time.monotonic()
                depth_frames.append(depth_mat.get_data().copy())
                rgb_frames.append(rgb_bgr.copy())
                timestamps.append(time.monotonic() - t_start)
                status_msg   = f'Recording — frame {len(depth_frames)}  [s]=stop'
                status_color = (0, 80, 255)

            display_frame = np.concatenate([rgb_bgr, colorized], axis=1)
            draw_status(display_frame, status_msg, status_color, recording)
            cv2.imshow(PC_WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF

            try:
                if cv2.getWindowProperty(PC_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = 'quit'
                    break
            except cv2.error:
                exit_reason = 'quit'
                break

            if key == ord('s'):
                if not recording:
                    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
                    outdir = os.path.join(RECORDING_DIR, f"pc_{ts}")
                    os.makedirs(outdir, exist_ok=True)
                    depth_frames = []
                    rgb_frames   = []
                    timestamps   = []
                    t_start      = None
                    recording    = True
                    status_msg   = 'Recording started - press [s] again to stop'
                    status_color = (0, 80, 255)
                    print(f"\n[Started]  Recording to {outdir}/")
                else:
                    recording = False
                    print()

                    if not depth_frames:
                        print("[Warning]  No frames captured.")
                        status_msg   = 'No frames captured - check depth range'
                        status_color = (0, 200, 200)
                        if outdir and os.path.exists(outdir):
                            shutil.rmtree(outdir, ignore_errors=True)
                        outdir = None
                        continue

                    # --- compute XY from intrinsics + depth -----------------
                    computing_frame = display_frame.copy()
                    draw_status(computing_frame,
                                f'Computing {len(depth_frames)} frames — building point clouds ...',
                                (0, 220, 255), False)
                    cv2.imshow(PC_WINDOW_NAME, computing_frame)
                    cv2.waitKey(1)

                    H, W  = depth_frames[0].shape[:2]
                    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                    u_off = (cols - cx).astype(np.float32)
                    v_off = (rows - cy).astype(np.float32)

                    all_points   = []
                    frame_counts = []
                    for df, rf in zip(depth_frames, rgb_frames):
                        # df: (H, W) float32, positive metres from camera
                        # rf: (H, W, 3) uint8 BGR
                        valid = np.isfinite(df) & (df >= depth_min) & (df <= depth_max)
                        d_v   = df[valid]
                        # RIGHT_HANDED_Y_UP: X=right, Y=up, Z=backward
                        # Z = -depth (forward objects are negative Z)
                        # X = (u - cx) / fx * depth
                        # Y = -(v - cy) / fy * depth  (image Y is inverted vs world Y)
                        xyz = np.stack([
                            u_off[valid] / fx * d_v,
                            -(v_off[valid] / fy) * d_v,
                            -d_v,
                        ], axis=1).astype(np.float32)
                        rgb = rf[valid][:, ::-1].astype(np.float32) / 255.0  # BGR→RGB, normalise
                        all_points.append(np.concatenate([xyz, rgb], axis=1))
                        frame_counts.append(len(xyz))

                    # --- save -----------------------------------------------
                    saving_frame = display_frame.copy()
                    draw_status(saving_frame,
                                f'Saving {len(depth_frames)} frames to {outdir} ...',
                                (0, 220, 255), False)
                    cv2.imshow(PC_WINDOW_NAME, saving_frame)
                    cv2.waitKey(1)

                    npz_path = os.path.join(outdir, 'pc.npz')
                    np.savez_compressed(
                        npz_path,
                        points       = np.concatenate(all_points, axis=0),
                        frame_counts = np.array(frame_counts,  dtype=np.int32),
                        timestamps   = np.array(timestamps,    dtype=np.float64),
                        depth_min    = np.float32(depth_min),
                        depth_max    = np.float32(depth_max),
                    )
                    size_mb = os.path.getsize(npz_path) / 1e6
                    print(f"[Saved]    {outdir}/  ({len(depth_frames)} frames, {size_mb:.1f} MB)")
                    status_msg   = (f'Saved: {len(depth_frames)} frames ({size_mb:.1f} MB)'
                                    f' — press [s] to record again')
                    status_color = (0, 210, 0)
                    depth_frames = []
                    rgb_frames   = []
                    timestamps   = []
                    t_start      = None
                    outdir       = None

            elif key == ord('q'):
                exit_reason = 'back'
                break

    finally:
        image_mat.free(sl.MEM.CPU)
        depth_mat.free(sl.MEM.CPU)
        if recording and outdir is not None and os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
            print("[Discarded] Recording cancelled - no files saved.")
        zed.close()
        cv2.destroyAllWindows()

    return exit_reason
