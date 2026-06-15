from _common import *

PC_WINDOW_NAME = "ZED Point Cloud Recorder  |  RGB (left) + Depth (right)"


def pc_record_mode(resolution, depth_min, depth_max):
    """
    resolution:  sl.RESOLUTION enum
    depth_min, depth_max: float, metres
    Records per-pixel XYZRGB point clouds at the selected camera resolution.
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

    print(f"[Depth range] {depth_min:.1f} – {depth_max:.1f} m")
    print(f"[Record res]  {rec_res.width} x {rec_res.height} per-pixel")
    print("Press 's' to start/stop recording | 'q' to return to menu")

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    pc_mat    = sl.Mat()

    recording    = False
    outdir       = None
    all_points   = []
    frame_counts = []
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
                zed.retrieve_measure(pc_mat, sl.MEASURE.XYZRGBA, sl.MEM.CPU, rec_res)
                raw = pc_mat.get_data()   # (H, W, 4) float32

                if raw is not None and raw.size > 0:
                    flat = raw.reshape(-1, 4)
                    xyz  = flat[:, :3]

                    # RIGHT_HANDED_Y_UP: camera looks along -Z, forward objects have negative Z.
                    # depth_min/max are positive distances → negate for comparison.
                    valid = (np.isfinite(xyz).all(axis=1) &
                             (xyz[:, 2] <= -depth_min) &
                             (xyz[:, 2] >= -depth_max))
                    n_valid = int(valid.sum())

                    if valid.any():
                        rgba_u32 = np.ascontiguousarray(flat[:, 3]).view(np.uint32)
                        r = ( rgba_u32        & 0xFF).astype(np.float32) / 255.0
                        g = ((rgba_u32 >>  8) & 0xFF).astype(np.float32) / 255.0
                        b = ((rgba_u32 >> 16) & 0xFF).astype(np.float32) / 255.0

                        xyz_v  = xyz[valid]
                        rgb_v  = np.stack([r[valid], g[valid], b[valid]], axis=1)
                        xyzrgb = np.concatenate([xyz_v, rgb_v], axis=1).astype(np.float32)

                        if t_start is None:
                            t_start = time.monotonic()
                        all_points.append(xyzrgb)
                        frame_counts.append(n_valid)
                        timestamps.append(time.monotonic() - t_start)

                        status_msg   = (f'Recording — frame {len(frame_counts)}'
                                        f'  pts: {n_valid:,}  [s]=stop')
                        status_color = (0, 80, 255)
                    else:
                        z_finite = xyz[np.isfinite(xyz[:, 2]), 2]
                        if len(z_finite):
                            d_range = f'{-z_finite.max():.2f}–{-z_finite.min():.2f} m'
                            status_msg = f'Recording — 0 pts in range (scene: {d_range})  [s]=stop'
                        else:
                            status_msg = 'Recording — no valid depth in frame  [s]=stop'
                        status_color = (0, 150, 255)

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
                    all_points   = []
                    frame_counts = []
                    timestamps   = []
                    t_start      = None
                    recording    = True
                    status_msg   = 'Recording started - press [s] again to stop'
                    status_color = (0, 80, 255)
                    print(f"\n[Started]  Recording to {outdir}/")
                else:
                    recording = False
                    print()

                    if not all_points:
                        print("[Warning]  No frames captured.")
                        status_msg   = 'No frames captured - check depth range'
                        status_color = (0, 200, 200)
                        if outdir and os.path.exists(outdir):
                            shutil.rmtree(outdir, ignore_errors=True)
                        outdir = None
                        continue

                    saving_frame = display_frame.copy()
                    draw_status(saving_frame,
                                f'Saving {len(frame_counts)} frames to {outdir} ...',
                                (0, 220, 255), False)
                    cv2.imshow(PC_WINDOW_NAME, saving_frame)
                    cv2.waitKey(1)

                    npz_path = os.path.join(outdir, 'pc.npz')
                    np.savez_compressed(
                        npz_path,
                        points       = np.concatenate(all_points, axis=0),
                        frame_counts = np.array(frame_counts, dtype=np.int32),
                        timestamps   = np.array(timestamps,   dtype=np.float64),
                        depth_min    = np.float32(depth_min),
                        depth_max    = np.float32(depth_max),
                    )
                    size_mb = os.path.getsize(npz_path) / 1e6
                    print(f"[Saved]    {outdir}/  ({len(frame_counts)} frames, {size_mb:.1f} MB)")
                    status_msg   = (f'Saved: {len(frame_counts)} frames ({size_mb:.1f} MB)'
                                    f' — press [s] to record again')
                    status_color = (0, 210, 0)
                    all_points   = []
                    frame_counts = []
                    timestamps   = []
                    t_start      = None
                    outdir       = None

            elif key == ord('q'):
                exit_reason = 'back'
                break

    finally:
        image_mat.free(sl.MEM.CPU)
        depth_mat.free(sl.MEM.CPU)
        pc_mat.free(sl.MEM.CPU)
        if recording and outdir is not None and os.path.exists(outdir):
            shutil.rmtree(outdir, ignore_errors=True)
            print("[Discarded] Recording cancelled - no files saved.")
        zed.close()
        cv2.destroyAllWindows()

    return exit_reason
