"""Initialize one stationary ZED using IMU gravity alignment, without floor detection.

The world origin is the camera centre; no floor height is estimated.
World heading retains the SDK initial yaw reference, not a measured room heading."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
CAMERA_FRAME = 'RIGHT_HANDED_Y_UP'
WORLD_FRAME = 'RIGHT_HANDED_Z_UP_X_FORWARD'
# SDK Y-up world -> Z-up world; preserve a right-handed frame.
WORLD_FROM_SDK = np.array([[0, 0, -1, 0], [-1, 0, 0, 0],
                           [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float)


def load_calibration(path, serial=None):
    with open(path) as f:
        data = json.load(f)
    if (data.get('schema_version') != 1 or data.get('units') != 'metres'
            or data.get('camera_frame') != CAMERA_FRAME
            or data.get('world_frame') != WORLD_FRAME):
        raise ValueError('Unsupported calibration schema, units or coordinate frames')
    if serial is not None and int(data['camera_serial']) != int(serial):
        raise ValueError('Calibration camera serial does not match the opened camera')
    t = np.asarray(data['T_world_from_camera'], dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-4)):
        raise ValueError('Calibration must contain a finite proper rigid transform')
    return data


def calibration_metadata(data):
    meta = dict(pointcloud_frame=CAMERA_FRAME, coordinate_units='metres',
                calibration_available=data is not None)
    if data is not None:
        meta.update(T_world_from_camera=np.asarray(data['T_world_from_camera']),
                    world_frame=WORLD_FRAME, calibration_json=json.dumps(data))
    return meta


def camera_tilt_degrees(T_world_from_camera):
    """Angles relative to world vertical, independent of world heading.

    Camera axes: X right, Y up, -Z forward. Pitch is forward-axis elevation
    (positive looking upward). Roll is positive when the image-right axis
    tips upward; undefined when looking exactly vertically. Total tilt is
    the angle between camera +Y and world +Z. Values describe the camera
    pose, not Euler angles of the Z-up axis-conversion matrix.
    """
    up = np.asarray(T_world_from_camera, dtype=float)[2, :3]
    up = up / np.linalg.norm(up)
    horizontal = np.hypot(up[0], up[1])
    return dict(
        pitch_up_deg=float(np.degrees(np.arctan2(-up[2], horizontal))),
        roll_right_up_deg=(float(np.degrees(np.arctan2(up[0], up[1])))
                           if horizontal > 1e-8 else None),
        total_tilt_deg=float(np.degrees(np.arctan2(np.hypot(up[0], up[2]), up[1]))))



def main():
    import pyzed.sl as sl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial', type=int)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--settle-seconds', type=float, default=5,
                        help='Continuous OK tracking required before saving (minimum 1 second)')
    parser.add_argument('--resolution', choices=['HD720', 'HD1080', 'HD2K'], default='HD2K')
    args = parser.parse_args()
    if args.timeout <= 0 or args.settle_seconds < 1 or args.timeout <= args.settle_seconds:
        parser.error('timeout must exceed settle-seconds, which must be at least 1')
    camera = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    if args.serial is not None:
        init.set_from_serial_number(args.serial)
    try:
        status = camera.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f'Camera open failed: {status}')
        parameters = sl.PositionalTrackingParameters()
        parameters.enable_imu_fusion = True
        parameters.set_gravity_as_origin = True
        parameters.set_floor_as_origin = False
        parameters.set_as_static = True
        status = camera.enable_positional_tracking(parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f'Tracking initialization failed: {status}')
        print('Keep the camera stationary. Gravity alignment only; no floor visibility is required.', flush=True)
        start = time.monotonic()
        ready_since = None
        last_state = None
        samples = []
        pose = sl.Pose()
        runtime = sl.RuntimeParameters()
        while time.monotonic() - start < args.timeout:
            if camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                ready_since = None
                samples.clear()
                continue
            state = camera.get_position(pose, sl.REFERENCE_FRAME.WORLD)
            if state != last_state:
                print(f'Tracking: {state}', flush=True)
                last_state = state
            if state != sl.POSITIONAL_TRACKING_STATE.OK:
                ready_since = None
                samples.clear()
                continue
            if ready_since is None:
                ready_since = time.monotonic()
            samples.append(pose.pose_data().m.copy())
            if time.monotonic() - ready_since >= args.settle_seconds:
                break
        else:
            raise RuntimeError('Timed out waiting for continuous OK gravity-aligned tracking; no calibration saved')
        # Keep only the gravity-aligned rotation: this calibration does not
        # measure camera position or floor height.
        t_sdk = samples[-1].copy()
        t_sdk[:3, 3] = 0.0
        t_world = WORLD_FROM_SDK @ t_sdk
        tilt = camera_tilt_degrees(t_world)
        info = camera.get_camera_information()
        config = info.camera_configuration
        intr = config.calibration_parameters.left_cam
        data = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                    camera_serial=int(info.serial_number), sdk_version=sl.Camera.get_sdk_version(),
                    units='metres', camera_frame=CAMERA_FRAME, world_frame=WORLD_FRAME,
                    transform_convention='column vectors: p_world = T_world_from_camera @ p_camera',
                    T_world_from_camera=t_world.tolist(), T_sdk_world_from_camera=t_sdk.tolist(),
                    tilt_degrees=tilt,
                    tilt_convention='pitch positive looking up; roll positive image-right up; total angle camera +Y to world +Z',
                    method='zed_sdk_gravity_static', stationary_camera_required=True,
                    world_origin='camera_centre',
                    heading_reference='sdk_initial_yaw',
                    floor_height_estimated=False,
                    tracking_settings=dict(enable_imu_fusion=True, set_gravity_as_origin=True,
                                           set_floor_as_origin=False, set_as_static=True),
                    resolution=[int(config.resolution.width), int(config.resolution.height)],
                    rectified_intrinsics={k:float(getattr(intr,k)) for k in ['fx','fy','cx','cy']},
                    quality=dict(continuous_ok_frames=len(samples),
                                 translation_span_m=np.ptp(np.array(samples)[:,:3,3],axis=0).tolist(),
                                 pose_held_static=True,
                                 readiness_only=True,
                                 independently_validated=False))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S.%f')
        output_path = Path(__file__).resolve().parent / 'calibration' / f'{timestamp}.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects previous calibrations.
        with output_path.open('x') as f:
            json.dump(data, f, indent=2)
            f.write('\n')
        load_calibration(output_path, info.serial_number)
        print(f'Saved {output_path}\nOrigin: camera centre; floor height not estimated.\nT_world_from_camera:\n{t_world}')
        roll_text = 'undefined (vertical view)' if tilt['roll_right_up_deg'] is None else f"{tilt['roll_right_up_deg']:+.3f} deg"
        print(f"Pitch (positive looking up): {tilt['pitch_up_deg']:+.3f} deg\n"
              f"Roll (positive image-right up): {roll_text}\n"
              f"Total tilt from level: {tilt['total_tilt_deg']:.3f} deg")
        print('Static tracking holds the pose fixed: the OK window checks readiness, not convergence or repeatability.')
        print('Recalibrate after moving the camera. OK tracking is not an independent accuracy check.')
    finally:
        camera.close()


if __name__ == '__main__':
    main()
