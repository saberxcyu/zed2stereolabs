import os
import sys
import math
import glob
import configparser
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# calib_suffix matches LEFT_CAM_<suffix> / RIGHT_CAM_<suffix> sections in the .conf file
# and CV_<suffix>, RX_<suffix>, RZ_<suffix> keys in [STEREO]
RESOLUTION_OPTIONS = {
    "HD2K":   ("2K",  2208, 1242),
    "HD1080": ("FHD", 1920, 1080),
    "HD720":  ("HD",  1280,  720),
    "VGA":    ("VGA",  672,  376),
}

# ---------------------------------------------------------------------------
# Calibration detection
# ---------------------------------------------------------------------------

def find_calibration_file():
    """Auto-detect *.conf in calibration/ next to this script. Returns path or None."""
    matches = glob.glob(os.path.join(SCRIPT_DIR, 'calibration', '*.conf'))
    if matches:
        print(f"[Calibration] Using {os.path.basename(matches[0])}")
        return matches[0]
    return None

# ---------------------------------------------------------------------------
# Calibration and rectification
# ---------------------------------------------------------------------------

def load_calibration_and_maps(conf_path, suffix, w, h):
    cfg = configparser.ConfigParser()
    cfg.read(conf_path)

    def cam_params(section):
        s = cfg[section]
        K = np.array([
            [float(s['fx']), 0.0,            float(s['cx'])],
            [0.0,            float(s['fy']), float(s['cy'])],
            [0.0,            0.0,            1.0           ],
        ], dtype=np.float64)
        D = np.array([
            float(s.get('k1', 0)), float(s.get('k2', 0)),
            float(s.get('p1', 0)), float(s.get('p2', 0)),
            float(s.get('k3', 0)),
        ], dtype=np.float64)
        return K, D

    K_l, D_l = cam_params(f'LEFT_CAM_{suffix}')
    K_r, D_r = cam_params(f'RIGHT_CAM_{suffix}')

    st  = cfg['STEREO']
    suf = suffix.lower()
    baseline = float(st['baseline'])
    cv_ang   = float(st.get(f'cv_{suf}', 0.0))
    rx_ang   = float(st.get(f'rx_{suf}', 0.0))
    rz_ang   = float(st.get(f'rz_{suf}', 0.0))
    ty       = float(st.get('ty', 0.0))   # not resolution-suffixed in .conf
    tz       = float(st.get('tz', 0.0))   # not resolution-suffixed in .conf

    R = (np.array([[math.cos(rz_ang), -math.sin(rz_ang), 0],
                   [math.sin(rz_ang),  math.cos(rz_ang), 0],
                   [0,                 0,                 1]], dtype=np.float64) @
         np.array([[math.cos(cv_ang),  0, math.sin(cv_ang)],
                   [0,                 1, 0               ],
                   [-math.sin(cv_ang), 0, math.cos(cv_ang)]], dtype=np.float64) @
         np.array([[1, 0,                0               ],
                   [0, math.cos(rx_ang), -math.sin(rx_ang)],
                   [0, math.sin(rx_ang),  math.cos(rx_ang)]], dtype=np.float64))
    T = np.array([[baseline], [ty], [tz]], dtype=np.float64)

    R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(
        K_l, D_l, K_r, D_r, (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map_l = cv2.initUndistortRectifyMap(K_l, D_l, R1, P1, (w, h), cv2.CV_32FC1)
    map_r = cv2.initUndistortRectifyMap(K_r, D_r, R2, P2, (w, h), cv2.CV_32FC1)
    return map_l, map_r

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def process_video(video_path, conf_path, calib_suffix, eye_w, eye_h, output_dir):
    os.makedirs(os.path.join(output_dir, "left_rectified"),  exist_ok=True)
    os.makedirs(os.path.join(output_dir, "right_rectified"), exist_ok=True)

    print(f"[Processing] {os.path.basename(video_path)}")
    print(f"[Resolution] {eye_w} x {eye_h} per eye  (suffix: {calib_suffix})")
    print(f"[Output]     {output_dir}")

    map_l, map_r = load_calibration_and_maps(conf_path, calib_suffix, eye_w, eye_h)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Frames]     {total_frames} total\n")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        left_raw  = frame[:, :eye_w]
        right_raw = frame[:, eye_w:]

        left_rectified  = cv2.remap(left_raw,  map_l[0], map_l[1], cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_raw, map_r[0], map_r[1], cv2.INTER_LINEAR)

        cv2.imwrite(
            os.path.join(output_dir, "left_rectified",  f"frame_{frame_idx:06d}.png"),
            left_rectified)
        cv2.imwrite(
            os.path.join(output_dir, "right_rectified", f"frame_{frame_idx:06d}.png"),
            right_rectified)

        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx * 100 // total_frames if total_frames > 0 else 0
            print(f"  Processed {frame_idx} / {total_frames} frames ({pct}%)...")

    cap.release()
    print(f"\nDone. {frame_idx} frames extracted to:\n  {output_dir}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python process.py <video_file> [HD2K|HD1080|HD720|VGA]")
        return

    video_path = os.path.abspath(sys.argv[1])
    if not os.path.isfile(video_path):
        print(f"[Error] File not found: {video_path}")
        return

    res_name = sys.argv[2].upper() if len(sys.argv) > 2 else "HD2K"
    if res_name not in RESOLUTION_OPTIONS:
        print(f"[Error] Unknown resolution '{res_name}'.")
        print(f"        Options: {', '.join(RESOLUTION_OPTIONS)}")
        return
    calib_suffix, eye_w, eye_h = RESOLUTION_OPTIONS[res_name]

    conf_path = find_calibration_file()
    if conf_path is None:
        calib_dir = os.path.join(SCRIPT_DIR, 'calibration')
        print(f"[Error] No calibration file found in {calib_dir}/")
        print("        Copy your SN*.conf file there and retry.")
        return

    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path),
                              f"{video_stem}_extracted_{calib_suffix}")

    process_video(video_path, conf_path, calib_suffix, eye_w, eye_h, output_dir)


if __name__ == "__main__":
    main()
