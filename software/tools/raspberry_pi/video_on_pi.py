import os
import sys
from datetime import datetime
import threading
import queue
import signal

import cv2

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

# (name, per-eye width, per-eye height, fps)
# UVC composite width = per-eye width * 2 (left|right side-by-side)
RESOLUTION_OPTIONS = [
    ("HD2K",   2208, 1242, 15),
    ("HD1080", 1920, 1080, 30),
    ("HD720",  1280,  720, 30),
    ("VGA",     672,  376, 30),
]

# ---------------------------------------------------------------------------
# Threaded Camera Capture
# ---------------------------------------------------------------------------

class ThreadedCameraUVC:
    def __init__(self, composite_width, height, fps):
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  composite_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, float(fps))

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != composite_width or actual_h != height:
            self.cap.release()
            raise RuntimeError(
                f"Resolution mismatch: requested {composite_width}x{height}, "
                f"got {actual_w}x{actual_h}. Is the ZED on a USB 3.0 (blue) port?"
            )

        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # 8 frames at HD2K = ~131 MB - safe headroom without OOM risk
        self.frame_queue = queue.Queue(maxsize=8)
        self.running = False
        self.thread  = None

    def start(self):
        if not self.cap.isOpened():
            return False
        self.running = True
        self.thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put(frame)

    def read_frame(self):
        try:
            return self.frame_queue.get(timeout=0.05)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()

# ---------------------------------------------------------------------------
# Recording pipeline
# ---------------------------------------------------------------------------

def run_raw_processor(resolution_option):
    name, eye_w, eye_h, fps = resolution_option
    composite_w = eye_w * 2

    try:
        cam_stream = ThreadedCameraUVC(composite_w, eye_h, fps)
    except RuntimeError as e:
        print(f"[Error] {e}")
        return

    if not cam_stream.start():
        print("[Error] Failed to open camera.")
        return

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(TOOLS_RECORDING_DIR, f"raw_stereo_{ts}.avi")

    actual_fps = cam_stream.actual_fps
    if actual_fps != float(fps):
        print(f"[Warning]   Camera FPS differs from requested: {fps} requested, "
              f"{actual_fps:.1f} actual. Using actual for VideoWriter.")

    if eye_w == 2208:
        print("[Warning]   HD2K + HFYU: AVI 2 GB limit reached in ~25 s. "
              "Use HD1080/HD720 for longer recordings.")

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'HFYU'),
        actual_fps,
        (composite_w, eye_h),
        True,
    )

    if not writer.isOpened():
        print("[Error] VideoWriter failed to open. Check codec availability and disk space.")
        cam_stream.stop()
        return

    print(f"[Recording] Started -> {video_path}")
    print("[Exit]      Press Ctrl+C to stop and save.\n")

    keep_running = True

    def signal_handler(sig, frame):
        nonlocal keep_running
        print("\n[Stopping]  Saving recording...")
        keep_running = False

    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while keep_running:
            frame = cam_stream.read_frame()
            if frame is None:
                continue
            writer.write(frame)
    finally:
        cam_stream.stop()
        while True:
            try:
                frame = cam_stream.frame_queue.get_nowait()
                writer.write(frame)
            except queue.Empty:
                break
        writer.release()
        size_mb = os.path.getsize(video_path) / 1e6
        print(f"[Saved]     {size_mb:.1f} MB -> {video_path}\n")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    res_name = sys.argv[1].upper() if len(sys.argv) > 1 else "HD2K"
    options  = {name: (name, w, h, fps) for name, w, h, fps in RESOLUTION_OPTIONS}

    if res_name not in options:
        print(f"[Error] Unknown resolution '{res_name}'.")
        print(f"        Options: {', '.join(options)}")
        return

    name, w, h, fps = options[res_name]
    print(f"[Config]    Resolution: {name}  ({w * 2} x {h} composite @ {fps} fps)")
    run_raw_processor(options[res_name])


if __name__ == "__main__":
    main()
