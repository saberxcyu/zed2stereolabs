import os
import sys
import time
import shutil
from datetime import datetime
import threading
import queue
import signal

import cv2

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

MIN_FREE_SPACE      = 2_000_000_000   # bytes - stop if < 2 GB free (covers ~25 s of HD2K HFYU)
DISK_CHECK_INTERVAL = 30.0            # seconds between disk-space checks
WARMUP_SECONDS      = 2.0             # discard frames while camera auto-exposure converges

# (name, per-eye width, per-eye height, fps)
# UVC composite width = per-eye width * 2 (left|right side-by-side)
RESOLUTION_OPTIONS = [
    ("HD2K",   2208, 1242,  8),
    ("HD1080", 1920, 1080, 10),
    ("HD720",  1280,  720, 20),
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

def run_raw_processor(resolution_option, use_mkv=False):
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

    if use_mkv:
        ext, fourcc_str, info = '.mkv', 'HFYU', 'MKV/HFYU lossless'
    else:
        ext, fourcc_str, info = '.mp4', 'mp4v', 'MP4/mp4v lossy'

    print(f"[Warmup]    {WARMUP_SECONDS:.0f}s (letting camera stabilize)...")
    time.sleep(WARMUP_SECONDS)
    while True:
        try:
            cam_stream.frame_queue.get_nowait()
        except queue.Empty:
            break

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(TOOLS_RECORDING_DIR, f"raw_stereo_{ts}{ext}")
    actual_fps = float(fps)
    print(f"[Info]      {info} - no file size limit. Stops if < 2 GB free.")

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*fourcc_str),
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

    written     = [0]
    write_queue = queue.Queue(maxsize=16)   # ~1s buffer at HD2K 15fps

    def enqueue_frame(f):
        if not write_queue.full():
            write_queue.put(f)
        else:
            try:
                write_queue.get_nowait()   # drop oldest
            except queue.Empty:
                pass
            write_queue.put(f)

    def write_loop():
        while True:
            try:
                f = write_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if f is None:   # sentinel
                break
            writer.write(f)
            written[0] += 1

    write_thread    = threading.Thread(target=write_loop, daemon=True)
    write_thread.start()
    record_start    = time.time()
    last_disk_check = 0.0

    try:
        while keep_running:
            frame = cam_stream.read_frame()
            if frame is None:
                continue
            enqueue_frame(frame)

            now = time.time()
            if now - last_disk_check >= DISK_CHECK_INTERVAL:
                last_disk_check = now
                free = shutil.disk_usage(TOOLS_RECORDING_DIR).free
                if free < MIN_FREE_SPACE:
                    print(f"\n[Warning]   Disk nearly full ({free // 1_000_000:.0f} MB free). Stopping.")
                    keep_running = False

    finally:
        cam_stream.stop()
        while True:
            try:
                frame = cam_stream.frame_queue.get_nowait()
                enqueue_frame(frame)
            except queue.Empty:
                break
        write_queue.put(None)
        write_thread.join(timeout=60)
        writer.release()
        duration   = time.time() - record_start
        actual_fps = written[0] / duration if duration > 0 else 0.0
        size_mb    = os.path.getsize(video_path) / 1e6
        print(f"[Saved]     {size_mb:.1f} MB -> {video_path}")
        print(f"[Stats]     {written[0]} frames  {duration:.1f}s  {actual_fps:.1f} fps  (target {fps} fps)")
        if written[0] > 0 and actual_fps < fps * 0.95:
            print(f"[Warning]   Write rate {actual_fps:.1f} fps below target {fps} fps. Video may be sped up.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    options    = {name: (name, w, h, fps) for name, w, h, fps in RESOLUTION_OPTIONS}
    argv_lower = [a.lower() for a in sys.argv[1:]]
    use_mkv    = 'mkv' in argv_lower
    res_args   = [a.upper() for a in argv_lower if a != 'mkv']

    unknown = [a for a in res_args if a not in options]
    if unknown:
        print(f"[Error] Unknown argument(s): {', '.join(unknown)}")
        print(f"        Resolution options: {', '.join(options)}")
        print(f"        Format options: mkv (default: mp4)")
        return

    res_name = res_args[0] if res_args else 'HD2K'
    name, w, h, fps = options[res_name]
    fmt = 'MKV lossless' if use_mkv else 'MP4 lossy'
    print(f"[Config]    Resolution: {name}  ({w * 2} x {h} composite @ {fps} fps)  Format: {fmt}")
    run_raw_processor(options[res_name], use_mkv=use_mkv)


if __name__ == "__main__":
    main()
