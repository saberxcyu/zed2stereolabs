import sys
import os
import json
import math
import shutil
import time
import argparse
import contextlib
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
from datetime import datetime, timezone

import cv2
import numpy as np
import pyzed.sl as sl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _common import RESOLUTION_OPTIONS, draw_status

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

SINGLE_WINDOW_NAME = "ZED Stereo Viewer  |  Left Rectified  +  Right Rectified"
MULTI_WINDOW_NAME  = "ZED Multi-Camera Viewer  |  Left Rectified"

SVO_COMPRESSION_OPTIONS = {
    "H264":     sl.SVO_COMPRESSION_MODE.H264,
    "H265":     sl.SVO_COMPRESSION_MODE.H265,
    "LOSSLESS": sl.SVO_COMPRESSION_MODE.LOSSLESS,
}

CAMERA_NAMES = {
    40932664: "phoenix",   # Reference camera
    41248751: "tortoise",  # Across from phoenix (-Z)
    48029520: "dragon",    # Left of phoenix (-X)
    44577743: "tiger",     # Right of phoenix (+X)
}


def camera_name(serial: int) -> str:
    return CAMERA_NAMES.get(int(serial), f"cam_{int(serial)}")


def camera_label(serial: int) -> str:
    return f"{camera_name(serial)} ({int(serial)})"


@contextlib.contextmanager
def _suppress_c_output():
    """Silence C-level stdout/stderr (e.g. Jetson NVENC / NvMMLite chatter) around ZED calls."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)
        os.close(devnull_fd)


class CameraWorker:
    """Manages one opened ZED camera, its grab thread, and optional preview frame."""

    def __init__(self, index: int, serial: int, model_name: str, zed: sl.Camera,
                 preview_res: sl.Resolution, stereo_preview: bool,
                 enable_preview: bool = True, preview_stride: int = 3):
        self.index = index
        self.serial = serial
        self.model_name = model_name
        self.zed = zed
        self.preview_res = preview_res
        self.stereo_preview = stereo_preview
        self.enable_preview = enable_preview
        self.preview_stride = max(1, preview_stride)

        cfg = zed.get_camera_information().camera_configuration
        self.width = int(cfg.resolution.width)
        self.height = int(cfg.resolution.height)
        self.fps = float(cfg.fps)

        self.op_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.current_fps = 0.0
        self.grab_count = 0

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=3.0)

    def _run(self):
        runtime_params = sl.RuntimeParameters()
        left_mat = sl.Mat()
        right_mat = sl.Mat() if self.stereo_preview else None

        try:
            n = 0
            while not self._stop_event.is_set():
                with self.op_lock:
                    status = self.zed.grab(runtime_params)
                    if status != sl.ERROR_CODE.SUCCESS:
                        continue
                    n += 1
                    frame = None
                    if self.enable_preview and (n % self.preview_stride == 0):
                        self.zed.retrieve_image(left_mat, sl.VIEW.LEFT, sl.MEM.CPU, self.preview_res)
                        left_bgr = cv2.cvtColor(left_mat.get_data(), cv2.COLOR_BGRA2BGR)
                        if self.stereo_preview:
                            self.zed.retrieve_image(right_mat, sl.VIEW.RIGHT, sl.MEM.CPU, self.preview_res)
                            right_bgr = cv2.cvtColor(right_mat.get_data(), cv2.COLOR_BGRA2BGR)
                            frame = np.concatenate([left_bgr, right_bgr], axis=1)
                        else:
                            frame = left_bgr
                    cur_fps = self.zed.get_current_fps()

                with self.frame_lock:
                    if frame is not None:
                        self.latest_frame = frame
                    self.current_fps = cur_fps
                    self.grab_count += 1
        finally:
            left_mat.free(sl.MEM.CPU)
            if right_mat is not None:
                right_mat.free(sl.MEM.CPU)

    def get_preview(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None, self.current_fps
            return self.latest_frame.copy(), self.current_fps

    def enable_recording(self, svo_path: str, compression_name: str):
        rec_params = sl.RecordingParameters()
        rec_params.video_filename = svo_path
        rec_params.compression_mode = SVO_COMPRESSION_OPTIONS[compression_name]
        with self.op_lock, _suppress_c_output():
            err = self.zed.enable_recording(rec_params)
            if err == sl.ERROR_CODE.SUCCESS:
                # Trigger first grab inside _suppress_c_output to silence lazy "NvVideo: bBlitMode" init log
                self.zed.grab(sl.RuntimeParameters())
            return None if err == sl.ERROR_CODE.SUCCESS else repr(err)

    def disable_recording(self):
        with self.op_lock, _suppress_c_output():
            self.zed.disable_recording()

    def close(self):
        self.stop()
        with self.op_lock, _suppress_c_output():
            self.zed.close()


def _worker_proc(serial: int, res_name: str, target_fps: int, tile_w: int, tile_h: int,
                 enable_preview: bool, preview_stride: int,
                 cmd_q, ack_q, shm_name, shm_lock, fps_val, has_frame):
    """Subprocess entry point: owns one ZED camera so NVENC/NvMedia2D contexts stay isolated."""
    # Silence C-level NVENC/V4L2 chatter (bBlitMode, dqthread timeouts) in worker subprocesses
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)

    res_enum = next(opt[1] for opt in RESOLUTION_OPTIONS if opt[0] == res_name)
    init = sl.InitParameters()
    init.set_from_serial_number(serial)
    init.camera_resolution = res_enum
    init.camera_fps = target_fps
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.sdk_verbose = 0

    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        ack_q.put(("ready", False, 0, 0, 0.0, repr(status)))
        return

    cfg = zed.get_camera_information().camera_configuration
    ack_q.put(("ready", True, int(cfg.resolution.width), int(cfg.resolution.height), float(cfg.fps), ""))

    shm = shared_memory.SharedMemory(name=shm_name) if (enable_preview and shm_name) else None
    shm_arr = np.ndarray((tile_h, tile_w, 3), dtype=np.uint8, buffer=shm.buf) if shm is not None else None
    preview_res = sl.Resolution(tile_w, tile_h)
    runtime_params = sl.RuntimeParameters()
    left_mat = sl.Mat()
    recording = False
    n = 0

    try:
        while True:
            quit_now = False
            while True:
                try:
                    cmd, arg = cmd_q.get_nowait()
                except Exception:
                    break
                if cmd == "start_rec":
                    svo_path, comp_name = arg
                    rec_params = sl.RecordingParameters()
                    rec_params.video_filename = svo_path
                    rec_params.compression_mode = SVO_COMPRESSION_OPTIONS[comp_name]
                    err = zed.enable_recording(rec_params)
                    if err == sl.ERROR_CODE.SUCCESS:
                        recording = True
                        ack_q.put(("rec_started", None))
                    else:
                        ack_q.put(("rec_started", repr(err)))
                elif cmd == "stop_rec":
                    if recording:
                        zed.disable_recording()
                        recording = False
                    ack_q.put(("rec_stopped", True))
                elif cmd == "quit":
                    quit_now = True
                    break
            if quit_now:
                break

            if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                continue
            n += 1
            if shm_arr is not None and (n % preview_stride == 0):
                zed.retrieve_image(left_mat, sl.VIEW.LEFT, sl.MEM.CPU, preview_res)
                bgr = cv2.cvtColor(left_mat.get_data(), cv2.COLOR_BGRA2BGR)
                with shm_lock:
                    shm_arr[:] = bgr
                    has_frame.value = 1
            fps_val.value = float(zed.get_current_fps())
    finally:
        left_mat.free(sl.MEM.CPU)
        if recording:
            zed.disable_recording()
        zed.close()
        if shm is not None:
            shm.close()


class ProcessCameraWorker:
    """Manages one ZED camera in a dedicated spawned subprocess (for multi-cam Jetson stability)."""

    def __init__(self, ctx, index: int, serial: int, model_name: str,
                 res_name: str, target_fps: int, tile_w: int = 640, tile_h: int = 400,
                 enable_preview: bool = True, preview_stride: int = 3):
        self.index = index
        self.serial = serial
        self.model_name = model_name
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.enable_preview = enable_preview

        self.cmd_q = ctx.Queue()
        self.ack_q = ctx.Queue()
        self.fps_val = ctx.Value('d', 0.0)
        self.has_frame = ctx.Value('b', 0)
        self.shm_lock = ctx.Lock()
        self.shm = shared_memory.SharedMemory(create=True, size=tile_h * tile_w * 3) if enable_preview else None
        self.shm_arr = (
            np.ndarray((tile_h, tile_w, 3), dtype=np.uint8, buffer=self.shm.buf)
            if self.shm is not None else None
        )

        self._proc = ctx.Process(
            target=_worker_proc,
            args=(
                serial, res_name, target_fps, tile_w, tile_h, enable_preview, preview_stride,
                self.cmd_q, self.ack_q, self.shm.name if self.shm else None,
                self.shm_lock, self.fps_val, self.has_frame,
            ),
            daemon=True,
        )
        self._proc.start()

        tag, ok, width, height, actual_fps, err_msg = self.ack_q.get(timeout=30.0)
        if tag != "ready" or not ok:
            self._proc.join(timeout=3.0)
            self._cleanup_shm()
            raise RuntimeError(err_msg or "Failed to open camera")

        self.width = width
        self.height = height
        self.fps = actual_fps

    def start(self):
        pass  # Subprocess starts grabbing immediately after open()

    def get_preview(self):
        if self.shm_arr is None or not self.has_frame.value:
            return None, float(self.fps_val.value)
        with self.shm_lock:
            return self.shm_arr.copy(), float(self.fps_val.value)

    def enable_recording(self, svo_path: str, compression_name: str):
        self.cmd_q.put(("start_rec", (svo_path, compression_name)))
        _, err_msg = self.ack_q.get(timeout=15.0)
        return err_msg

    def disable_recording(self):
        self.cmd_q.put(("stop_rec", None))
        self.ack_q.get(timeout=15.0)

    def _cleanup_shm(self):
        if self.shm is not None:
            try:
                self.shm.close()
            finally:
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass
                self.shm = None
                self.shm_arr = None

    def close(self):
        if self._proc.is_alive():
            try:
                self.cmd_q.put(("quit", None))
                self._proc.join(timeout=5.0)
            finally:
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=2.0)
        self._cleanup_shm()


def select_devices(serials=None):
    """Discover connected ZED cameras, optionally filtered/ordered by --serials."""
    devs = sl.Camera.get_device_list()
    if not devs:
        return []
    if not serials:
        return sorted(devs, key=lambda d: int(d.serial_number))
    by_sn = {int(d.serial_number): d for d in devs}
    selected = []
    for sn in serials:
        if sn not in by_sn:
            raise RuntimeError(f"Requested serial {sn} not found. Connected: {list(by_sn.keys())}")
        selected.append(by_sn[sn])
    return selected


def default_resolution_for_devices(devs):
    """Default to HD1200 for ZED X cameras and HD2K otherwise."""
    if any("ZED X" in str(d.camera_model) for d in devs):
        return "HD1200"
    return "HD2K"


def open_workers(devs, res_name: str, res_enum, target_fps: int, enable_preview: bool = True):
    """Open all selected cameras sequentially (in-process for 1 cam, spawned subprocesses for N>1)."""
    workers = []
    stereo_preview = len(devs) == 1
    tile_w, tile_h = (960, 540) if stereo_preview else (640, 400)
    preview_res = sl.Resolution(tile_w, tile_h)
    preview_stride = 1
    ctx = None if stereo_preview else mp.get_context("spawn")

    for idx, dev in enumerate(devs):
        sn = int(dev.serial_number)
        model_name = str(dev.camera_model)
        if stereo_preview:
            init = sl.InitParameters()
            init.set_from_serial_number(sn)
            init.camera_resolution = res_enum
            init.camera_fps = target_fps
            init.depth_mode = sl.DEPTH_MODE.NONE
            init.sdk_verbose = 0

            zed = sl.Camera()
            with _suppress_c_output():
                status = zed.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print(f"[Error] Failed to open {camera_label(sn)} ({model_name}): {repr(status)}")
                for w in workers:
                    w.close()
                return None

            worker = CameraWorker(
                idx, sn, model_name, zed, preview_res, stereo_preview,
                enable_preview=enable_preview, preview_stride=preview_stride,
            )
        else:
            try:
                worker = ProcessCameraWorker(
                    ctx, idx, sn, model_name, res_name, target_fps,
                    tile_w=tile_w, tile_h=tile_h,
                    enable_preview=enable_preview, preview_stride=preview_stride,
                )
            except Exception as exc:
                print(f"[Error] Failed to open {camera_label(sn)} ({model_name}): {exc}")
                for w in workers:
                    w.close()
                return None

        print(f"[Opened]    [{idx}] {camera_label(sn)} {model_name} ({worker.width}x{worker.height} @ {worker.fps:.0f} fps)")
        workers.append(worker)

    return workers


def build_grid(workers, tile_w=640, tile_h=400):
    """Arrange N camera previews into a grid with camera & serial overlays."""
    n = len(workers)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, w in enumerate(workers):
        frame, _ = w.get_preview()
        r, c = divmod(idx, cols)
        y0, x0 = r * tile_h, c * tile_w
        if frame is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        else:
            tile = frame if (frame.shape[1] == tile_w and frame.shape[0] == tile_h) else cv2.resize(
                frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA
            )
        label = camera_label(w.serial)
        cv2.putText(tile, label, (10, tile_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        canvas[y0:y0 + tile_h, x0:x0 + tile_w] = tile

    return canvas


def start_recording_session(workers, compression_name, res_name, target_fps):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
    outdir = os.path.join(TOOLS_RECORDING_DIR, f"svo_{ts}")
    os.makedirs(outdir, exist_ok=True)

    svo_paths = []
    started = []
    for w in workers:
        fname = "recording.svo2" if len(workers) == 1 else f"{camera_name(w.serial)}.svo2"
        svo_path = os.path.join(outdir, fname)
        err_msg = w.enable_recording(svo_path, compression_name)
        if err_msg is not None:
            print(f"[Error]     Could not start SVO recording on {camera_label(w.serial)}: {err_msg}")
            for sw in started:
                sw.disable_recording()
            shutil.rmtree(outdir, ignore_errors=True)
            return None, None, None
        started.append(w)
        svo_paths.append((w, fname, svo_path))

    session_meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolution": res_name,
        "target_fps": int(target_fps),
        "compression": compression_name,
        "cameras": [
            {
                "index": w.index,
                "name": camera_name(w.serial),
                "serial_number": w.serial,
                "model": w.model_name,
                "width": w.width,
                "height": w.height,
                "actual_fps": w.fps,
                "svo_file": fname,
            }
            for w, fname, _ in svo_paths
        ],
    }
    with open(os.path.join(outdir, "session.json"), "w") as f:
        json.dump(session_meta, f, indent=2)

    return ts, outdir, svo_paths


def stop_recording_session(workers, svo_paths, rec_start, outdir):
    for w in workers:
        w.disable_recording()
    elapsed = time.monotonic() - rec_start
    total_mb = sum(os.path.getsize(p) for _, _, p in svo_paths if os.path.exists(p)) / 1e6
    print()
    print(f"[Saved]     {total_mb:.1f} MB total ({len(workers)} SVO2), {elapsed:.1f}s -> {outdir}")
    return total_mb, elapsed


def record_mode(devs, resolution_option, target_fps: int, compression_type, compression_name: str, no_preview: bool = False):
    """Open 1..N ZED cameras, display preview (unless --no-preview), record SVO2 on 's'."""
    res_name, res_enum, _, _, _ = resolution_option
    workers = open_workers(devs, res_name, res_enum, target_fps, enable_preview=not no_preview)
    if workers is None:
        return 'back'

    for w in workers:
        w.start()

    recording = False
    ts = None
    outdir = None
    svo_paths = None
    rec_start = None
    exit_reason = 'back'

    window_name = SINGLE_WINDOW_NAME if len(workers) == 1 else MULTI_WINDOW_NAME
    if not no_preview:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        if len(workers) == 1:
            cv2.resizeWindow(window_name, 1920, 540)
        else:
            cols = math.ceil(math.sqrt(len(workers)))
            rows = math.ceil(len(workers) / cols)
            cv2.resizeWindow(window_name, min(1920, cols * 640), min(1080, rows * 400))
        print("Press 's' to start/stop recording | 'q' to quit")
    else:
        print("Headless mode: press Enter to start/stop recording | 'q' + Enter to quit")

    status_msg = 'Press [s] to start recording - [q] to quit'
    status_color = (200, 200, 200)

    try:
        while True:
            if not no_preview:
                if len(workers) == 1:
                    frame, _ = workers[0].get_preview()
                    display_frame = frame if frame is not None else np.zeros((540, 1920, 3), dtype=np.uint8)
                else:
                    display_frame = build_grid(workers)

                draw_status(display_frame, status_msg, status_color, recording)
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(15) & 0xFF

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    exit_reason = 'quit'
                    break
            else:
                cmd = input().strip().lower()
                key = ord('q') if cmd == 'q' else ord('s')

            if recording:
                elapsed = time.monotonic() - rec_start
                print(f"\r[Recording] {elapsed:6.1f}s", end='', flush=True)

            if key == ord('s'):
                if not recording:
                    ts, outdir, svo_paths = start_recording_session(
                        workers, compression_name, res_name, target_fps
                    )
                    if outdir is not None:
                        recording = True
                        rec_start = time.monotonic()
                        print(f"[Recording] {outdir}")
                        status_msg = f'Recording {len(workers)} cam(s) - press [s] to stop'
                        status_color = (0, 80, 255)
                else:
                    stop_recording_session(workers, svo_paths, rec_start, outdir)
                    recording = False
                    status_msg = f'Saved: svo_{ts}/ - press [s] to record again'
                    status_color = (0, 210, 0)
                    outdir = None
                    svo_paths = None
                    rec_start = None

            elif key == ord('q'):
                exit_reason = 'back'
                break

    finally:
        if recording and outdir:
            for w in workers:
                w.disable_recording()
            shutil.rmtree(outdir, ignore_errors=True)
            print("\n[Discarded] Recording cancelled - no file saved.")
        for w in workers:
            w.close()
        if not no_preview:
            cv2.destroyAllWindows()

    return exit_reason


def main():
    parser = argparse.ArgumentParser(description="Capture stereo video from 1..N ZED cameras, recording to SVO2 on 's'.")
    parser.add_argument('--resolution', choices=[opt[0] for opt in RESOLUTION_OPTIONS], default=None,
                        help="Camera resolution (default: HD1200 for ZED X, HD2K otherwise)")
    parser.add_argument('--fps', type=int, default=None,
                        help="Camera FPS (default: resolution default, e.g. 30 for HD1200)")
    parser.add_argument('--serials', type=int, nargs='+', default=None,
                        help="Specific camera serial number(s) to open (default: all connected)")
    parser.add_argument('--compression', choices=list(SVO_COMPRESSION_OPTIONS.keys()), default=None,
                        help="SVO compression mode (default: H265 for ZED X, H264 otherwise)")
    parser.add_argument('--no-preview', action='store_true',
                        help="Disable OpenCV preview window (use Enter / 'q' in terminal)")
    args = parser.parse_args()

    devs = select_devices(args.serials)
    if not devs:
        print("[Error] No ZED cameras detected.")
        sys.exit(1)

    has_zedx = any("ZED X" in str(d.camera_model) for d in devs)
    res_name = args.resolution or ("HD1200" if has_zedx else "HD2K")
    compression_name = args.compression or ("H265" if has_zedx else "H264")
    res_option = next(opt for opt in RESOLUTION_OPTIONS if opt[0] == res_name)
    target_fps = args.fps if args.fps is not None else res_option[4]
    compression_type = SVO_COMPRESSION_OPTIONS[compression_name]

    record_mode(
        devs,
        res_option,
        target_fps=target_fps,
        compression_type=compression_type,
        compression_name=compression_name,
        no_preview=args.no_preview,
    )


if __name__ == "__main__":
    main()
