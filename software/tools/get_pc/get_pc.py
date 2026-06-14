"""
Point cloud assessment tool for the ZED camera.

Opens the ZED camera and displays a live colored point cloud in an interactive
3D viewer. Uses MILLIMETER units to match the ogl_viewer camera calibration.

Controls (in viewer window):
  v         Toggle voxel decimation / full point cloud
  +/-       Increase / decrease voxel size
  1/2/3     Voxel mode: FIXED / STEREO_UNCERTAINTY / LINEAR
  ,/.       Decrease / increase resolution scale
  c         Toggle centroid / grid center
  p/P       Decrease / increase point size
  d/D       Decrease / increase depth confidence
  r         Reset all parameters to defaults
  s         Save current point cloud to .ply in tools/recordings/
  q / Esc   Quit
"""

import sys
import os
from datetime import datetime
import tkinter as tk

import pyzed.sl as sl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ogl_viewer.viewer as gl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from _common import RESOLUTION_OPTIONS, _center_window

RECORDINGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'recordings'
)
os.makedirs(RECORDINGS_DIR, exist_ok=True)

MODE_MAP = [
    sl.VOXELIZATION_MODE.FIXED,
    sl.VOXELIZATION_MODE.STEREO_UNCERTAINTY,
    sl.VOXELIZATION_MODE.LINEAR,
]


def show_settings_dialog():
    """Resolution picker. Returns a RESOLUTION_OPTIONS tuple or None if closed."""
    result = [None]

    root = tk.Tk()
    root.title("ZED Point Cloud Settings")
    root.resizable(False, False)
    _center_window(root, 440, 220)

    tk.Label(root, text="Point Cloud Settings",
             font=("DejaVu Sans", 14, "bold")).pack(pady=(16, 10))

    frame = tk.LabelFrame(root, text="Resolution",
                          font=("DejaVu Sans", 10, "bold"), padx=10, pady=8)
    frame.pack(padx=20, fill='x')

    choice = tk.IntVar(value=0)
    for i, (name, _, w, h, fps) in enumerate(RESOLUTION_OPTIONS):
        tk.Radiobutton(frame, text=f"{name}   {w} x {h}  @  {fps} fps",
                       variable=choice, value=i,
                       font=("DejaVu Sans", 10)).pack(anchor='w')

    def on_start():
        result[0] = RESOLUTION_OPTIONS[choice.get()]
        root.destroy()

    tk.Button(root, text="Open Viewer", width=14, font=("DejaVu Sans", 11),
              command=on_start, bg="#1a3a5c", fg="white").pack(pady=14)

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return result[0]


def point_cloud_mode(resolution_option):
    name, resolution, w, h, fps = resolution_option

    init = sl.InitParameters(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        coordinate_units=sl.UNIT.MILLIMETER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
    )
    init.camera_resolution  = resolution
    init.depth_stabilization = True

    zed    = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"[Error] Camera open failed: {repr(status)}")
        return

    print(f"[Resolution] {name}  ({w} x {h} @ {fps} fps)")
    print("v=voxel/full  +/-=size  d/D=conf  p/P=pt size  s=save PLY  q=quit")

    use_gpu  = gl.GPU_ACCELERATION_AVAILABLE
    mem_type = sl.MEM.GPU if use_gpu else sl.MEM.CPU
    if use_gpu:
        print("[Info] GPU data transfer enabled (CuPy)")

    voxel_params                  = sl.VoxelMeasureParameters()
    voxel_params.voxel_size       = 50.0
    voxel_params.centroid         = True
    voxel_params.resolution_mode  = sl.VOXELIZATION_MODE.STEREO_UNCERTAINTY
    voxel_params.resolution_scale = 0.2

    point_cloud = sl.Mat()
    buf_res     = sl.Resolution()
    buf_res.width  = -1
    buf_res.height = -1
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, buf_res)
    buf_res = point_cloud.get_resolution()

    viewer = gl.GLViewer()
    viewer.init(1, sys.argv, buf_res)

    left_image  = sl.Mat()
    preview_res = sl.Resolution(640, 360)
    run_params  = sl.RuntimeParameters()

    try:
        while viewer.is_available():
            run_params.confidence_threshold = viewer.confidenceThreshold

            if zed.grab(run_params) > sl.ERROR_CODE.SUCCESS:
                continue

            # Sync voxel params from viewer HUD state
            voxel_params.voxel_size       = viewer.voxel_size
            voxel_params.resolution_scale = viewer.resolution_scale
            voxel_params.resolution_mode  = MODE_MAP[viewer.resolution_mode]
            voxel_params.centroid         = viewer.centroid

            # Handle keyboard controls
            key = viewer.key_pressed
            if key:
                if key in (ord('+'), ord('=')):
                    viewer.voxel_size *= 1.25
                elif key == ord('-'):
                    viewer.voxel_size = max(5.0, viewer.voxel_size * 0.8)
                elif key == ord('1'):
                    viewer.resolution_mode = 0
                elif key == ord('2'):
                    viewer.resolution_mode = 1
                elif key == ord('3'):
                    viewer.resolution_mode = 2
                elif key in (ord('.'), ord('>')):
                    viewer.resolution_scale = min(3.0, viewer.resolution_scale * 1.25)
                elif key in (ord(','), ord('<')):
                    viewer.resolution_scale = max(0.01, viewer.resolution_scale * 0.8)
                elif key in (ord('c'), ord('C')):
                    viewer.centroid = not viewer.centroid
                elif key == ord('p'):
                    viewer.pointSize = max(0.5, viewer.pointSize - 0.2)
                elif key == ord('P'):
                    viewer.pointSize = min(20.0, viewer.pointSize + 0.2)
                elif key == ord('d'):
                    viewer.confidenceThreshold = max(1, viewer.confidenceThreshold - 5)
                elif key == ord('D'):
                    viewer.confidenceThreshold = min(100, viewer.confidenceThreshold + 5)
                elif key in (ord('r'), ord('R')):
                    viewer.voxel_size          = 50.0
                    viewer.resolution_scale    = 0.2
                    viewer.resolution_mode     = 1
                    viewer.centroid            = True
                    viewer.pointSize           = 3.0
                    viewer.confidenceThreshold = 50
                    viewer.use_voxels          = True
                viewer.key_pressed = 0
                voxel_params.voxel_size       = viewer.voxel_size
                voxel_params.resolution_scale = viewer.resolution_scale
                voxel_params.resolution_mode  = MODE_MAP[viewer.resolution_mode]
                voxel_params.centroid         = viewer.centroid

            if viewer.use_voxels:
                zed.retrieve_voxel_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, voxel_params)
            else:
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, mem_type, buf_res)

            zed.retrieve_image(left_image, sl.VIEW.LEFT, sl.MEM.CPU, preview_res)
            viewer.updateData(point_cloud)
            viewer.updateImage(left_image)

            pt_count = point_cloud.get_width()
            if viewer.use_voxels:
                print(f"\rFPS: {zed.get_current_fps():.0f}"
                      f" | VOXEL | pts: {pt_count}"
                      f" | size: {viewer.voxel_size:.1f} mm"
                      f" | mode: {viewer.mode_names[viewer.resolution_mode]}"
                      f" | scale: {viewer.resolution_scale:.2f}"
                      f" | conf: {viewer.confidenceThreshold}"
                      f"        ", end='', flush=True)
            else:
                print(f"\rFPS: {zed.get_current_fps():.0f}"
                      f" | FULL PC | pts: {pt_count}"
                      f" | conf: {viewer.confidenceThreshold}"
                      f"        ", end='', flush=True)

            if viewer.save_data:
                pc_save = sl.Mat()
                if viewer.use_voxels:
                    zed.retrieve_voxel_measure(pc_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU, voxel_params)
                else:
                    zed.retrieve_measure(pc_save, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
                ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                ply_path = os.path.join(RECORDINGS_DIR, f"pc_{ts}.ply")
                err      = pc_save.write(ply_path)
                if err <= sl.ERROR_CODE.SUCCESS:
                    print(f"\n[Saved] {ply_path}  ({pc_save.get_width()} points)")
                else:
                    print(f"\n[Error] Failed to save PLY: {repr(err)}")
                viewer.save_data = False

    finally:
        print()
        viewer.exit()
        zed.close()


def main():
    settings = show_settings_dialog()
    if settings is None:
        return
    point_cloud_mode(settings)


if __name__ == "__main__":
    main()
