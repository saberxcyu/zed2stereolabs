import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import filedialog

from _common import show_mode_dialog, show_record_settings_dialog, show_pose_settings_dialog, show_pc_settings_dialog
from _common import RECORDING_DIR
from _depth_record  import depth_record_mode
from _depth_analyze import depth_analyze_mode
from _pose_record   import pose_record_mode
from _pose_analyze  import pose_analyze_mode
from _pc_record     import pc_record_mode
from _pc_analyze    import pc_analyze_mode


def main():
    while True:
        mode = show_mode_dialog()
        if mode is None:          # X on mode dialog -> exit
            break

        if mode == 'record':
            settings = show_record_settings_dialog()
            if settings is None:  # X on settings dialog -> back to menu
                continue
            resolution, min_d, max_d = settings
            result = depth_record_mode(min_d, max_d, resolution)
            if result == 'quit':  # window X -> exit
                break
            # result == 'back' (q pressed) -> loop back to menu

        elif mode == 'analyze':
            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory(title="Select recording folder", initialdir=RECORDING_DIR)
            root.destroy()
            if not folder:        # cancelled -> back to menu
                continue
            result = depth_analyze_mode(folder)
            if result == 'quit':  # window X -> exit
                break
            # 'back' (q pressed) -> loop back to menu

        elif mode == 'pose':
            settings = show_pose_settings_dialog()
            if settings is None:  # X on settings dialog -> back to menu
                continue
            resolution, body_fmt = settings
            result = pose_record_mode(resolution, body_fmt)
            if result == 'quit':  # window X -> exit
                break
            # 'back' (q pressed) -> loop back to menu

        elif mode == 'pose_analyze':
            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory(title="Select pose recording folder", initialdir=RECORDING_DIR)
            root.destroy()
            if not folder:        # cancelled -> back to menu
                continue
            result = pose_analyze_mode(folder)
            if result == 'quit':  # window X -> exit
                break
            # 'back' (q pressed) -> loop back to menu

        elif mode == 'pc_record':
            settings = show_pc_settings_dialog()
            if settings is None:  # X on settings dialog -> back to menu
                continue
            resolution, depth_min, depth_max = settings
            result = pc_record_mode(resolution, depth_min, depth_max)
            if result == 'quit':  # window X -> exit
                break
            # 'back' (q pressed) -> loop back to menu

        elif mode == 'pc_analyze':
            root = tk.Tk()
            root.withdraw()
            folder = filedialog.askdirectory(title="Select PC recording folder", initialdir=RECORDING_DIR)
            root.destroy()
            if not folder:        # cancelled -> back to menu
                continue
            result = pc_analyze_mode(folder)
            if result == 'quit':  # window X -> exit
                break
            # 'back' (q pressed) -> loop back to menu


if __name__ == "__main__":
    main()
