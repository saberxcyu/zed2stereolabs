import sys
import os
import argparse
from collections import Counter

import numpy as np
import tkinter as tk
from tkinter import filedialog

TOOLS_RECORDING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recordings')
os.makedirs(TOOLS_RECORDING_DIR, exist_ok=True)

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
SAM3_CHECKPOINT = os.path.join(MODELS_DIR, 'sam3.pt')


def resolve_folder(arg):
    if arg is None:
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title="Select recording folder", initialdir=TOOLS_RECORDING_DIR)
        root.destroy()
        return folder or None
    if os.path.isdir(arg):
        return arg
    print(f"[Error] Not a directory: {arg}")
    return None


def build_results_stream(checkpoint, video_path, text, mask_conf, imgsz, preview, outdir):
    """Return a per-frame Results generator from SAM3's open-vocabulary text-prompted
    video predictor. Yields standard Ultralytics Results objects with the same
    .boxes.id / .masks.data interface consumed below."""
    if not os.path.exists(checkpoint):
        print(f"[Error] sam3.pt not found at {checkpoint}.")
        print("        Request access at https://huggingface.co/facebook/sam3 and download it there.")
        sys.exit(1)

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    # SAM3's ViT backbone has stride=14; Ultralytics rounds imgsz up to the nearest multiple
    # of stride automatically (unset defaults to the global 640 -> rounds to 644, which is too
    # coarse for HD2K silhouette edges). SAM models require a square imgsz and stretch (not
    # letterbox-pad) the frame to fill it, so this is the only real lever over edge fidelity --
    # see CLAUDE.md for the reasoning. Higher imgsz costs roughly quadratically more compute.
    overrides = {
        "conf": mask_conf, "task": "segment", "mode": "predict",
        "model": checkpoint, "quantize": 16, "imgsz": imgsz,
    }
    if preview:
        # write_results() -- which draws the mask overlay and saves it -- runs inside the
        # same per-frame loop the caller iterates below, so this needs no extra code of
        # its own. save_dir (not project/name) is used as-is, bypassing Ultralytics'
        # global auto-incrementing runs/ tree.
        overrides["save"] = True
        overrides["save_dir"] = os.path.join(outdir, "sam3_preview")
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    return predictor(source=video_path, text=[text], stream=True)


def segment(outdir, checkpoint, text, mask_conf, imgsz, preview):
    video_path = os.path.join(outdir, "video.mp4")
    meta_path  = os.path.join(outdir, "pointcloud_meta.npz")
    if not os.path.exists(video_path) or not os.path.exists(meta_path):
        print(f"[Error] {video_path} / {meta_path} not found -- run process.py first.")
        sys.exit(1)

    meta = np.load(meta_path)
    H, W = int(meta['height']), int(meta['width'])
    timestamps_ns = meta['timestamps_ns']
    n_expected = timestamps_ns.shape[0]

    results = build_results_stream(checkpoint, video_path, text, mask_conf, imgsz, preview, outdir)

    # Single pass over the video: buffer every detected instance's mask (bit-packed to
    # keep RAM down) per frame, plus a running count of which track ID persists across
    # the most frames. Re-running SAM3 a second time just to pick the subject would double
    # GPU cost, so the whole clip is held in RAM instead -- validate on a short clip first
    # (same caution CLAUDE.md already gives for pointcloud.bin's own RAM/disk footprint).
    frame_records = []
    id_counts = Counter()

    frame_idx = 0
    for r in results:
        record = {}
        if r.masks is not None and r.boxes is not None and r.boxes.id is not None:
            ids   = r.boxes.id.cpu().numpy().astype(np.int64)
            masks = r.masks.data.cpu().numpy().astype(np.uint8)
            for obj_id, mask in zip(ids, masks):
                record[int(obj_id)] = np.packbits(mask.ravel())
                id_counts[int(obj_id)] += 1
        frame_records.append(record)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"\r[Segmenting] {frame_idx}/{n_expected} frames", end='', flush=True)
    print()

    n_frames = len(frame_records)
    if n_frames != n_expected:
        print(f"[Error] Frame count mismatch: video.mp4 decoded to {n_frames} frames, "
              f"but pointcloud_meta.npz expects {n_expected}.")
        sys.exit(1)

    if not id_counts:
        print(f"[Warning] No person instances detected in the whole clip -- mask.bin will be all-empty.")
        primary_id = None
    else:
        primary_id, n_present = id_counts.most_common(1)[0]
        print(f"[Selected] primary instance id={primary_id}  (present in {n_present}/{n_frames} frames)")

    mask_path = os.path.join(outdir, "mask.bin")
    empty_mask = np.zeros((H, W), dtype=np.uint8)
    with open(mask_path, "wb") as f:
        for record in frame_records:
            if primary_id is not None and primary_id in record:
                packed = record[primary_id]
                mask = np.unpackbits(packed)[:H * W].reshape(H, W)
                f.write((mask * 255).astype(np.uint8).tobytes())
            else:
                f.write(empty_mask.tobytes())

    mask_meta_path = os.path.join(outdir, "mask_meta.npz")
    np.savez_compressed(
        mask_meta_path,
        timestamps_ns = timestamps_ns,
        height        = np.int32(H),
        width         = np.int32(W),
        model         = 'sam3',
        checkpoint    = checkpoint,
        text_prompt   = text,
        svo_source    = str(meta['svo_source']),
    )

    mask_mb = os.path.getsize(mask_path) / 1e6
    print(f"[Saved] {mask_path}  ({mask_mb:.1f} MB, {n_frames} frames)")
    print(f"[Saved] {mask_meta_path}")
    if preview:
        print(f"[Saved] {os.path.join(outdir, 'sam3_preview')}  (annotated preview video, for visual QC only)")


def main():
    parser = argparse.ArgumentParser(
        description="Segment the recording's person with SAM3, writing a per-frame foreground mask.")
    parser.add_argument('path', nargs='?', default=None,
                         help="Recording folder (containing video.mp4). Omit for a folder picker.")
    parser.add_argument('--checkpoint', default=SAM3_CHECKPOINT,
                         help=f"SAM3 model checkpoint (default {SAM3_CHECKPOINT}, must be downloaded manually -- "
                              "see https://huggingface.co/facebook/sam3)")
    parser.add_argument('--text', default="person",
                         help="Concept prompt (default 'person')")
    parser.add_argument('--conf', type=float, default=0.25,
                         help="Detection confidence threshold (default 0.25)")
    parser.add_argument('--imgsz', type=int, default=1246,
                         help="Square input resolution fed to SAM3 (default 1246, ~= HD2K's 1242px height; "
                              "auto-rounded up to a multiple of SAM3's stride=14). Higher = better edge "
                              "fidelity, roughly quadratically more compute. Unset elsewhere defaults to "
                              "Ultralytics' global 640 -> rounds to 644, which is too coarse for HD2K.")
    parser.add_argument('--no-preview', action='store_true',
                         help="Skip writing the annotated sam3_preview/ QC video (on by default)")
    args = parser.parse_args()

    outdir = resolve_folder(args.path)
    if outdir is None:
        print("[Cancelled] No folder selected.")
        sys.exit(1)

    segment(outdir, args.checkpoint, args.text, args.conf, args.imgsz, preview=not args.no_preview)


if __name__ == "__main__":
    main()
