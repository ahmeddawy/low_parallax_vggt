#!/usr/bin/env python3
"""
dump_roi_crops.py — Save the ROI-cropped video that the model sees during ROI eval.

For each sequence with ae_data/corners.csv, computes the ROI bounding box
(same logic as eval_track_head_roi.py / eval_3d_roi.py), crops each frame,
resizes to the VGGT model resolution (518 × H_roi), and encodes as mp4.

Also saves a side-by-side video showing the original frame with the ROI box
drawn, next to the cropped ROI frame.

Usage:
    python dump_roi_crops.py \
        --dataset-dir /path/to/dataset \
        --val-split-file /path/to/val_split.txt \
        [--train-split-file /path/to/train_split.txt] \
        [--roi-pad 0.3] \
        [--output-dir roi_crops]
"""

import argparse
import glob
import os
import subprocess
import tempfile
import shutil

import cv2
import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description="Dump ROI-cropped videos")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--val-split-file", required=True)
    p.add_argument("--train-split-file", default=None)
    p.add_argument("--roi-pad", type=float, default=0.3)
    p.add_argument("--output-dir", default="roi_crops")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Cap sequences to this many frames (evenly subsampled). 0 = no limit.")
    return p.parse_args()


def load_roi_from_corners(seq_dir, pad_factor, W_orig, H_orig):
    corners_path = os.path.join(seq_dir, "ae_data", "corners.csv")
    if not os.path.isfile(corners_path):
        return None
    data = np.loadtxt(corners_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis]
    xs = np.concatenate([data[:, 1], data[:, 3], data[:, 5], data[:, 7]])
    ys = np.concatenate([data[:, 2], data[:, 4], data[:, 6], data[:, 8]])
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    x1 = max(0.0, xs.min() - w * pad_factor)
    y1 = max(0.0, ys.min() - h * pad_factor)
    x2 = min(W_orig, xs.max() + w * pad_factor)
    y2 = min(H_orig, ys.max() + h * pad_factor)
    return (x1, y1, x2, y2)


def get_sequence_fps(seq_dir, fallback=25.0):
    for name in ("original.mp4", "video.mp4"):
        vpath = os.path.join(seq_dir, name)
        if os.path.isfile(vpath):
            cap = cv2.VideoCapture(vpath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                return float(fps)
    return float(fallback)


def encode_video(frames_dir, out_path, fps):
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%06d.jpg"),
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p",
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def process_sequence(seq_name, dataset_dir, roi_pad, output_dir, max_frames):
    seq_dir = os.path.join(dataset_dir, seq_name)
    img_dir = os.path.join(seq_dir, "images")

    if not os.path.isdir(img_dir):
        print(f"  [{seq_name}] SKIP — no images dir")
        return

    image_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
    if not image_paths:
        print(f"  [{seq_name}] SKIP — no images")
        return

    img0 = Image.open(image_paths[0])
    W_orig, H_orig = img0.size

    roi = load_roi_from_corners(seq_dir, roi_pad, W_orig, H_orig)
    if roi is None:
        print(f"  [{seq_name}] SKIP — no corners.csv")
        return

    x1, y1, x2, y2 = roi
    ix1, iy1, ix2, iy2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    crop_w = ix2 - ix1
    crop_h = iy2 - iy1
    model_w = 518
    model_h = round(crop_h * model_w / crop_w / 14) * 14

    S = len(image_paths)
    if max_frames > 0 and S > max_frames:
        indices = np.linspace(0, S - 1, max_frames, dtype=int)
        image_paths = [image_paths[i] for i in indices]
        S = max_frames

    fps = get_sequence_fps(seq_dir)

    print(
        f"  [{seq_name}] {S} frames, ROI: ({ix1},{iy1})-({ix2},{iy2}) "
        f"crop: {crop_w}x{crop_h} -> model: {model_w}x{model_h} @ {fps:.1f}fps"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save ROI metadata
    import json as _json
    meta = {
        "seq_name": seq_name,
        "roi_pad": roi_pad,
        "original_resolution": {"W": W_orig, "H": H_orig},
        "roi_box_original_px": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)},
        "roi_crop_size": {"W": crop_w, "H": crop_h},
        "model_input_size": {"W": model_w, "H": model_h},
        "scale_factors": {"sx": model_w / crop_w, "sy": model_h / crop_h},
        "n_frames_original": len(sorted(glob.glob(os.path.join(img_dir, "*")))),
        "n_frames_used": S,
        "fps": fps,
    }
    meta_path = os.path.join(output_dir, f"{seq_name}_roi_meta.json")
    with open(meta_path, "w") as fh:
        _json.dump(meta, fh, indent=2)
    print(f"    -> {meta_path}")

    # Render frames
    tmp_crop = tempfile.mkdtemp(prefix="roi_crop_")
    tmp_side = tempfile.mkdtemp(prefix="roi_side_")

    try:
        for fi, img_path in enumerate(image_paths):
            img = Image.open(img_path).convert("RGB")
            orig_arr = np.array(img)

            # Cropped + resized (what the model sees)
            crop = img.crop((ix1, iy1, ix2, iy2))
            crop_resized = crop.resize((model_w, model_h), Image.Resampling.BICUBIC)
            crop_arr = np.array(crop_resized)

            # Save crop frame
            cv2.imwrite(
                os.path.join(tmp_crop, f"{fi:06d}.jpg"),
                crop_arr[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            # Side-by-side: original (with ROI box) | crop
            # Scale original to same height as crop for side-by-side
            scale = model_h / H_orig
            orig_resized = cv2.resize(orig_arr, (int(W_orig * scale), model_h))
            # Draw ROI box on resized original
            bx1 = int(x1 * scale)
            by1 = int(y1 * scale)
            bx2 = int(x2 * scale)
            by2 = int(y2 * scale)
            cv2.rectangle(orig_resized, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(orig_resized, "ROI", (bx1, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Divider
            divider = np.full((model_h, 2, 3), 180, dtype=np.uint8)
            side = np.concatenate([orig_resized, divider, crop_arr], axis=1)

            cv2.imwrite(
                os.path.join(tmp_side, f"{fi:06d}.jpg"),
                side[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

        # Encode videos
        crop_path = os.path.join(output_dir, f"{seq_name}_roi_crop.mp4")
        side_path = os.path.join(output_dir, f"{seq_name}_roi_sidebyside.mp4")

        if encode_video(tmp_crop, crop_path, fps):
            print(f"    -> {crop_path}")
        if encode_video(tmp_side, side_path, fps):
            print(f"    -> {side_path}")

    finally:
        shutil.rmtree(tmp_crop, ignore_errors=True)
        shutil.rmtree(tmp_side, ignore_errors=True)


def main():
    args = parse_args()

    splits = []
    with open(args.val_split_file) as f:
        val_seqs = [l.strip() for l in f if l.strip()]
    splits.append(("val", val_seqs))

    if args.train_split_file and os.path.isfile(args.train_split_file):
        with open(args.train_split_file) as f:
            train_seqs = [l.strip() for l in f if l.strip()]
        splits.append(("train", train_seqs))

    for split_name, sequences in splits:
        print(f"\n{'='*60}")
        print(f"SPLIT: {split_name.upper()} ({len(sequences)} sequences)")
        print(f"{'='*60}")

        split_out = os.path.join(args.output_dir, split_name)
        for seq in sequences:
            process_sequence(seq, args.dataset_dir, args.roi_pad, split_out, args.max_frames)

    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
