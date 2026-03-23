#!/usr/bin/env python3
"""
Apply an AE plane mask to depth maps, zeroing out the plane region.

The depth loss should NOT supervise the plane area because the plane's depth
from ViPE is often unreliable (object insertion surface, occlusions, etc.).
This script keeps background depth valid and zeros the plane.

Supported mask formats:
  1. NumPy array: masks.npy of shape (N_frames, H, W), dtype uint8 or bool
                  1 = plane (will be zeroed), 0 = background (kept)
  2. Image folder: a directory of per-frame PNG/JPG masks named numerically
                   (e.g. 0000.png, 0001.png, ...). White (>127) = plane.

Input depth: <seq_dir>/depth/depths.npy  — shape (N_frames, H, W), float16/32
Output:      <seq_dir>/masked_depth/depths.npy — same shape, plane pixels = 0

Usage examples:
    # Numpy mask, entire dataset
    python apply_plane_mask_to_depth.py \\
        --dataset-dir /data/my_dataset \\
        --mask-type npy

    # Image folder mask, single sequence
    python apply_plane_mask_to_depth.py \\
        --dataset-dir /data/my_dataset/seq_001 \\
        --mask-type folder \\
        --mask-subdir ae_mask

    # Custom depth filename
    python apply_plane_mask_to_depth.py \\
        --dataset-dir /data/my_dataset \\
        --mask-type npy \\
        --depth-filename depths.npy \\
        --mask-filename plane_mask.npy
"""

import argparse
import os
import os.path as osp

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Mask loading helpers
# ---------------------------------------------------------------------------

def load_mask_npy(mask_path, n_frames):
    """Load mask from a .npy file. Returns (N_frames, H, W) bool array."""
    masks = np.load(mask_path)
    if masks.ndim == 2:
        # Single mask broadcast to all frames
        masks = np.broadcast_to(masks[None], (n_frames,) + masks.shape)
    assert masks.shape[0] == n_frames, (
        f"Mask has {masks.shape[0]} frames but depth has {n_frames}"
    )
    return masks.astype(bool)


def load_mask_folder(mask_dir, n_frames):
    """Load per-frame masks from an image folder. Returns (N_frames, H, W) bool."""
    exts = (".png", ".jpg", ".jpeg")
    files = sorted(
        f for f in os.listdir(mask_dir)
        if osp.splitext(f)[1].lower() in exts
    )
    assert len(files) == n_frames, (
        f"Mask folder has {len(files)} images but depth has {n_frames} frames"
    )
    masks = []
    for fname in files:
        img = np.array(Image.open(osp.join(mask_dir, fname)).convert("L"))
        masks.append(img > 127)
    return np.stack(masks, axis=0)   # (N_frames, H, W) bool


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(seq_dir, mask_type, depth_filename, mask_filename,
                     mask_subdir, output_subdir, output_filename, overwrite):
    depth_path = osp.join(seq_dir, "depth", depth_filename)
    out_dir    = osp.join(seq_dir, output_subdir)
    out_path   = osp.join(out_dir, output_filename)

    if not osp.isfile(depth_path):
        print(f"  SKIP — depth not found: {depth_path}")
        return

    if osp.isfile(out_path) and not overwrite:
        print(f"  SKIP — output already exists (use --overwrite): {out_path}")
        return

    depths = np.load(depth_path)   # (N_frames, H, W)
    N      = depths.shape[0]

    # Locate mask
    if mask_type == "npy":
        mask_path = osp.join(seq_dir, mask_filename)
        if not osp.isfile(mask_path):
            print(f"  SKIP — mask not found: {mask_path}")
            return
        plane_mask = load_mask_npy(mask_path, N)

    elif mask_type == "folder":
        mask_dir = osp.join(seq_dir, mask_subdir)
        if not osp.isdir(mask_dir):
            print(f"  SKIP — mask folder not found: {mask_dir}")
            return
        plane_mask = load_mask_folder(mask_dir, N)

    else:
        print(f"  SKIP — unknown mask_type '{mask_type}'")
        return

    # Resize mask if spatial dims don't match depth
    dH, dW = depths.shape[1], depths.shape[2]
    mH, mW = plane_mask.shape[1], plane_mask.shape[2]
    if (mH, mW) != (dH, dW):
        print(f"  Resizing mask from ({mH},{mW}) to ({dH},{dW})")
        from PIL import Image as _Image
        resized = []
        for i in range(N):
            m = _Image.fromarray(plane_mask[i].astype(np.uint8) * 255).resize(
                (dW, dH), _Image.NEAREST
            )
            resized.append(np.array(m) > 127)
        plane_mask = np.stack(resized, axis=0)

    # Zero out plane pixels in depth
    masked_depths = depths.copy()
    masked_depths[plane_mask] = 0.0

    n_zeroed = int(plane_mask.sum())
    n_total  = int(plane_mask.size)
    print(f"  Zeroed {n_zeroed}/{n_total} pixels ({100*n_zeroed/n_total:.1f}%) across {N} frames")

    os.makedirs(out_dir, exist_ok=True)
    np.save(out_path, masked_depths)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Dataset root (containing sequence sub-dirs) or a single sequence directory.",
    )
    parser.add_argument(
        "--mask-type", choices=["npy", "folder"], default="npy",
        help="Mask format: 'npy' (plane_mask.npy) or 'folder' (per-frame images). Default: npy.",
    )
    parser.add_argument(
        "--depth-filename", default="depths.npy",
        help="Depth filename inside <seq>/depth/. Default: depths.npy.",
    )
    parser.add_argument(
        "--mask-filename", default="plane_mask.npy",
        help="Mask filename inside <seq>/ when --mask-type=npy. Default: plane_mask.npy.",
    )
    parser.add_argument(
        "--mask-subdir", default="ae_mask",
        help="Sub-directory inside <seq>/ containing per-frame mask images when "
             "--mask-type=folder. Default: ae_mask.",
    )
    parser.add_argument(
        "--output-subdir", default="masked_depth",
        help="Output sub-directory name inside <seq>/. Default: masked_depth.",
    )
    parser.add_argument(
        "--output-filename", default="depths.npy",
        help="Output depth filename. Default: depths.npy.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    dataset_dir = osp.abspath(args.dataset_dir)

    # Single sequence or dataset root?
    if osp.isfile(osp.join(dataset_dir, "depth", args.depth_filename)):
        seq_dirs = [dataset_dir]
    else:
        seq_dirs = sorted(
            osp.join(dataset_dir, d)
            for d in os.listdir(dataset_dir)
            if osp.isdir(osp.join(dataset_dir, d))
        )

    print(f"Found {len(seq_dirs)} sequence(s) under {dataset_dir}\n")

    for seq_dir in seq_dirs:
        print(f"[{osp.basename(seq_dir)}]")
        process_sequence(
            seq_dir,
            mask_type=args.mask_type,
            depth_filename=args.depth_filename,
            mask_filename=args.mask_filename,
            mask_subdir=args.mask_subdir,
            output_subdir=args.output_subdir,
            output_filename=args.output_filename,
            overwrite=args.overwrite,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
