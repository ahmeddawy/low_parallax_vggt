#!/usr/bin/env python3
"""
Smooth COLMAP camera translations for fixed-camera and low-parallax sequences.

For fixed camera:
    All frame translations are replaced by the per-axis median — effectively
    pinning the camera at a single world position.

For low parallax:
    Translations are smoothed with a Savitzky-Golay filter to remove jitter
    while preserving the slow drift of the real camera motion.

Camera type classification:
    Automatic (default): reads tracks.npy from the sequence directory.
    If the max plane-track displacement across all frames is below
    --motion-threshold (default 5 px in original space), the sequence is
    classified as fixed; otherwise as low_parallax.
    Manual override: --camera-type fixed | low_parallax

Usage examples:
    # Auto-classify entire dataset
    python smooth_colmap_poses.py --dataset-dir /data/my_dataset

    # Force all sequences to low_parallax
    python smooth_colmap_poses.py --dataset-dir /data/my_dataset --camera-type low_parallax

    # Single sequence, fixed camera
    python smooth_colmap_poses.py --dataset-dir /data/my_dataset/seq_001 --camera-type fixed

    # Custom smoothing window for low_parallax
    python smooth_colmap_poses.py --dataset-dir /data/my_dataset --savgol-window 21
"""

import argparse
import os
import os.path as osp
import shutil

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# COLMAP images.txt I/O
# ---------------------------------------------------------------------------

def read_images_txt(path):
    """Parse COLMAP images.txt → dict keyed by image_id."""
    images = {}
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].strip().split()
        if len(parts) < 10:
            i += 1
            continue
        image_id  = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name      = parts[9]
        points2d  = lines[i + 1].strip() if i + 1 < len(lines) else ""
        images[image_id] = dict(
            qw=qw, qx=qx, qy=qy, qz=qz,
            tx=tx, ty=ty, tz=tz,
            camera_id=camera_id, name=name,
            points2d=points2d,
        )
        i += 2
    return images


def write_images_txt(images, path):
    """Write COLMAP images.txt from dict keyed by image_id."""
    with open(path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for image_id in sorted(images.keys()):
            img = images[image_id]
            f.write(
                f"{image_id} "
                f"{img['qw']:.10f} {img['qx']:.10f} {img['qy']:.10f} {img['qz']:.10f} "
                f"{img['tx']:.10f} {img['ty']:.10f} {img['tz']:.10f} "
                f"{img['camera_id']} {img['name']}\n"
            )
            f.write(f"{img['points2d']}\n")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_camera_type(seq_dir, tracks_path, motion_threshold):
    """
    Return 'fixed' or 'low_parallax'.

    Priority:
      1. ae_data/meta.json  — uses 'camera_motion' field if present (most reliable)
      2. tracks.npy         — computes max plane displacement as fallback
    """
    import json

    # 1. meta.json (already computed by AE export pipeline)
    meta_path = osp.join(seq_dir, "ae_data", "meta.json")
    if osp.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        motion = meta.get("camera_motion", "").strip().lower()
        if motion in ("fixed", "low_parallax"):
            print(f"    meta.json camera_motion = '{motion}'")
            return motion

    # 2. Fallback: inspect tracks.npy displacement
    if not osp.isfile(tracks_path):
        return None

    tracks = np.load(tracks_path)   # (N_frames, N_tracks, 2)
    if tracks.shape[0] < 2:
        return None

    displacements = np.linalg.norm(tracks - tracks[0:1], axis=-1)  # (N_frames, N_tracks)
    max_disp      = float(displacements.max())
    camera_type   = "fixed" if max_disp < motion_threshold else "low_parallax"
    print(f"    tracks max displacement = {max_disp:.2f} px  →  {camera_type}")
    return camera_type


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def _odd(n):
    """Return n if odd, else n+1."""
    return n if n % 2 == 1 else n + 1


def smooth_translations_savgol(translations, window_length=None, polyorder=2):
    """
    Apply Savitzky-Golay filter along each translation axis.

    Args:
        translations: (N, 3) float array
        window_length: filter window (default: ~10% of N, minimum 5, always odd)
        polyorder: polynomial order (default: 2)
    Returns:
        smoothed (N, 3) float array
    """
    N = translations.shape[0]

    if window_length is None:
        window_length = _odd(max(5, int(N * 0.10)))

    # Clamp to valid range
    window_length = min(_odd(window_length), N if N % 2 == 1 else N - 1)
    window_length = max(window_length, polyorder + 1)
    if window_length % 2 == 0:
        window_length += 1

    smoothed = np.empty_like(translations)
    for axis in range(3):
        smoothed[:, axis] = savgol_filter(translations[:, axis], window_length, polyorder)
    return smoothed


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(seq_dir, camera_type, motion_threshold, savgol_window, savgol_polyorder, backup):
    colmap_dir = osp.join(seq_dir, "colmap")
    images_txt = osp.join(colmap_dir, "images.txt")
    tracks_path = osp.join(seq_dir, "tracks.npy")

    if not osp.isfile(images_txt):
        print(f"  SKIP — no colmap/images.txt found")
        return

    # Auto-classify if not forced
    resolved_type = camera_type
    if resolved_type is None:
        resolved_type = classify_camera_type(seq_dir, tracks_path, motion_threshold)
        if resolved_type is None:
            print(f"  SKIP — cannot classify (no tracks.npy). Use --camera-type to force.")
            return

    print(f"  camera_type = {resolved_type}")

    # Backup original
    if backup:
        backup_path = images_txt + ".orig"
        if not osp.isfile(backup_path):
            shutil.copy2(images_txt, backup_path)
            print(f"  Backup saved → {backup_path}")
        else:
            print(f"  Backup already exists, skipping backup")

    images    = read_images_txt(images_txt)
    image_ids = sorted(images.keys())

    translations = np.array(
        [[images[i]["tx"], images[i]["ty"], images[i]["tz"]] for i in image_ids],
        dtype=np.float64,
    )   # (N, 3)

    if resolved_type == "fixed":
        median_t       = np.median(translations, axis=0)
        new_translations = np.broadcast_to(median_t, translations.shape).copy()
        print(f"  Fixed: median translation = [{median_t[0]:.4f}, {median_t[1]:.4f}, {median_t[2]:.4f}]")

    elif resolved_type == "low_parallax":
        new_translations = smooth_translations_savgol(
            translations,
            window_length=savgol_window,
            polyorder=savgol_polyorder,
        )
        mean_change = float(np.abs(new_translations - translations).mean())
        print(f"  Low parallax: mean |Δt| after smoothing = {mean_change:.4f}")

    else:
        print(f"  SKIP — unknown camera_type '{resolved_type}'")
        return

    for idx, image_id in enumerate(image_ids):
        images[image_id]["tx"] = float(new_translations[idx, 0])
        images[image_id]["ty"] = float(new_translations[idx, 1])
        images[image_id]["tz"] = float(new_translations[idx, 2])

    write_images_txt(images, images_txt)
    print(f"  Saved → {images_txt}")


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
        "--camera-type", choices=["fixed", "low_parallax"], default=None,
        help="Force camera type for all sequences. Default: auto-classify from tracks.npy.",
    )
    parser.add_argument(
        "--motion-threshold", type=float, default=5.0,
        help="Max AE plane-track displacement (px, original resolution) to classify as "
             "fixed camera. Default: 5.0.",
    )
    parser.add_argument(
        "--savgol-window", type=int, default=None,
        help="Savitzky-Golay window length for low_parallax smoothing. "
             "Default: ~10%% of sequence length (minimum 5, always odd).",
    )
    parser.add_argument(
        "--savgol-polyorder", type=int, default=2,
        help="Savitzky-Golay polynomial order. Default: 2.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Do not back up the original images.txt before overwriting.",
    )
    args = parser.parse_args()

    dataset_dir = osp.abspath(args.dataset_dir)

    # Single sequence or dataset root?
    if osp.isfile(osp.join(dataset_dir, "colmap", "images.txt")):
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
            camera_type=args.camera_type,
            motion_threshold=args.motion_threshold,
            savgol_window=args.savgol_window,
            savgol_polyorder=args.savgol_polyorder,
            backup=not args.no_backup,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
