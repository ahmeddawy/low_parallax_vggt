#!/usr/bin/env python3
"""
Prepare Rembrand dataset sequences for VGGT training.

Your dataset is already partially organized (colmap/ files are in place).
This script handles all remaining preparation steps:

  1. Convert depth/<seq_name>.zip (EXR frames) → depth/depths.npy
  2. Symlink ae_data/tracks.npy      → tracks.npy      (sequence root)
     Symlink ae_data/track_masks.npy → track_masks.npy (sequence root)
  3. Generate masked_depth/depths.npy by zeroing the AE plane region
     (polygon from ae_data/corners.csv) so the depth loss skips plane pixels.
  4. Apply dynamic-object segmentation masks (mask/<seq>.zip + mask/<seq>.txt)
     on top of masked_depth/depths.npy, zeroing any non-background pixels.
     This stacks on step 3: plane + dynamic objects are both zeroed.

Expected input layout per sequence:
    <seq>/
      colmap/
        cameras.txt
        images.txt
        points3D.txt
      depth/
        <seq_name>.zip     ← EXR depth frames
      ae_data/
        corners.csv        ← per-frame plane corners (tl/tr/bl/br in original px)
        tracks.npy         ← (N_frames, N_tracks, 2)
        track_masks.npy    ← (N_frames, N_tracks) bool
        meta.json          ← contains video_width, video_height
      mask/                ← optional, for dynamic object masking
        <seq_name>.zip     ← per-frame PNG class-ID masks (0.png, 1.png, ...)
        <seq_name>.txt     ← class map: "ID: class_name" one per line

Output (added by this script):
    <seq>/
      depth/depths.npy          ← (N_frames, H, W) float16
      masked_depth/depths.npy   ← plane + dynamic objects zeroed
      tracks.npy                ← symlink → ae_data/tracks.npy
      track_masks.npy           ← symlink → ae_data/track_masks.npy

Usage examples:
    # Entire dataset
    python prepare_rembrand_dataset.py --dataset-dir /data/my_dataset

    # Single sequence
    python prepare_rembrand_dataset.py --dataset-dir /data/my_dataset/seq_001

    # Skip masked depth generation
    python prepare_rembrand_dataset.py --dataset-dir /data/my_dataset --skip-masked-depth

    # Overwrite everything
    python prepare_rembrand_dataset.py --dataset-dir /data/my_dataset \\
        --overwrite-depth --overwrite-masked-depth --overwrite-tracks

    # Copy tracks instead of symlinking
    python prepare_rembrand_dataset.py --dataset-dir /data/my_dataset --copy-tracks
"""

import argparse
import csv
import json
import os
import os.path as osp
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import Imath
import numpy as np
import OpenEXR
from PIL import Image, ImageDraw


FRAME_RE = re.compile(r"(\d+)")


@dataclass
class SequenceResult:
    seq_name: str
    ok: bool
    reason: str


# ---------------------------------------------------------------------------
# EXR → numpy (reused verbatim from prepare_colmap_folder_for_vggt.py)
# ---------------------------------------------------------------------------

def read_exr_depth_from_zip(zf: zipfile.ZipFile, member_name: str) -> np.ndarray:
    """Read a single EXR depth frame from inside a zip, via a temp file."""
    with zf.open(member_name) as f:
        exr_bytes = f.read()

    with tempfile.NamedTemporaryFile(suffix=".exr") as tmp:
        tmp.write(exr_bytes)
        tmp.flush()

        exr    = OpenEXR.InputFile(tmp.name)
        header = exr.header()
        dw     = header["dataWindow"]
        width  = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        z_raw = exr.channel("Z", Imath.PixelType(Imath.PixelType.HALF))
        exr.close()

    depth = np.frombuffer(z_raw, dtype=np.float16).reshape((height, width))
    return depth.astype(np.float32, copy=False)


def convert_depth_zip_to_npy(zip_path: str, out_npy_path: str, out_dtype: str) -> Tuple[bool, str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".exr")]
        if not members:
            return False, f"no .exr files in {zip_path}"

        parsed: List[Tuple[int, str]] = []
        for m in members:
            stem  = osp.splitext(osp.basename(m))[0]
            match = FRAME_RE.fullmatch(stem)
            if match:
                parsed.append((int(match.group(1)), m))
        if not parsed:
            return False, f"no numeric frame names in {zip_path}"

        parsed.sort(key=lambda x: x[0])
        max_idx = parsed[-1][0]

        # Probe shape from first readable frame
        first_depth = None
        for _, member_name in parsed:
            try:
                first_depth = read_exr_depth_from_zip(zf, member_name)
                break
            except Exception:
                continue
        if first_depth is None:
            return False, f"failed to decode any EXR in {zip_path}"

        h, w  = first_depth.shape
        dtype = np.float16 if out_dtype == "float16" else np.float32

        depth_mm = np.lib.format.open_memmap(
            out_npy_path, mode="w+", dtype=dtype, shape=(max_idx + 1, h, w)
        )
        depth_mm[:] = np.nan

        for frame_idx, member_name in parsed:
            try:
                d = read_exr_depth_from_zip(zf, member_name)
            except Exception:
                continue
            if d.shape == (h, w):
                depth_mm[frame_idx] = d.astype(dtype, copy=False)

        del depth_mm

    return True, f"({max_idx + 1}, {h}, {w}) {out_dtype}"


def resolve_depth_zip(seq_name: str, seq_dir: str) -> Optional[str]:
    depth_dir = osp.join(seq_dir, "depth")
    if not osp.isdir(depth_dir):
        return None
    candidates = [osp.join(depth_dir, f"{seq_name}.zip")]
    candidates += sorted(
        osp.join(depth_dir, f) for f in os.listdir(depth_dir) if f.lower().endswith(".zip")
    )
    for p in candidates:
        if osp.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Intrinsics normalization (cameras.txt)
# ---------------------------------------------------------------------------

def normalize_cameras_txt(cameras_txt_path: str, backup: bool = True) -> Tuple[bool, str]:
    """
    Enforce isotropic pinhole intrinsics: set fx = fy = (fx + fy) / 2.

    Only modifies PINHOLE / SIMPLE_PINHOLE / OPENCV cameras.
    Backs up the original to cameras.txt.orig (unless backup=False).
    Returns (changed, message).
    """
    with open(cameras_txt_path) as f:
        lines = f.readlines()

    new_lines = []
    changed   = False
    report    = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            new_lines.append(line)
            continue

        parts = stripped.split()
        model = parts[1].upper()

        if model in ("PINHOLE", "OPENCV"):
            # PARAMS: fx fy cx cy [k1 k2 p1 p2 ...]
            fx, fy = float(parts[4]), float(parts[5])
            f_avg  = (fx + fy) / 2.0
            if abs(fx - fy) > 1e-4:
                parts[4] = f"{f_avg:.6f}"
                parts[5] = f"{f_avg:.6f}"
                report.append(f"camera {parts[0]}: fx={fx:.4f} fy={fy:.4f} → f={f_avg:.4f}")
                changed = True
            new_lines.append(" ".join(parts) + "\n")

        elif model == "SIMPLE_PINHOLE":
            # PARAMS: f cx cy  (single focal length — already isotropic)
            new_lines.append(line)

        else:
            new_lines.append(line)

    if not changed:
        return False, "already isotropic (fx≈fy), no change"

    if backup:
        backup_path = cameras_txt_path + ".orig"
        if not osp.isfile(backup_path):
            shutil.copy2(cameras_txt_path, backup_path)

    with open(cameras_txt_path, "w") as f:
        f.writelines(new_lines)

    return True, "; ".join(report)


# ---------------------------------------------------------------------------
# Plane mask from corners.csv
# ---------------------------------------------------------------------------

def load_corners_csv(corners_path: str) -> dict:
    """
    Parse corners.csv → dict mapping frame_idx (int) → list of 4 (x,y) tuples.
    Column order: frame, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y
    Polygon order for fill: tl → tr → br → bl  (clockwise)
    """
    corners = {}
    with open(corners_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(row["frame"])
            pts   = [
                (float(row["tl_x"]), float(row["tl_y"])),
                (float(row["tr_x"]), float(row["tr_y"])),
                (float(row["br_x"]), float(row["br_y"])),
                (float(row["bl_x"]), float(row["bl_y"])),
            ]
            corners[frame] = pts
    return corners


def corners_to_mask(corners_pts, orig_w: int, orig_h: int, target_w: int, target_h: int) -> np.ndarray:
    """
    Fill the quadrilateral defined by corners_pts at (orig_w, orig_h),
    then resize to (target_h, target_w). Returns bool array (target_h, target_w).
    """
    img = Image.new("L", (orig_w, orig_h), 0)
    ImageDraw.Draw(img).polygon(corners_pts, fill=255)

    if (orig_w, orig_h) != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.NEAREST)

    return np.array(img) > 127


def generate_masked_depth(
    depth_npy_path: str,
    corners_path: str,
    out_path: str,
    orig_w: int,
    orig_h: int,
) -> Tuple[bool, str]:
    """
    Load depths.npy, zero out the plane polygon per frame, save masked_depth/depths.npy.
    """
    depths  = np.load(depth_npy_path)          # (N_frames, H, W)
    corners = load_corners_csv(corners_path)   # frame_idx → 4 pts

    N, dH, dW = depths.shape
    masked    = depths.copy()
    n_zeroed  = 0

    for frame_idx in range(N):
        # Use nearest available corner frame (handles off-by-one at boundaries)
        if frame_idx in corners:
            pts = corners[frame_idx]
        elif corners:
            closest = min(corners.keys(), key=lambda k: abs(k - frame_idx))
            pts     = corners[closest]
        else:
            continue

        mask = corners_to_mask(pts, orig_w, orig_h, dW, dH)
        masked[frame_idx][mask] = 0.0
        n_zeroed += int(mask.sum())

    os.makedirs(osp.dirname(out_path), exist_ok=True)
    np.save(out_path, masked)

    total    = N * dH * dW
    pct      = 100.0 * n_zeroed / max(total, 1)
    return True, f"{n_zeroed}/{total} pixels zeroed ({pct:.1f}%) across {N} frames"


# ---------------------------------------------------------------------------
# Dynamic object masking (from mask/<seq>.zip + mask/<seq>.txt)
# ---------------------------------------------------------------------------

MASK_NAME_RE = re.compile(r"^(\d+)\.png$", re.IGNORECASE)


def read_class_map(txt_path: str) -> Dict[int, str]:
    """
    Parse class map file → dict[int, str].

    Supports two formats:
      - 'ID: class_name'  (e.g. original.txt from segmentation pipeline)
      - 'ID class_name'   (space-separated)
    Multiple IDs can map to the same class name (e.g. different person instances).
    """
    id_to_name: Dict[int, str] = {}
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = re.match(r"^(\d+)\s*[:\s]\s*(.+)$", line)
            if m:
                id_to_name[int(m.group(1))] = m.group(2).strip().lower()
    return id_to_name


def apply_segmentation_masks(
    depth_path: str,
    mask_dir: str,
    class_map_path: str,
    out_path: str,
    keep_classes: Set[str],
) -> Tuple[bool, str]:
    """
    Load depth, zero pixels whose segmentation class is NOT in keep_classes.

    Mask format: per-frame PNG files named 00000.png, 00001.png, ... in mask_dir.
    Class map:   mask_dir/original.txt with 'ID: class_name' lines.

    Reads from depth_path, writes to out_path (can be the same file for in-place).
    """
    depths     = np.load(depth_path)              # (N, H, W)
    id_to_name = read_class_map(class_map_path)
    keep_ids   = {cid for cid, name in id_to_name.items() if name in keep_classes}

    if not keep_ids:
        return False, f"no keep IDs found for classes {keep_classes} in {class_map_path}"

    # Index per-frame PNG files: frame_idx → full path
    frame_to_path: Dict[int, str] = {}
    for fname in os.listdir(mask_dir):
        m = MASK_NAME_RE.match(fname)
        if m:
            frame_to_path[int(m.group(1))] = osp.join(mask_dir, fname)

    N, dH, dW = depths.shape
    if len(frame_to_path) != N:
        return False, f"frame count mismatch: depth={N}, mask folder={len(frame_to_path)}"

    out_arr  = np.lib.format.open_memmap(out_path, mode="w+", dtype=depths.dtype, shape=depths.shape)
    n_zeroed = 0

    for frame_idx in range(N):
        if frame_idx not in frame_to_path:
            out_arr[frame_idx] = depths[frame_idx]
            continue

        mask_img = np.array(Image.open(frame_to_path[frame_idx]))
        if mask_img.shape != (dH, dW):
            # Resize mask to match depth resolution
            mask_img = np.array(
                Image.fromarray(mask_img).resize((dW, dH), Image.NEAREST)
            )

        keep_mask             = np.isin(mask_img, list(keep_ids))
        frame_depth           = depths[frame_idx].copy()
        frame_depth[~keep_mask] = 0.0
        out_arr[frame_idx]    = frame_depth
        n_zeroed             += int((~keep_mask).sum())

    del out_arr
    pct = 100.0 * n_zeroed / max(N * dH * dW, 1)
    return True, f"{n_zeroed} additional pixels zeroed ({pct:.1f}%)"


# ---------------------------------------------------------------------------
# Tracks linking / copying
# ---------------------------------------------------------------------------

def link_or_copy(src: str, dst: str, use_symlink: bool, overwrite: bool) -> Tuple[bool, str]:
    if osp.lexists(dst):
        if not overwrite:
            return True, "already exists"
        os.remove(dst)

    if use_symlink:
        os.symlink(osp.abspath(src), dst)
        return True, f"symlinked → {src}"
    else:
        shutil.copy2(src, dst)
        return True, f"copied from {src}"


# ---------------------------------------------------------------------------
# Per-sequence processing
# ---------------------------------------------------------------------------

def process_sequence(
    seq_dir: str,
    out_dtype: str,
    overwrite_depth: bool,
    overwrite_masked_depth: bool,
    overwrite_tracks: bool,
    use_symlink: bool,
    skip_masked_depth: bool,
    keep_classes: Set[str],
) -> SequenceResult:
    seq_name = osp.basename(seq_dir.rstrip("/"))
    issues   = []

    # ---- 0. Normalize intrinsics (enforce fx = fy = mean) -------------------
    cameras_txt = osp.join(seq_dir, "colmap", "cameras.txt")
    if osp.isfile(cameras_txt):
        changed, msg = normalize_cameras_txt(cameras_txt, backup=True)
        label = "normalized" if changed else "isotropic"
        print(f"  intrinsics    : {label} — {msg}")
    else:
        print(f"  intrinsics    : SKIP — colmap/cameras.txt not found")

    # ---- 1. Depth conversion ------------------------------------------------
    depth_npy = osp.join(seq_dir, "depth", "depths.npy")

    if osp.isfile(depth_npy) and not overwrite_depth:
        print(f"  depth         : already exists, skipping")
    else:
        zip_path = resolve_depth_zip(seq_name, seq_dir)
        if zip_path is None:
            issues.append("no depth zip found")
            print(f"  depth         : FAIL — no zip found")
        else:
            ok, msg = convert_depth_zip_to_npy(zip_path, depth_npy, out_dtype)
            if ok:
                print(f"  depth         : OK  shape={msg}")
            else:
                issues.append(f"depth conversion: {msg}")
                print(f"  depth         : FAIL — {msg}")

    # ---- 2a. Images symlink (ColmapDataset expects <seq>/images/, not <seq>/colmap/images/) ---
    colmap_images_src = osp.join(seq_dir, "colmap", "images")
    images_dst        = osp.join(seq_dir, "images")
    if osp.isdir(colmap_images_src):
        ok, msg = link_or_copy(colmap_images_src, images_dst, use_symlink=True, overwrite=overwrite_tracks)
        status  = "OK" if ok else "FAIL"
        print(f"  {'images':<20}: {status} — {msg}")
        if not ok:
            issues.append(f"images: {msg}")
    else:
        print(f"  {'images':<20}: SKIP — colmap/images/ not found")

    # ---- 2b. Tracks link/copy -----------------------------------------------
    for fname in ["tracks.npy", "track_masks.npy"]:
        src = osp.join(seq_dir, "ae_data", fname)
        dst = osp.join(seq_dir, fname)

        if not osp.isfile(src):
            print(f"  {fname:<20}: SKIP — not found in ae_data/")
            continue

        ok, msg = link_or_copy(src, dst, use_symlink=use_symlink, overwrite=overwrite_tracks)
        status  = "OK" if ok else "FAIL"
        print(f"  {fname:<20}: {status} — {msg}")
        if not ok:
            issues.append(f"{fname}: {msg}")

    # ---- 3. Masked depth from AE plane corners ------------------------------
    masked_depth_path  = osp.join(seq_dir, "masked_depth", "depths.npy")
    masked_depth_fresh = False  # True only if step 3 actually (re)created the file this run

    if not skip_masked_depth:
        corners_path = osp.join(seq_dir, "ae_data", "corners.csv")
        meta_path    = osp.join(seq_dir, "ae_data", "meta.json")

        if osp.isfile(masked_depth_path) and not overwrite_masked_depth:
            print(f"  masked_depth  : already exists, skipping")
        elif not osp.isfile(depth_npy):
            print(f"  masked_depth  : SKIP — depth/depths.npy not available")
        elif not osp.isfile(corners_path):
            print(f"  masked_depth  : SKIP — ae_data/corners.csv not found")
        else:
            # Get original video resolution from meta.json (fallback: 1920x1080)
            orig_w, orig_h = 1920, 1080
            if osp.isfile(meta_path):
                with open(meta_path) as f:
                    meta   = json.load(f)
                orig_w = meta.get("video_width",  orig_w)
                orig_h = meta.get("video_height", orig_h)

            ok, msg = generate_masked_depth(depth_npy, corners_path, masked_depth_path, orig_w, orig_h)
            status  = "OK" if ok else "FAIL"
            print(f"  masked_depth  : {status} — {msg}")
            if ok:
                masked_depth_fresh = True
            else:
                issues.append(f"masked_depth: {msg}")

    # ---- 4. Dynamic object masking ------------------------------------------
    # Format: mask/00000.png, 00001.png, ... + mask/original.txt (class map)
    # Only runs when masked_depth was (re)created this run OR --overwrite-masked-depth is set.
    mask_dir       = osp.join(seq_dir, "mask")
    class_map_path = osp.join(mask_dir, "original.txt")

    if osp.isdir(mask_dir) and osp.isfile(class_map_path):
        if skip_masked_depth:
            print(f"  seg masks      : SKIP — masked depth skipped")
        elif not osp.isfile(masked_depth_path):
            print(f"  seg masks      : SKIP — masked_depth/depths.npy not ready yet")
        elif not masked_depth_fresh and not overwrite_masked_depth:
            print(f"  seg masks      : already applied, skipping")
        else:
            # Apply in-place: read masked_depth, zero dynamic objects, write back
            ok, msg = apply_segmentation_masks(
                depth_path=masked_depth_path,
                mask_dir=mask_dir,
                class_map_path=class_map_path,
                out_path=masked_depth_path,
                keep_classes=keep_classes,
            )
            status = "OK" if ok else "FAIL"
            print(f"  seg masks      : {status} — {msg}")
            if not ok:
                issues.append(f"seg masks: {msg}")
    else:
        print(f"  seg masks      : SKIP — mask/original.txt not found")

    if issues:
        return SequenceResult(seq_name, False, "; ".join(issues))
    return SequenceResult(seq_name, True, "prepared")


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
        "--depth-dtype", default="float16", choices=["float16", "float32"],
        help="dtype for depth/depths.npy. Default: float16.",
    )
    parser.add_argument(
        "--overwrite-depth", action="store_true",
        help="Recreate depth/depths.npy even if it already exists.",
    )
    parser.add_argument(
        "--overwrite-masked-depth", action="store_true",
        help="Recreate masked_depth/depths.npy even if it already exists.",
    )
    parser.add_argument(
        "--overwrite-tracks", action="store_true",
        help="Overwrite existing tracks.npy / track_masks.npy symlinks.",
    )
    parser.add_argument(
        "--copy-tracks", action="store_true",
        help="Copy track files instead of symlinking.",
    )
    parser.add_argument(
        "--skip-masked-depth", action="store_true",
        help="Skip masked_depth generation (steps 3 & 4).",
    )
    parser.add_argument(
        "--keep-classes", nargs="+", default=["background"],
        help="Segmentation class names to keep as valid depth (step 4). Default: background.",
    )
    args = parser.parse_args()

    dataset_dir = osp.abspath(args.dataset_dir)

    # Single sequence or dataset root?
    if osp.isdir(osp.join(dataset_dir, "colmap")):
        seq_dirs = [dataset_dir]
    else:
        seq_dirs = sorted(
            osp.join(dataset_dir, d)
            for d in os.listdir(dataset_dir)
            if osp.isdir(osp.join(dataset_dir, d))
        )

    print(f"Found {len(seq_dirs)} sequence(s) under {dataset_dir}\n")

    results: List[SequenceResult] = []
    for seq_dir in seq_dirs:
        print(f"[{osp.basename(seq_dir)}]")
        res = process_sequence(
            seq_dir,
            out_dtype=args.depth_dtype,
            overwrite_depth=args.overwrite_depth,
            overwrite_masked_depth=args.overwrite_masked_depth,
            overwrite_tracks=args.overwrite_tracks,
            use_symlink=not args.copy_tracks,
            skip_masked_depth=args.skip_masked_depth,
            keep_classes={c.strip().lower() for c in args.keep_classes},
        )
        results.append(res)
        print()

    ok_count   = sum(1 for r in results if r.ok)
    fail_count = len(results) - ok_count
    print(f"Done. total={len(results)}  ok={ok_count}  fail={fail_count}")

    if fail_count > 0:
        print("\nFailed sequences:")
        for r in results:
            if not r.ok:
                print(f"  - {r.seq_name}: {r.reason}")


if __name__ == "__main__":
    main()
