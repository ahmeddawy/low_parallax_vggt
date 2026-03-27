#!/usr/bin/env python3
"""
eval_track_head.py - Evaluate VGGT track head against GT AE tracks.

Tests the model's raw feedforward track head (no BA, no post-processing).
Compares vanilla VGGT vs fine-tuned checkpoint on both train and val splits.

Protocol:
  - Images loaded at 16:9 aspect ratio (crop mode) to match fine-tuning.
  - GT frame-0 track positions used as query points.
  - Predicted positions for frames 1..S compared against GT (visible only).
  - Aggregator runs once per sequence; tracker runs per chunk (OOM safe).
  - Train split is randomly subsampled (--train-max-seqs, default 20).

Metrics (reported in original image resolution):
  - ATE:       mean L2 pixel error over all visible GT track points
  - median_te: median L2 pixel error
  - delta_Npx: fraction of predictions within N pixels of GT (1,2,5,10)
  - vis_acc:   visibility prediction accuracy vs GT track_masks

Homography refinement (--ransac-thresh, --min-h-inliers):
  - For each frame t, a homography H is estimated (RANSAC) from
    query_pts -> pred_tracks[t], then query_pts are projected through H.
  - Enforces global planar consistency; metrics reported as *_h variants.
  - inlier_ratio: fraction of tracks consistent with the fitted plane.

Visualization (--vis-max-seqs, default 3 per split):
  - One video per sequence saved to --vis-dir/{split}/{seq}.mp4
  - Left panel: vanilla vs GT | Right panel: finetuned vs GT
  - Dots: GT=green filled, prediction=colored outline

Usage:
    python eval_track_head.py \\
        --vanilla-ckpt     /workspace/model.pt \\
        --finetuned-ckpt   /workspace/ckpts/checkpoint.pt \\
        --dataset-dir      /mnt/bucket/.../tracking_whisper_sample_dataset \\
        --train-split-file /mnt/bucket/.../train_split.txt \\
        --val-split-file   /mnt/bucket/.../val_split.txt \\
        [--train-max-seqs 20] \\
        [--vis-max-seqs 3] [--vis-n-tracks 50] [--vis-fps 8] [--vis-dir track_eval_viz] \\
        [--lora] [--lora-r 16] [--lora-alpha 32] \\
        [--track-chunk 256] \\
        [--ransac-thresh 3.0] [--min-h-inliers 8] \\
        [--output-json track_eval_results.json]
"""

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VGGT track head vs GT AE tracks"
    )
    p.add_argument(
        "--vanilla-ckpt", required=True,
        help="Path to vanilla VGGT checkpoint",
    )
    p.add_argument(
        "--finetuned-ckpt", required=True,
        help="Path to fine-tuned checkpoint",
    )
    p.add_argument(
        "--dataset-dir", required=True,
        help="Root dataset directory",
    )
    p.add_argument(
        "--train-split-file", default=None,
        help=(
            "Train split: one sequence name per line. "
            "If neither split file is given, all valid subdirectories of "
            "--dataset-dir are used as a single 'eval' split."
        ),
    )
    p.add_argument(
        "--val-split-file", default=None,
        help="Val split: one sequence name per line.",
    )
    p.add_argument(
        "--train-max-seqs", type=int, default=20,
        help=(
            "Randomly sample this many sequences from the train split "
            "(default: 20, -1 = use all)"
        ),
    )
    p.add_argument(
        "--lora", action="store_true", default=False,
        help="Fine-tuned ckpt uses LoRA (wrap aggregator with PEFT)",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-target-modules", type=str, default="qkv,proj")
    p.add_argument(
        "--track-chunk", type=int, default=256,
        help="Tracks per tracker forward pass (reduce if OOM)",
    )
    p.add_argument(
        "--vis-thresh", type=float, default=0.5,
        help="Threshold on predicted vis score to call a point visible",
    )
    # Homography refinement
    p.add_argument(
        "--ransac-thresh", type=float, default=3.0,
        help="RANSAC reprojection threshold in pixels for homography estimation (default: 3.0)",
    )
    p.add_argument(
        "--min-h-inliers", type=int, default=8,
        help="Minimum RANSAC inliers required to accept a homography (default: 8)",
    )
    # Visualization
    p.add_argument(
        "--vis-max-seqs", type=int, default=3,
        help="Visualize first N sequences per split (0 = disabled)",
    )
    p.add_argument(
        "--vis-n-tracks", type=int, default=50,
        help="Number of tracks to draw per frame",
    )
    p.add_argument(
        "--vis-fps", type=int, default=8,
        help="Frame rate of output videos (default: 8)",
    )
    p.add_argument(
        "--vis-dir", default="track_eval_viz",
        help="Output directory for visualization images",
    )
    p.add_argument("--output-json", default="track_eval_results.json")
    p.add_argument(
        "--ae-output-dir", default=None,
        help=(
            "If set, save per-tracker corner metric JSONs compatible with "
            "aggregate_metrics.py. Layout: "
            "<ae-output-dir>/<tag>/metrics/<seq>.json"
        ),
    )
    p.add_argument(
        "--homography-dir", default=None,
        help=(
            "If set, save estimated homographies to this directory as .npy files. "
            "Layout: <homography-dir>/<split>/<seq>/<tag>_H.npy  (S, 3, 3) float32 "
            "and <tag>_inliers.npy  (S,) float32."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path, lora=False, lora_r=16, lora_alpha=32.0,
               lora_targets="qkv,proj", device="cuda"):
    """Load VGGT with optional LoRA wrapping."""
    model = VGGT(enable_point=False)  # point head not needed

    if lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_targets.split(","),
            lora_dropout=0.0,
            bias="none",
        )
        model.aggregator = get_peft_model(model.aggregator, lora_config)
        print(
            f"  LoRA applied "
            f"(r={lora_r}, alpha={lora_alpha}, targets={lora_targets})"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"  [WARN] {len(missing)} missing keys "
            f"(expected for LoRA or partial ckpt)"
        )

    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def get_model_resolution(image_path_or_size):
    """
    Return (W_orig, H_orig, W_model, H_model).
    crop mode: width=518, height=round(H*518/W/14)*14.
    Accepts either a file path (str) or a (W, H) tuple.
    """
    if isinstance(image_path_or_size, (tuple, list)):
        W_orig, H_orig = int(image_path_or_size[0]), int(image_path_or_size[1])
    else:
        img = Image.open(image_path_or_size)
        W_orig, H_orig = img.size
    W_model = 518
    H_model = round(H_orig * (W_model / W_orig) / 14) * 14
    return W_orig, H_orig, W_model, H_model


def get_sequence_fps(seq_dir, fallback=25.0):
    """Read FPS from original.mp4 in seq_dir; return fallback if not found."""
    for name in ("original.mp4", "video.mp4"):
        vpath = os.path.join(seq_dir, name)
        if os.path.isfile(vpath):
            cap = cv2.VideoCapture(vpath)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                return float(fps)
    return float(fallback)


def load_seq_data(seq_dir):
    """
    Load all per-sequence data, supporting two dataset layouts:

    Format A  (original):
        seq_dir/images/*.jpg|png   — pre-extracted frames
        seq_dir/tracks.npy
        seq_dir/track_masks.npy    (optional)

    Format B  (ae_data):
        seq_dir/original.mp4       — source video
        seq_dir/ae_data/meta.json  — metadata (video_width, video_height, n_frames, …)
        seq_dir/ae_data/tracks.npy
        seq_dir/ae_data/track_masks.npy  (optional)

    Returns:
        image_paths   : list[str]         — JPEG paths (temp dir if from video)
        W_orig        : int
        H_orig        : int
        gt_tracks_orig: (S, N, 2) float32
        gt_vis_mask   : (S, N)    bool
        tmp_dir       : str or None       — caller must shutil.rmtree if not None
        error         : str or None
    """
    import shutil, tempfile, json as _json

    ae_dir = os.path.join(seq_dir, "ae_data")

    # ---- detect format ----
    img_dir = os.path.join(seq_dir, "images")
    video_path = os.path.join(seq_dir, "original.mp4")
    use_video = (
        not os.path.isdir(img_dir) or not os.listdir(img_dir)
    ) and os.path.isfile(video_path)

    # ---- tracks + masks ----
    if os.path.isfile(os.path.join(ae_dir, "tracks.npy")):
        tracks_path = os.path.join(ae_dir, "tracks.npy")
        masks_path  = os.path.join(ae_dir, "track_masks.npy")
    else:
        tracks_path = os.path.join(seq_dir, "tracks.npy")
        masks_path  = os.path.join(seq_dir, "track_masks.npy")

    if not os.path.isfile(tracks_path):
        return None, None, None, None, None, None, "no tracks.npy"

    gt_tracks_orig = np.load(tracks_path).astype(np.float32)
    gt_vis_mask = (
        np.load(masks_path).astype(bool)
        if os.path.isfile(masks_path)
        else np.ones(gt_tracks_orig.shape[:2], dtype=bool)
    )

    # ---- image paths ----
    tmp_dir = None

    if use_video:
        # Read resolution from meta.json if available
        meta_path = os.path.join(ae_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as fh:
                meta = _json.load(fh)
            W_orig      = int(meta["video_width"])
            H_orig      = int(meta["video_height"])
            n_frames    = int(meta.get("n_frames", 0)) or None
            start_frame = int(meta.get("video_frame_start", 0))
        else:
            cap = cv2.VideoCapture(video_path)
            W_orig      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H_orig      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            n_frames    = None
            start_frame = 0

        # Extract frames
        tmp_dir = tempfile.mkdtemp(prefix="vggt_eval_")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        image_paths = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            p = os.path.join(tmp_dir, f"{count:06d}.jpg")
            cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            image_paths.append(p)
            count += 1
            if n_frames is not None and count >= n_frames:
                break
        cap.release()

        if not image_paths:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None, None, None, None, None, None, f"could not read frames from {video_path}"

    else:
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        if not image_paths:
            return None, None, None, None, None, None, f"no images in {img_dir}"
        W_orig, H_orig, _, _ = get_model_resolution(image_paths[0])

    S = len(image_paths)
    if gt_tracks_orig.shape[0] < S:
        # Tracks may cover fewer frames than the video — trim to tracks length
        S = gt_tracks_orig.shape[0]
        image_paths = image_paths[:S]

    gt_tracks_orig = gt_tracks_orig[:S]
    gt_vis_mask    = gt_vis_mask[:S]

    if S < 2:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None, None, None, None, None, "need at least 2 frames"

    return image_paths, W_orig, H_orig, gt_tracks_orig, gt_vis_mask, tmp_dir, None


# ---------------------------------------------------------------------------
# Track head inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_track_head(
    model, images_tensor, query_points_np, chunk_size, device, dtype
):
    """
    Run the VGGT track head for all query tracks, chunked to avoid OOM.

    Args:
        model:           VGGT model on device
        images_tensor:   (S, 3, H, W) float32 [0,1] on CPU
        query_points_np: (N, 2) float32 in MODEL pixel coords
        chunk_size:      max tracks per forward pass
        device, dtype:   inference device/dtype

    Returns:
        pred_tracks: (S, N, 2) float32 in model pixel coords
        pred_vis:    (S, N)    float32 [0,1]
    """
    N = query_points_np.shape[0]
    images_batch = images_tensor.unsqueeze(0).to(device)  # (1,S,3,H,W)

    # Aggregator + feature extractor run once, shared across all chunks
    with torch.cuda.amp.autocast(dtype=dtype):
        agg_tokens, patch_start_idx = model.aggregator(images_batch)
        feature_maps = model.track_head.feature_extractor(
            agg_tokens, images_batch, patch_start_idx
        )

    all_tracks = []
    all_vis = []
    query_tensor = torch.from_numpy(query_points_np)  # (N, 2)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        qpts = query_tensor[start:end].unsqueeze(0).to(device)  # (1,chunk,2)

        with torch.cuda.amp.autocast(dtype=dtype):
            coord_preds, vis, _ = model.track_head.tracker(
                query_points=qpts,
                fmaps=feature_maps,
                iters=model.track_head.iters,
            )

        # coord_preds: list[(1,S,chunk,2)] per iter — take last
        pred_t = coord_preds[-1].squeeze(0).cpu().float()  # (S,chunk,2)
        pred_v = vis.squeeze(0).cpu().float()               # (S,chunk)

        all_tracks.append(pred_t)
        all_vis.append(pred_v)

    pred_tracks = torch.cat(all_tracks, dim=1).numpy()  # (S, N, 2)
    pred_vis = torch.cat(all_vis, dim=1).numpy()         # (S, N)

    return pred_tracks, pred_vis


# ---------------------------------------------------------------------------
# Homography refinement
# ---------------------------------------------------------------------------

def estimate_homography_refined_tracks(
    query_pts, pred_tracks, min_inliers=8, ransac_thresh=3.0
):
    """
    For each frame t, estimate a homography H (RANSAC) mapping query_pts to
    pred_tracks[t], then re-project query_pts through H to get geometrically
    consistent (planar) track positions.

    If RANSAC fails or yields fewer than min_inliers, the raw predictions are
    kept for that frame unchanged.

    Args:
        query_pts:    (N, 2) float32 — frame-0 GT positions (original px)
        pred_tracks:  (S, N, 2) float32 — raw tracker output (original px)
        min_inliers:  int   — minimum inliers to accept H
        ransac_thresh: float — RANSAC reprojection threshold in pixels

    Returns:
        refined:       (S, N, 2) float32 — H-refined track positions
        inlier_ratios: (S,)      float32 — inlier fraction per frame
                                           (frame 0 = 1.0 by definition)
    """
    S, N, _ = pred_tracks.shape
    refined = pred_tracks.copy()
    inlier_ratios = np.ones(S, dtype=np.float32)

    src = query_pts.astype(np.float32)           # (N, 2)
    src_h = np.concatenate(                      # (N, 3) homogeneous
        [src, np.ones((N, 1), dtype=np.float32)], axis=1
    )

    homographies = np.stack([np.eye(3, dtype=np.float32)] * S)  # (S, 3, 3)

    for t in range(1, S):
        dst = pred_tracks[t].astype(np.float32)  # (N, 2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)

        n_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratios[t] = n_inliers / N if N > 0 else 0.0

        if H is None or n_inliers < min_inliers:
            # Not enough support — keep raw predictions for this frame
            continue

        homographies[t] = H.astype(np.float32)

        # Project query points through H
        proj = (H @ src_h.T).T          # (N, 3)
        proj /= proj[:, 2:3]            # normalise homogeneous coord
        refined[t] = proj[:, :2]

    return refined, inlier_ratios, homographies


# ---------------------------------------------------------------------------
# GT corner loading + corner/centroid/jitter metrics
# ---------------------------------------------------------------------------

def load_gt_corners(seq_dir):
    """
    Load GT plane corners from ae_data/corners.csv.

    CSV format (one row per frame):
        frame, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y

    Returns (frame_indices, corners) or (None, None) if the file is absent.
        frame_indices: (N,) int32
        corners:       (N, 4, 2) float32  — order: TL, TR, BL, BR
    """
    csv_path = os.path.join(seq_dir, "ae_data", "corners.csv")
    if not os.path.isfile(csv_path):
        return None, None
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis]
    frame_indices = data[:, 0].astype(np.int32)
    # columns 1-8: tl_x tl_y  tr_x tr_y  bl_x bl_y  br_x br_y → (N, 4, 2)
    corners = data[:, 1:].reshape(-1, 4, 2).astype(np.float32)
    return frame_indices, corners


def compute_corner_metrics(homographies, gt_frame_indices, gt_corners):
    """
    Evaluate predicted homographies against Mocha GT corners.

    For each frame t that has GT data, the reference corners (frame 0) are
    projected through H_t and compared to the GT corners at frame t.

    Args:
        homographies:     (S, 3, 3) float32 — one per sequence frame
        gt_frame_indices: (N,) int32        — frame numbers with GT data
        gt_corners:       (N, 4, 2) float32 — TL, TR, BL, BR per frame

    Returns dict with CORNER_METRIC_KEYS, or None if no overlap.
    """
    # Reference corners must exist at frame 0
    frame0_mask = gt_frame_indices == 0
    if not frame0_mask.any():
        return None
    ref_corners = gt_corners[frame0_mask][0].astype(np.float64)  # (4, 2)
    ref_h = np.concatenate(
        [ref_corners, np.ones((4, 1), dtype=np.float64)], axis=1
    )  # (4, 3)

    gt_lookup = {int(gt_frame_indices[i]): gt_corners[i]
                 for i in range(len(gt_frame_indices))}

    centroid_errors = []
    corner_errors   = []
    pred_centroids  = []

    for t in range(len(homographies)):
        if t not in gt_lookup:
            continue
        H    = homographies[t].astype(np.float64)
        gt_c = gt_lookup[t].astype(np.float64)   # (4, 2)

        # Project reference corners through H_t
        proj  = (H @ ref_h.T).T                   # (4, 3)
        proj /= proj[:, 2:3]
        pred_c = proj[:, :2]                       # (4, 2)

        # Per-corner L2, averaged over 4 corners
        corner_errors.append(float(np.linalg.norm(pred_c - gt_c, axis=1).mean()))

        # Centroid L2
        pred_cen = pred_c.mean(axis=0)
        gt_cen   = gt_c.mean(axis=0)
        centroid_errors.append(float(np.linalg.norm(pred_cen - gt_cen)))
        pred_centroids.append(pred_cen)

    if not centroid_errors:
        return None

    pt  = np.array(centroid_errors)
    ce  = np.array(corner_errors)
    ctr = np.array(pred_centroids)   # (M, 2)

    # Jitter: RMS of the third-order finite difference of centroid trajectory
    jitter = 0.0
    if len(ctr) >= 4:
        jerk   = np.diff(ctr, n=3, axis=0)        # (M-3, 2)
        jitter = float(np.sqrt((jerk ** 2).sum(axis=1).mean()))

    return {
        "pt_mean_px":      float(pt.mean()),
        "pt_median_px":    float(np.median(pt)),
        "pt_max_px":       float(pt.max()),
        "pt_p95_px":       float(np.percentile(pt, 95)),
        "corner_mean_px":  float(ce.mean()),
        "corner_median_px": float(np.median(ce)),
        "corner_max_px":   float(ce.max()),
        "corner_p95_px":   float(np.percentile(ce, 95)),
        "jitter_score":    jitter,
    }


def project_corners_through_homographies(ref_corners, homographies):
    """
    Project reference corners through every H_t.

    Args:
        ref_corners:  (4, 2) float32 — quad corners at frame 0 (TL,TR,BL,BR)
        homographies: (S, 3, 3) float32

    Returns: (S, 4, 2) float32 — predicted corners per frame
    """
    S = len(homographies)
    ref_h = np.concatenate(
        [ref_corners.astype(np.float64),
         np.ones((4, 1), dtype=np.float64)], axis=1
    )  # (4, 3)
    pred = np.zeros((S, 4, 2), dtype=np.float32)
    for t in range(S):
        proj  = (homographies[t].astype(np.float64) @ ref_h.T).T  # (4, 3)
        proj /= proj[:, 2:3]
        pred[t] = proj[:, :2].astype(np.float32)
    return pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pred_tracks_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh
):
    """
    All coordinates in original image resolution.

    pred_tracks_orig: (S, N, 2)
    gt_tracks_orig:   (S, N, 2)
    gt_vis_mask:      (S, N) bool
    pred_vis:         (S, N) float

    Frame 0 is the query frame and is skipped.
    Returns dict or None if no visible GT points in frames 1..S.
    """
    gt_t = gt_tracks_orig[1:]
    pred_t = pred_tracks_orig[1:]
    gt_mask = gt_vis_mask[1:]
    pred_v = pred_vis[1:]

    diff = pred_t - gt_t
    l2 = np.sqrt((diff ** 2).sum(-1))  # (S-1, N)
    valid_l2 = l2[gt_mask]

    if len(valid_l2) == 0:
        return None

    pred_vis_bool = pred_v > vis_thresh
    vis_acc = float((pred_vis_bool == gt_mask).mean())

    return {
        "ate": float(valid_l2.mean()),
        "median_te": float(np.median(valid_l2)),
        "delta_1px": float((valid_l2 < 1.0).mean()),
        "delta_2px": float((valid_l2 < 2.0).mean()),
        "delta_5px": float((valid_l2 < 5.0).mean()),
        "delta_10px": float((valid_l2 < 10.0).mean()),
        "vis_acc": vis_acc,
        "n_points": int(len(valid_l2)),
        "n_tracks": int(gt_vis_mask.shape[1]),
        "n_frames": int(gt_vis_mask.shape[0]),
    }


# ---------------------------------------------------------------------------
# Per-sequence evaluation  (returns raw predictions for optional viz)
# ---------------------------------------------------------------------------

def eval_sequence(
    seq_name, dataset_dir, model, chunk_size, vis_thresh, device, dtype,
    ransac_thresh=3.0, min_h_inliers=8,
    # Pre-loaded by eval_split (avoids re-extracting video frames per model)
    image_paths=None, W_orig=None, H_orig=None,
    gt_tracks_orig=None, gt_vis_mask=None,
):
    """
    Returns: (metrics, metrics_h, metrics_corners, mean_inlier_ratio, error, preds, preds_h)
      metrics:           dict or None  — raw track head metrics
      metrics_h:         dict or None  — homography-refined metrics
      metrics_corners:   dict or None  — corner/centroid/jitter vs Mocha GT
      mean_inlier_ratio: float         — mean RANSAC inlier fraction (frames 1..S)
      error:             str or None
      preds:             (pred_tracks_orig, pred_vis) in original px, or None
      preds_h:           (pred_h_orig, pred_vis, homographies, inlier_ratios), or None
    """
    seq_dir = os.path.join(dataset_dir, seq_name)

    # Data loading is done once per sequence in eval_split and passed in;
    # fall back to self-loading only when called standalone.
    if image_paths is None or gt_tracks_orig is None:
        image_paths, W_orig, H_orig, gt_tracks_orig, gt_vis_mask, _tmp, err = \
            load_seq_data(seq_dir)
        if err:
            return None, None, None, 0.0, err, None, None

    S = len(image_paths)
    if S < 2:
        return None, None, None, 0.0, "need at least 2 frames", None, None

    _, _, W_model, H_model = get_model_resolution((W_orig, H_orig))
    scale_x = W_model / W_orig
    scale_y = H_model / H_orig

    images_tensor = load_and_preprocess_images(image_paths, mode="crop")

    q_orig = gt_tracks_orig[0].copy()  # (N, 2)
    q_model = np.stack(
        [q_orig[:, 0] * scale_x, q_orig[:, 1] * scale_y], axis=-1
    ).astype(np.float32)

    try:
        import time
        _t0 = time.perf_counter()
        pred_model, pred_vis = run_track_head(
            model, images_tensor, q_model, chunk_size, device, dtype
        )
        runtime_s = time.perf_counter() - _t0
    except RuntimeError as exc:
        return None, None, None, 0.0, f"runtime error: {exc}", None, None

    # Scale predictions back to original space
    pred_orig = np.stack(
        [pred_model[:, :, 0] / scale_x, pred_model[:, :, 1] / scale_y],
        axis=-1,
    )

    # Raw metrics
    metrics = compute_metrics(
        pred_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh
    )
    if metrics is not None:
        metrics["runtime_s"] = runtime_s

    # Homography-refined tracks and metrics
    pred_h_orig, inlier_ratios, homographies = estimate_homography_refined_tracks(
        q_orig, pred_orig,
        min_inliers=min_h_inliers,
        ransac_thresh=ransac_thresh,
    )
    metrics_h = compute_metrics(
        pred_h_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh
    )
    mean_inlier_ratio = float(inlier_ratios[1:].mean()) if S > 1 else 1.0
    if metrics_h is not None:
        metrics_h["inlier_ratio"] = mean_inlier_ratio

    # Corner / centroid / jitter metrics vs Mocha GT corners
    gt_frame_indices, gt_corners = load_gt_corners(seq_dir)
    metrics_corners = None
    if gt_frame_indices is not None:
        metrics_corners = compute_corner_metrics(
            homographies, gt_frame_indices, gt_corners
        )

    return (
        metrics, metrics_h, metrics_corners, mean_inlier_ratio,
        None,
        (pred_orig, pred_vis),
        (pred_h_orig, pred_vis, homographies, inlier_ratios),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Panel definitions: (tag, BGR-compatible RGB color, label)
# Order determines left-to-right panel order in the output video.
_GT_COLOR = (0, 220, 0)
_PANEL_DEFS = [
    ("vanilla",     (220, 60,  60),  "Vanilla"),
    ("vanilla_h",   (255, 150, 50),  "Vanilla+H"),
    ("finetuned",   (60,  130, 255), "Finetuned"),
    ("finetuned_h", (60,  210, 160), "Finetuned+H"),
]


def _render_frame(image_path, fi, gt_tracks_orig, gt_vis_mask,
                  preds_by_tag, sampled_ids, sx, sy, W_disp, H_disp,
                  draw_tracks=True, draw_quads=False,
                  gt_frame_lookup=None, pred_corners_by_tag=None,
                  single_tag=None, gt_only=False):
    """
    Render one video frame as a numpy uint8 RGB array (H, total_W, 3).

    One panel per entry in _PANEL_DEFS that has a key in preds_by_tag.
    Panels are separated by a 2px grey divider.

    draw_tracks: draw GT dots (green filled) + prediction dots (colored outline)
    draw_quads:  draw GT plane quad (green) + predicted plane quad (tag color)

    Corner order: TL(0), TR(1), BL(2), BR(3).
    Quad draw order (non-self-intersecting): TL→TR→BR→BL = [0,1,3,2].
    """
    def load_bg(path):
        arr = np.array(Image.open(path).convert("RGB"))
        H_raw, W_raw = arr.shape[:2]
        if W_raw == W_disp and H_raw == H_disp:
            return arr
        # Pad to even dimensions — never interpolate, preserves every pixel
        out = np.zeros((H_disp, W_disp, 3), dtype=np.uint8)
        out[:min(H_raw, H_disp), :min(W_raw, W_disp)] = arr[:H_disp, :W_disp]
        return out

    bg = load_bg(image_path)

    _SHIFT = 4
    _S16 = 1 << _SHIFT          # 16 — subpixel scale factor
    dot_r = 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar_h = 20
    _QUAD_ORDER = [0, 1, 3, 2]

    # GT-only panel: background with GT overlay, no predictions
    if gt_only:
        panel = bg.copy()
        if draw_tracks:
            for tid in sampled_ids:
                if gt_vis_mask[fi, tid]:
                    gx_s = round(float(gt_tracks_orig[fi, tid, 0]) * sx * _S16)
                    gy_s = round(float(gt_tracks_orig[fi, tid, 1]) * sy * _S16)
                    cv2.circle(panel, (gx_s, gy_s), dot_r * _S16,
                               _GT_COLOR[::-1], -1, cv2.LINE_AA, _SHIFT)
        if draw_quads and gt_frame_lookup is not None and fi in gt_frame_lookup:
            # scale_corners needs _QUAD_ORDER defined — defined below, inline here
            sc = gt_frame_lookup[fi].astype(np.float64)
            sc_x = np.rint(sc[:, 0] * sx * _S16)
            sc_y = np.rint(sc[:, 1] * sy * _S16)
            pts = np.stack([sc_x, sc_y], axis=1)[[0, 1, 3, 2]].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(panel, [pts], True, _GT_COLOR[::-1], 2, cv2.LINE_AA, _SHIFT)
            for pt in pts[:, 0]:
                cv2.circle(panel, tuple(pt), 5 * _S16, _GT_COLOR[::-1], -1, cv2.LINE_AA, _SHIFT)
        cv2.putText(panel, "GT", (8, 22), font, 0.7, _GT_COLOR[::-1], 2, cv2.LINE_AA)
        cv2.putText(panel, f"frame {fi}", (8, H_disp - 8), font, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        gt_legend = "GT plane" if draw_quads and not draw_tracks else "GT"
        panel[-bar_h:] = (20, 20, 20)
        cv2.circle(panel[-bar_h:], (10, bar_h // 2), 5, _GT_COLOR[::-1], -1)
        cv2.putText(panel[-bar_h:], gt_legend, (18, bar_h - 5), font, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
        return panel

    def _scale_corners(corners_4x2):
        """
        Scale (4,2) corners from original px → display px → fixed-point
        subpixel space (multiply by _S16), matching the shift=_SHIFT
        convention used by cv2.circle / cv2.polylines with shift=_SHIFT.
        No rounding or flooring — uses round-half-to-even via np.rint.
        """
        sc = corners_4x2.astype(np.float64)
        sc[:, 0] = np.rint(sc[:, 0] * sx * _S16)
        sc[:, 1] = np.rint(sc[:, 1] * sy * _S16)
        return sc[_QUAD_ORDER].astype(np.int32).reshape(-1, 1, 2)

    panel_defs = (
        [(t, c, l) for t, c, l in _PANEL_DEFS if t == single_tag]
        if single_tag is not None else _PANEL_DEFS
    )
    panels = []
    for tag, color, label in panel_defs:
        if tag not in preds_by_tag:
            continue

        panel = bg.copy()

        # --- Track dots ---
        if draw_tracks:
            pred_tracks = preds_by_tag[tag][0]  # (S, N, 2)
            for tid in sampled_ids:
                if gt_vis_mask[fi, tid]:
                    gx_s = round(float(gt_tracks_orig[fi, tid, 0]) * sx * _S16)
                    gy_s = round(float(gt_tracks_orig[fi, tid, 1]) * sy * _S16)
                    cv2.circle(panel, (gx_s, gy_s), dot_r * _S16,
                               _GT_COLOR[::-1], -1, cv2.LINE_AA, _SHIFT)
                px_s = round(float(pred_tracks[fi, tid, 0]) * sx * _S16)
                py_s = round(float(pred_tracks[fi, tid, 1]) * sy * _S16)
                cv2.circle(panel, (px_s, py_s), dot_r * _S16,
                           color[::-1], 2, cv2.LINE_AA, _SHIFT)

        # --- Plane quads (all coords in fixed-point subpixel space) ---
        if draw_quads:
            corner_r = 5 * _S16  # radius also in subpixel space
            if gt_frame_lookup is not None and fi in gt_frame_lookup:
                pts = _scale_corners(gt_frame_lookup[fi])
                cv2.polylines(panel, [pts], True, _GT_COLOR[::-1], 2,
                              cv2.LINE_AA, _SHIFT)
                for pt in pts[:, 0]:
                    cv2.circle(panel, tuple(pt), corner_r,
                               _GT_COLOR[::-1], -1, cv2.LINE_AA, _SHIFT)

            if pred_corners_by_tag is not None and tag in pred_corners_by_tag:
                pts = _scale_corners(pred_corners_by_tag[tag][fi])
                cv2.polylines(panel, [pts], True, color[::-1], 2,
                              cv2.LINE_AA, _SHIFT)
                for pt in pts[:, 0]:
                    cv2.circle(panel, tuple(pt), corner_r,
                               color[::-1], 1, cv2.LINE_AA, _SHIFT)

        # Panel label + frame index
        cv2.putText(panel, label, (8, 22), font, 0.7, color[::-1], 2, cv2.LINE_AA)
        cv2.putText(panel, f"frame {fi}", (8, H_disp - 8), font, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # Legend bar
        gt_legend  = "GT plane"  if draw_quads and not draw_tracks else "GT"
        pred_legend = "Pred plane" if draw_quads and not draw_tracks else "Pred"
        panel[-bar_h:] = (20, 20, 20)
        cv2.circle(panel[-bar_h:], (10, bar_h // 2), 5, _GT_COLOR[::-1], -1)
        cv2.putText(panel[-bar_h:], gt_legend,   (18, bar_h - 5), font, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
        lx = 18 + len(gt_legend) * 7 + 10
        cv2.circle(panel[-bar_h:], (lx, bar_h // 2), 5, color[::-1], 2)
        cv2.putText(panel[-bar_h:], pred_legend, (lx + 8, bar_h - 5), font, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)

        panels.append(panel)

    if not panels:
        return np.zeros((H_disp, W_disp, 3), dtype=np.uint8)
    divider = np.full((H_disp, 2, 3), 180, dtype=np.uint8)
    frame = panels[0]
    for p in panels[1:]:
        frame = np.concatenate([frame, divider, p], axis=1)
    return frame


def _encode_video(frames_dir, out_path, fps):
    """Encode a directory of JPEG frames to H.264 mp4 via ffmpeg."""
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%06d.jpg"),
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [viz] ffmpeg failed (code {result.returncode}):")
        print(result.stderr[-600:])
        return False
    return True


def _viz_common_setup(image_paths, gt_vis_mask, preds_by_tag, n_tracks,
                      W_orig=None, H_orig=None):
    """Shared setup for both viz functions. Returns (sampled_ids, sx, sy, W, H)."""
    S = len(image_paths)
    if S < 2 or not preds_by_tag:
        return None

    valid_ids = np.where(gt_vis_mask[1])[0]
    if len(valid_ids) == 0:
        return None

    rng = np.random.RandomState(42)
    sampled_ids = rng.choice(valid_ids, size=min(n_tracks, len(valid_ids)), replace=False)

    if W_orig is None or H_orig is None:
        W_orig, H_orig, _, _ = get_model_resolution(image_paths[0])

    # Render at original resolution — tracks are already in original pixel space
    W_disp = W_orig + (W_orig % 2)
    H_disp = H_orig + (H_orig % 2)
    sx = 1.0
    sy = 1.0
    return sampled_ids, sx, sy, W_disp, H_disp


def visualize_sequence(
    seq_name, image_paths, gt_tracks_orig, gt_vis_mask,
    preds_by_tag, split_name, vis_dir, n_tracks, fps=8,
    W_orig=None, H_orig=None, seq_dir=None,
):
    """
    Save track-dot comparison videos: composed + one per base model.
    Outputs:
      <vis_dir>/<split_name>/<seq_name>_tracks_composed.mp4
      <vis_dir>/<split_name>/<seq_name>_tracks_<tag>.mp4  (vanilla, finetuned)
    FPS is read from seq_dir/original.mp4 when seq_dir is provided.
    """
    import tempfile, shutil

    setup = _viz_common_setup(image_paths, gt_vis_mask, preds_by_tag, n_tracks,
                              W_orig=W_orig, H_orig=H_orig)
    if setup is None:
        return
    sampled_ids, sx, sy, W_disp, H_disp = setup
    S = len(image_paths)

    if seq_dir is not None:
        fps = get_sequence_fps(seq_dir, fallback=fps)

    out_dir = os.path.join(vis_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    def _save_video(render_kwargs, suffix):
        out_path = os.path.join(out_dir, f"{seq_name}_{suffix}.mp4")
        tmp_dir = tempfile.mkdtemp(prefix="track_viz_")
        try:
            for fi in range(S):
                frame_rgb = _render_frame(
                    image_paths[fi], fi,
                    gt_tracks_orig, gt_vis_mask,
                    preds_by_tag, sampled_ids,
                    sx, sy, W_disp, H_disp,
                    draw_tracks=True, draw_quads=False,
                    **render_kwargs,
                )
                cv2.imwrite(os.path.join(tmp_dir, f"{fi:06d}.jpg"),
                            frame_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if _encode_video(tmp_dir, out_path, fps):
                print(f"    [viz tracks] -> {out_path}  ({S} frames @ {fps:.1f}fps)")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Composed (all panels side by side)
    _save_video({}, "tracks_composed")

    # Per-model (base tags only — no _h suffix)
    base_tags = [tag for tag in preds_by_tag if not tag.endswith("_h")]
    for tag in base_tags:
        _save_video({"single_tag": tag}, f"tracks_{tag}")


def visualize_planes(
    seq_name, image_paths, gt_tracks_orig, gt_vis_mask,
    preds_by_tag, split_name, vis_dir, n_tracks, fps=8,
    gt_frame_lookup=None, pred_corners_by_tag=None,
    W_orig=None, H_orig=None, seq_dir=None,
):
    """
    Save plane-quad comparison videos: GT + per-model + composed.
    Outputs:
      <vis_dir>/<split_name>/<seq_name>_planes_gt.mp4
      <vis_dir>/<split_name>/<seq_name>_planes_<tag>.mp4  (vanilla, finetuned)
      <vis_dir>/<split_name>/<seq_name>_planes_composed.mp4
    FPS is read from seq_dir/original.mp4 when seq_dir is provided.
    Skipped silently if no GT corners are available (corners.csv missing).
    """
    import tempfile, shutil

    if gt_frame_lookup is None or pred_corners_by_tag is None:
        return

    setup = _viz_common_setup(image_paths, gt_vis_mask, preds_by_tag, n_tracks,
                              W_orig=W_orig, H_orig=H_orig)
    if setup is None:
        return
    sampled_ids, sx, sy, W_disp, H_disp = setup
    S = len(image_paths)

    if seq_dir is not None:
        fps = get_sequence_fps(seq_dir, fallback=fps)

    out_dir = os.path.join(vis_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    def _save_video(render_kwargs, suffix):
        out_path = os.path.join(out_dir, f"{seq_name}_{suffix}.mp4")
        tmp_dir = tempfile.mkdtemp(prefix="planes_viz_")
        try:
            for fi in range(S):
                frame_rgb = _render_frame(
                    image_paths[fi], fi,
                    gt_tracks_orig, gt_vis_mask,
                    preds_by_tag, sampled_ids,
                    sx, sy, W_disp, H_disp,
                    draw_tracks=False, draw_quads=True,
                    gt_frame_lookup=gt_frame_lookup,
                    pred_corners_by_tag=pred_corners_by_tag,
                    **render_kwargs,
                )
                cv2.imwrite(os.path.join(tmp_dir, f"{fi:06d}.jpg"),
                            frame_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if _encode_video(tmp_dir, out_path, fps):
                print(f"    [viz planes] -> {out_path}  ({S} frames @ {fps:.1f}fps)")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # GT-only
    _save_video({"gt_only": True}, "planes_gt")

    # Per-model (base tags only)
    base_tags = [tag for tag in preds_by_tag if not tag.endswith("_h")]
    for tag in base_tags:
        _save_video({"single_tag": tag}, f"planes_{tag}")

    # Composed (all panels side by side)
    _save_video({}, "planes_composed")


# ---------------------------------------------------------------------------
# Split-level evaluation
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "ate", "median_te",
    "delta_1px", "delta_2px", "delta_5px", "delta_10px",
    "vis_acc",
]

CORNER_METRIC_KEYS = [
    "pt_mean_px", "pt_median_px", "pt_max_px", "pt_p95_px",
    "corner_mean_px", "corner_median_px", "corner_max_px", "corner_p95_px",
    "jitter_score",
]


def mean_over_seqs(seq_results, extra_keys=()):
    all_keys = list(METRIC_KEYS) + list(extra_keys)
    vals = {k: [] for k in all_keys}
    for m in seq_results.values():
        for k in all_keys:
            if k in m:
                vals[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in vals.items() if v}


def eval_split(
    split_name, sequences, dataset_dir, models,
    chunk_size, vis_thresh, device, dtype,
    vis_max_seqs=3, vis_n_tracks=50, vis_fps=8, vis_dir="track_eval_viz",
    ransac_thresh=3.0, min_h_inliers=8,
    homography_dir=None,
    ae_output_dir=None,
):
    """
    Evaluate all sequences in one split for every model in `models`.

    Loops sequence-first so predictions from all models are available
    together for visualization before moving to the next sequence.

    models: dict  tag -> model
    Returns: dict  (per-seq results + '<tag>_mean' aggregates)
    """
    results = {tag: {} for tag in models}
    results.update({f"{tag}_h": {} for tag in models})
    results_corners = {tag: {} for tag in models}
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"SPLIT: {split_name.upper()} ({len(sequences)} sequences)")
    print(sep)

    viz_count = 0

    for seq in sequences:
        print(f"\n  [{seq}]")

        seq_dir = os.path.join(dataset_dir, seq)
        image_paths = sorted(
            glob.glob(os.path.join(seq_dir, "images", "*"))
        )
        tracks_path = os.path.join(seq_dir, "tracks.npy")
        masks_path = os.path.join(seq_dir, "track_masks.npy")

        # Load GT once, shared across all models
        if (
            image_paths
            and os.path.isfile(tracks_path)
        ):
            gt_tracks_orig = np.load(tracks_path).astype(np.float32)
            if os.path.isfile(masks_path):
                gt_vis_mask = np.load(masks_path).astype(bool)
            else:
                gt_vis_mask = np.ones(
                    gt_tracks_orig.shape[:2], dtype=bool
                )
            S = len(image_paths)
            gt_tracks_orig = gt_tracks_orig[:S]
            gt_vis_mask = gt_vis_mask[:S]
        else:
            gt_tracks_orig = None
            gt_vis_mask = None

        preds_by_tag = {}
        pred_corners_by_tag = {}

        # Load frames + GT data ONCE per sequence (shared across all models)
        import shutil as _shutil
        seq_dir = os.path.join(dataset_dir, seq)
        image_paths_seq, W_orig_seq, H_orig_seq, gt_tracks_seq, gt_vis_seq, tmp_dir_seq, load_err = \
            load_seq_data(seq_dir)

        if load_err:
            print(f"    SKIP ({load_err})")
            continue

        # Load GT corners once per sequence for quad visualization
        gt_frame_indices_seq, gt_corners_seq = load_gt_corners(seq_dir)
        gt_frame_lookup = (
            {int(gt_frame_indices_seq[i]): gt_corners_seq[i]
             for i in range(len(gt_frame_indices_seq))}
            if gt_frame_indices_seq is not None else None
        )

        for tag, model in models.items():
            metrics, metrics_h, metrics_corners, mean_inlier, err, preds, preds_h = eval_sequence(
                seq, dataset_dir, model,
                chunk_size, vis_thresh, device, dtype,
                ransac_thresh=ransac_thresh,
                min_h_inliers=min_h_inliers,
                image_paths=image_paths_seq,
                W_orig=W_orig_seq, H_orig=H_orig_seq,
                gt_tracks_orig=gt_tracks_seq, gt_vis_mask=gt_vis_seq,
            )
            if err:
                print(f"    {tag:10s}  SKIP ({err})")
                continue

            results[tag][seq] = metrics
            if metrics_h is not None:
                results[f"{tag}_h"][seq] = metrics_h

            if preds is not None:
                preds_by_tag[tag] = preds
            if preds_h is not None:
                pred_h_orig, pred_vis_h, homographies, _ = preds_h
                preds_by_tag[f"{tag}_h"] = (pred_h_orig, pred_vis_h)

                # Predicted plane corners for quad overlay
                if gt_frame_lookup is not None and gt_frame_indices_seq is not None:
                    frame0_mask = gt_frame_indices_seq == 0
                    if frame0_mask.any():
                        ref_corners = gt_corners_seq[frame0_mask][0]
                        pc = project_corners_through_homographies(ref_corners, homographies)
                        pred_corners_by_tag[tag]          = pc
                        pred_corners_by_tag[f"{tag}_h"]   = pc  # same H

            # Save homographies
            if homography_dir is not None and preds_h is not None:
                _, _, homographies, inlier_ratios = preds_h
                h_seq_dir = os.path.join(homography_dir, split_name, seq)
                os.makedirs(h_seq_dir, exist_ok=True)
                np.save(os.path.join(h_seq_dir, f"{tag}_H.npy"), homographies)
                np.save(os.path.join(h_seq_dir, f"{tag}_inliers.npy"), inlier_ratios)

            # Raw metrics line
            print(
                f"    {tag:10s}  "
                f"ATE={metrics['ate']:6.2f}px  "
                f"med={metrics['median_te']:6.2f}px  "
                f"d1={metrics['delta_1px']:.3f}  "
                f"d2={metrics['delta_2px']:.3f}  "
                f"d5={metrics['delta_5px']:.3f}  "
                f"vis={metrics['vis_acc']:.3f}  "
                f"N={metrics['n_points']}  "
                f"t={metrics.get('runtime_s', 0.0):.2f}s"
            )
            # H-refined metrics line
            if metrics_h is not None:
                h_ate_delta = metrics_h["ate"] - metrics["ate"]
                print(
                    f"    {'':10s}  "
                    f"[+H] ATE={metrics_h['ate']:6.2f}px  "
                    f"med={metrics_h['median_te']:6.2f}px  "
                    f"d5={metrics_h['delta_5px']:.3f}  "
                    f"inlier={mean_inlier:.2f}  "
                    f"ΔATE={h_ate_delta:+.2f}px"
                )

            # Corner / centroid / jitter metrics
            if metrics_corners is not None:
                results_corners[tag][seq] = metrics_corners
                print(
                    f"    {'':10s}  "
                    f"[corners] "
                    f"pt={metrics_corners['pt_mean_px']:.2f}px  "
                    f"corner={metrics_corners['corner_mean_px']:.2f}px  "
                    f"jitter={metrics_corners['jitter_score']:.3f}"
                )

            # Save per-tracker JSON for aggregate_metrics.py
            if ae_output_dir is not None and metrics_corners is not None:
                ae_seq_out = os.path.join(ae_output_dir, tag, "metrics")
                os.makedirs(ae_seq_out, exist_ok=True)
                ae_record = {
                    "stem": seq,
                    "n_frames": metrics["n_frames"],
                    "runtime_s": metrics.get("runtime_s", 0.0),
                    **metrics_corners,
                }
                with open(os.path.join(ae_seq_out, f"{seq}.json"), "w") as fh:
                    json.dump(ae_record, fh, indent=2)

        # Visualization for first vis_max_seqs sequences in this split
        # vis_max_seqs == 0: disabled; > 0: first N; -1: all
        if (
            vis_max_seqs != 0
            and (vis_max_seqs < 0 or viz_count < vis_max_seqs)
            and preds_by_tag
            and gt_tracks_seq is not None
            and image_paths_seq
        ):
            visualize_sequence(
                seq, image_paths_seq,
                gt_tracks_seq, gt_vis_seq,
                preds_by_tag, split_name, vis_dir,
                n_tracks=vis_n_tracks, fps=vis_fps,
                W_orig=W_orig_seq, H_orig=H_orig_seq, seq_dir=seq_dir,
            )
            visualize_planes(
                seq, image_paths_seq,
                gt_tracks_seq, gt_vis_seq,
                preds_by_tag, split_name, vis_dir,
                n_tracks=vis_n_tracks, fps=vis_fps,
                gt_frame_lookup=gt_frame_lookup,
                pred_corners_by_tag=pred_corners_by_tag,
                W_orig=W_orig_seq, H_orig=H_orig_seq, seq_dir=seq_dir,
            )
            viz_count += 1

        # Clean up temp frame dir (video format only)
        if tmp_dir_seq is not None:
            _shutil.rmtree(tmp_dir_seq, ignore_errors=True)

    # Aggregate stats + comparison
    print(f"\n  --- {split_name.upper()} aggregate ---")
    agg_by_tag = {}
    for tag in models:
        # Raw
        seqs = results[tag]
        if seqs:
            agg = mean_over_seqs(seqs)
            agg_by_tag[tag] = agg
            results[f"{tag}_mean"] = agg
            print(
                f"  {tag.upper()} ({len(seqs)} seqs)  "
                f"ATE={agg['ate']:.2f}px  "
                f"med={agg['median_te']:.2f}px  "
                f"d1={agg['delta_1px']:.3f}  "
                f"d2={agg['delta_2px']:.3f}  "
                f"d5={agg['delta_5px']:.3f}  "
                f"vis={agg['vis_acc']:.3f}"
            )
        else:
            print(f"  {tag.upper()}: no sequences evaluated")

        # H-refined
        seqs_h = results[f"{tag}_h"]
        if seqs_h:
            agg_h = mean_over_seqs(seqs_h, extra_keys=("inlier_ratio",))
            agg_by_tag[f"{tag}_h"] = agg_h
            results[f"{tag}_h_mean"] = agg_h
            inlier_str = (
                f"  inlier={agg_h['inlier_ratio']:.2f}"
                if "inlier_ratio" in agg_h else ""
            )
            print(
                f"  {tag.upper()}+H ({len(seqs_h)} seqs)  "
                f"ATE={agg_h['ate']:.2f}px  "
                f"med={agg_h['median_te']:.2f}px  "
                f"d1={agg_h['delta_1px']:.3f}  "
                f"d2={agg_h['delta_2px']:.3f}  "
                f"d5={agg_h['delta_5px']:.3f}"
                f"{inlier_str}"
            )

    # Corner metric aggregates
    print(f"\n  --- {split_name.upper()} corner metrics (vs Mocha GT) ---")
    for tag in models:
        seqs_c = results_corners[tag]
        if not seqs_c:
            continue
        agg_c = mean_over_seqs(seqs_c, extra_keys=CORNER_METRIC_KEYS)
        results[f"{tag}_corners_mean"] = agg_c
        print(
            f"  {tag.upper()} ({len(seqs_c)} seqs)  "
            f"pt_mean={agg_c.get('pt_mean_px', float('nan')):.2f}px  "
            f"pt_p95={agg_c.get('pt_p95_px', float('nan')):.2f}px  "
            f"corner_mean={agg_c.get('corner_mean_px', float('nan')):.2f}px  "
            f"corner_p95={agg_c.get('corner_p95_px', float('nan')):.2f}px  "
            f"jitter={agg_c.get('jitter_score', float('nan')):.3f}"
        )

    # Pairwise comparisons
    pairs = [
        ("vanilla", "finetuned", "vanilla -> finetuned"),
        ("vanilla", "vanilla_h", "vanilla -> vanilla+H"),
        ("finetuned", "finetuned_h", "finetuned -> finetuned+H"),
    ]
    for tag_a, tag_b, label in pairs:
        if tag_a not in agg_by_tag or tag_b not in agg_by_tag:
            continue
        a = agg_by_tag[tag_a]
        b = agg_by_tag[tag_b]
        ate_d = b["ate"] - a["ate"]
        d5_d = b["delta_5px"] - a["delta_5px"]
        pct = ate_d / a["ate"] * 100
        print(f"\n  IMPROVEMENT {label}:")
        print(
            f"    ATE:  {a['ate']:.2f} -> {b['ate']:.2f} px  "
            f"({ate_d:+.2f}px, {pct:+.1f}%)"
        )
        print(
            f"    d5:   {a['delta_5px']:.3f} -> {b['delta_5px']:.3f}  "
            f"({d5_d * 100:+.1f}pp)"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    capable = torch.cuda.get_device_capability()[0] >= 8
    dtype = torch.bfloat16 if capable else torch.float16

    def load_sequences(path):
        with open(path) as fh:
            return [ln.strip() for ln in fh if ln.strip()]

    def discover_sequences(dataset_dir):
        """
        Return sorted list of valid sequence subdirectories.
        Supports two layouts:
          Format A: images/ folder  + tracks.npy at seq root
          Format B: original.mp4   + ae_data/tracks.npy
        """
        seqs = []
        for name in sorted(os.listdir(dataset_dir)):
            seq_dir = os.path.join(dataset_dir, name)
            if not os.path.isdir(seq_dir):
                continue
            format_a = (
                os.path.isdir(os.path.join(seq_dir, "images"))
                and os.path.isfile(os.path.join(seq_dir, "tracks.npy"))
            )
            format_b = (
                os.path.isfile(os.path.join(seq_dir, "original.mp4"))
                and os.path.isfile(os.path.join(seq_dir, "ae_data", "tracks.npy"))
            )
            if format_a or format_b:
                seqs.append(name)
        return seqs

    # Build splits — fall back to auto-discovery if no split files given
    use_split_files = args.train_split_file or args.val_split_file
    if use_split_files:
        train_seqs = load_sequences(args.train_split_file) if args.train_split_file else []
        val_seqs   = load_sequences(args.val_split_file)   if args.val_split_file   else []

        # Subsample train split
        n_total = len(train_seqs)
        max_tr = args.train_max_seqs
        if max_tr > 0 and n_total > max_tr:
            train_seqs = random.sample(train_seqs, max_tr)
            print(f"Train split: sampled {max_tr} / {n_total} sequences")
        else:
            print(f"Train split: {n_total} sequences")
        print(f"Val   split: {len(val_seqs)} sequences")
        splits = [("train", train_seqs), ("val", val_seqs)]
    else:
        eval_seqs = discover_sequences(args.dataset_dir)
        if not eval_seqs:
            raise ValueError(
                f"No valid sequences found in {args.dataset_dir}. "
                "Each sequence must have an images/ subdirectory and tracks.npy."
            )
        print(f"Auto-discovered {len(eval_seqs)} sequences in {args.dataset_dir}")
        splits = [("eval", eval_seqs)]

    if args.vis_max_seqs != 0:
        label = "all" if args.vis_max_seqs < 0 else f"first {args.vis_max_seqs}"
        print(f"Visualization: {label} seqs per split -> {args.vis_dir}/")

    print(
        f"Homography refinement: "
        f"ransac_thresh={args.ransac_thresh}px  "
        f"min_inliers={args.min_h_inliers}"
    )

    # Load models once — reused across both splits
    print(f"\nLoading vanilla model   : {args.vanilla_ckpt}")
    vanilla_model = load_model(args.vanilla_ckpt, device=device)

    print(f"Loading fine-tuned model: {args.finetuned_ckpt}")
    ft_model = load_model(
        args.finetuned_ckpt,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_target_modules,
        device=device,
    )

    models = {"vanilla": vanilla_model, "finetuned": ft_model}

    all_results = {}
    for split_name, seqs in splits:
        all_results[split_name] = eval_split(
            split_name, seqs, args.dataset_dir, models,
            args.track_chunk, args.vis_thresh, device, dtype,
            vis_max_seqs=args.vis_max_seqs,
            vis_n_tracks=args.vis_n_tracks,
            vis_fps=args.vis_fps,
            vis_dir=args.vis_dir,
            ransac_thresh=args.ransac_thresh,
            min_h_inliers=args.min_h_inliers,
            homography_dir=args.homography_dir,
            ae_output_dir=args.ae_output_dir,
        )

    # Cross-split summary table
    sep = "=" * 70
    print(f"\n{sep}")
    split_names = list(all_results.keys())
    summary_note = "train vs val — check for overfitting" if set(split_names) >= {"train", "val"} else ", ".join(split_names)
    print(f"SUMMARY  ({summary_note})")
    print(sep)
    for tag in ("vanilla", "vanilla_h", "finetuned", "finetuned_h"):
        for split_name in split_names:
            key = f"{tag}_mean" if not tag.endswith("_h") else f"{tag}_mean"
            agg = all_results.get(split_name, {}).get(key)
            if agg is None:
                continue
            inlier_str = (
                f"  inlier={agg['inlier_ratio']:.2f}"
                if "inlier_ratio" in agg else ""
            )
            print(
                f"  {tag:12s} {split_name:5s}  "
                f"ATE={agg['ate']:.2f}px  "
                f"d5={agg['delta_5px']:.3f}  "
                f"vis={agg.get('vis_acc', float('nan')):.3f}"
                f"{inlier_str}"
            )

    with open(args.output_json, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
