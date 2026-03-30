#!/usr/bin/env python3
"""
eval_track_head_roi.py - Evaluate VGGT track head with ROI inference.

Same as eval_track_head.py, but instead of running on the full image the model
receives only a cropped region around the advertisement plane.  The ROI is
derived from ae_data/corners.csv (min/max of the 4 corner positions across all
frames in a sequence) and expanded by --roi-pad on every side.

Key differences vs eval_track_head.py:
  - Every sequence needs ae_data/corners.csv; sequences without it are skipped.
  - Images are cropped to the ROI before being passed to the model, giving it
    higher effective resolution on the plane at the cost of global context.
  - Query points and predicted tracks are converted between original and
    ROI-model coordinate spaces so all metrics stay in original image pixels.
  - Only the fine-tuned model is evaluated (EXP-06 checkpoint by default).
    Pass --run-vanilla to also evaluate the vanilla model for comparison.

Usage:
    python eval_track_head_roi.py \\
        --vanilla-ckpt     /workspace/model.pt \\
        --finetuned-ckpt   /workspace/ckpts/checkpoint.pt \\
        --dataset-dir      /mnt/bucket/.../tracking_whisper_sample_dataset \\
        --train-split-file /mnt/bucket/.../train_split.txt \\
        --val-split-file   /mnt/bucket/.../val_split.txt \\
        --roi-pad          0.3 \\
        [--run-vanilla] \\
        [--train-max-seqs 20] \\
        [--vis-max-seqs 3] [--vis-n-tracks 50] [--vis-fps 8] [--vis-dir roi_eval_viz] \\
        [--lora] [--lora-r 16] [--lora-alpha 32] \\
        [--track-chunk 256] \\
        [--ransac-thresh 3.0] [--min-h-inliers 8] \\
        [--output-json roi_eval_results.json]
"""

import argparse
import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as T_vis
from PIL import Image

from vggt.models.vggt import VGGT


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VGGT track head on plane ROI vs GT AE tracks"
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
        help="Train split: one sequence name per line.",
    )
    p.add_argument(
        "--val-split-file", default=None,
        help="Val split: one sequence name per line.",
    )
    p.add_argument(
        "--train-max-seqs", type=int, default=20,
        help="Randomly sample this many train sequences (-1 = all)",
    )
    p.add_argument(
        "--roi-pad", type=float, default=0.3,
        help=(
            "Expand the plane bounding box by this fraction on each side "
            "(e.g. 0.3 = 30%% padding around the tight bbox). Default: 0.3"
        ),
    )
    p.add_argument(
        "--run-vanilla", action="store_true", default=False,
        help="Also evaluate the vanilla model (disabled by default)",
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
        help="RANSAC reprojection threshold in pixels (default: 3.0)",
    )
    p.add_argument(
        "--min-h-inliers", type=int, default=8,
        help="Minimum RANSAC inliers to accept a homography (default: 8)",
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
        "--vis-dir", default="roi_eval_viz",
        help="Output directory for visualization images",
    )
    p.add_argument("--output-json", default="roi_eval_results.json")
    p.add_argument(
        "--ae-output-dir", default=None,
        help=(
            "If set, save per-sequence corner metric JSONs. "
            "Layout: <ae-output-dir>/<tag>/metrics/<seq>.json"
        ),
    )
    p.add_argument(
        "--homography-dir", default=None,
        help=(
            "If set, save estimated homographies. "
            "Layout: <homography-dir>/<split>/<seq>/<tag>_H.npy  (S, 3, 3)"
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path, lora=False, lora_r=16, lora_alpha=32.0,
               lora_targets="qkv,proj", device="cuda"):
    model = VGGT(enable_point=False)

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
        print(f"  LoRA applied (r={lora_r}, alpha={lora_alpha}, targets={lora_targets})")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys (expected for LoRA or partial ckpt)")

    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def get_model_resolution(image_path_or_size):
    """Return (W_orig, H_orig, W_model, H_model) for full-image crop mode."""
    if isinstance(image_path_or_size, (tuple, list)):
        W_orig, H_orig = int(image_path_or_size[0]), int(image_path_or_size[1])
    else:
        img = Image.open(image_path_or_size)
        W_orig, H_orig = img.size
    W_model = 518
    H_model = round(H_orig * (W_model / W_orig) / 14) * 14
    return W_orig, H_orig, W_model, H_model


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


# ---------------------------------------------------------------------------
# Sequence data loading (supports Format A images/ and Format B original.mp4)
# ---------------------------------------------------------------------------

def load_seq_data(seq_dir):
    import shutil, tempfile, json as _json

    ae_dir = os.path.join(seq_dir, "ae_data")
    img_dir = os.path.join(seq_dir, "images")
    video_path = os.path.join(seq_dir, "original.mp4")
    use_video = (
        not os.path.isdir(img_dir) or not os.listdir(img_dir)
    ) and os.path.isfile(video_path)

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

    tmp_dir = None

    if use_video:
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

        tmp_dir = tempfile.mkdtemp(prefix="vggt_roi_eval_")
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
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None, None, None, None, None, None, f"could not read frames from {video_path}"
    else:
        image_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        if not image_paths:
            return None, None, None, None, None, None, f"no images in {img_dir}"
        W_orig, H_orig, _, _ = get_model_resolution(image_paths[0])

    S = len(image_paths)
    if gt_tracks_orig.shape[0] < S:
        S = gt_tracks_orig.shape[0]
        image_paths = image_paths[:S]

    gt_tracks_orig = gt_tracks_orig[:S]
    gt_vis_mask    = gt_vis_mask[:S]

    if S < 2:
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, None, None, None, None, None, "need at least 2 frames"

    return image_paths, W_orig, H_orig, gt_tracks_orig, gt_vis_mask, tmp_dir, None


# ---------------------------------------------------------------------------
# ROI utilities
# ---------------------------------------------------------------------------

def load_roi_from_corners(seq_dir, pad_factor, W_orig, H_orig):
    """
    Compute plane ROI from ae_data/corners.csv.

    Returns (x1, y1, x2, y2) in original image pixels, or None if missing.
    The bbox is expanded by pad_factor (fraction of bbox size) on each side.
    """
    corners_path = os.path.join(seq_dir, "ae_data", "corners.csv")
    if not os.path.isfile(corners_path):
        return None
    data = np.loadtxt(corners_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis]
    # cols: frame, tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y
    xs = np.concatenate([data[:, 1], data[:, 3], data[:, 5], data[:, 7]])
    ys = np.concatenate([data[:, 2], data[:, 4], data[:, 6], data[:, 8]])
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    x1 = max(0.0,    xs.min() - w * pad_factor)
    y1 = max(0.0,    ys.min() - h * pad_factor)
    x2 = min(W_orig, xs.max() + w * pad_factor)
    y2 = min(H_orig, ys.max() + h * pad_factor)
    return (x1, y1, x2, y2)


def load_and_preprocess_images_roi(image_paths, roi):
    """
    Crop each image to roi and resize to VGGT-compatible resolution.

    W_model is always 518; H_model is rounded to the nearest multiple of 14
    to preserve the crop aspect ratio.

    Returns: (images_tensor, W_model, H_model)
        images_tensor: (S, 3, H_model, W_model) float32 [0, 1]
    """
    x1, y1, x2, y2 = (
        int(round(roi[0])), int(round(roi[1])),
        int(round(roi[2])), int(round(roi[3])),
    )
    crop_w = x2 - x1
    crop_h = y2 - y1
    target_w = 518
    target_h = round(crop_h * target_w / crop_w / 14) * 14
    to_tensor = T_vis.ToTensor()
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img = img.crop((x1, y1, x2, y2))
        img = img.resize((target_w, target_h), Image.Resampling.BICUBIC)
        images.append(to_tensor(img))
    return torch.stack(images), target_w, target_h


def roi_model_to_orig(coords, roi, W_model, H_model):
    """Map (..., 2) coords from ROI model space → original image pixels."""
    x1, y1, x2, y2 = roi
    sx = W_model / (x2 - x1)
    sy = H_model / (y2 - y1)
    out = coords.copy()
    out[..., 0] = coords[..., 0] / sx + x1
    out[..., 1] = coords[..., 1] / sy + y1
    return out


def roi_orig_to_model(coords, roi, W_model, H_model):
    """Map (..., 2) coords from original image pixels → ROI model space."""
    x1, y1, x2, y2 = roi
    sx = W_model / (x2 - x1)
    sy = H_model / (y2 - y1)
    out = coords.copy()
    out[..., 0] = (coords[..., 0] - x1) * sx
    out[..., 1] = (coords[..., 1] - y1) * sy
    return out


# ---------------------------------------------------------------------------
# Track head inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_track_head(model, images_tensor, query_points_np, chunk_size, device, dtype):
    N = query_points_np.shape[0]
    images_batch = images_tensor.unsqueeze(0).to(device)  # (1, S, 3, H, W)

    with torch.cuda.amp.autocast(dtype=dtype):
        agg_tokens, patch_start_idx = model.aggregator(images_batch)
        feature_maps = model.track_head.feature_extractor(
            agg_tokens, images_batch, patch_start_idx
        )

    all_tracks, all_vis = [], []
    query_tensor = torch.from_numpy(query_points_np)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        qpts = query_tensor[start:end].unsqueeze(0).to(device)

        with torch.cuda.amp.autocast(dtype=dtype):
            coord_preds, vis, _ = model.track_head.tracker(
                query_points=qpts,
                fmaps=feature_maps,
                iters=model.track_head.iters,
            )

        pred_t = coord_preds[-1].squeeze(0).cpu().float()
        pred_v = vis.squeeze(0).cpu().float()
        all_tracks.append(pred_t)
        all_vis.append(pred_v)

    pred_tracks = torch.cat(all_tracks, dim=1).numpy()
    pred_vis    = torch.cat(all_vis,    dim=1).numpy()
    return pred_tracks, pred_vis


# ---------------------------------------------------------------------------
# Homography refinement
# ---------------------------------------------------------------------------

def estimate_homography_refined_tracks(query_pts, pred_tracks,
                                       min_inliers=8, ransac_thresh=3.0):
    S, N, _ = pred_tracks.shape
    refined = pred_tracks.copy()
    inlier_ratios = np.ones(S, dtype=np.float32)

    src   = query_pts.astype(np.float32)
    src_h = np.concatenate([src, np.ones((N, 1), dtype=np.float32)], axis=1)
    homographies = np.stack([np.eye(3, dtype=np.float32)] * S)

    for t in range(1, S):
        dst = pred_tracks[t].astype(np.float32)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
        n_inliers = int(mask.sum()) if mask is not None else 0
        inlier_ratios[t] = n_inliers / N if N > 0 else 0.0

        if H is None or n_inliers < min_inliers:
            continue

        homographies[t] = H.astype(np.float32)
        proj  = (H @ src_h.T).T
        proj /= proj[:, 2:3]
        refined[t] = proj[:, :2]

    return refined, inlier_ratios, homographies


# ---------------------------------------------------------------------------
# GT corner loading + corner/centroid/jitter metrics
# ---------------------------------------------------------------------------

def load_gt_corners(seq_dir):
    csv_path = os.path.join(seq_dir, "ae_data", "corners.csv")
    if not os.path.isfile(csv_path):
        return None, None
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis]
    frame_indices = data[:, 0].astype(np.int32)
    corners = data[:, 1:].reshape(-1, 4, 2).astype(np.float32)
    return frame_indices, corners


def compute_corner_metrics(homographies, gt_frame_indices, gt_corners):
    frame0_mask = gt_frame_indices == 0
    if not frame0_mask.any():
        return None
    ref_corners = gt_corners[frame0_mask][0].astype(np.float64)
    ref_h = np.concatenate([ref_corners, np.ones((4, 1), dtype=np.float64)], axis=1)

    gt_lookup = {int(gt_frame_indices[i]): gt_corners[i]
                 for i in range(len(gt_frame_indices))}

    centroid_errors, corner_errors, pred_centroids = [], [], []

    for t in range(len(homographies)):
        if t not in gt_lookup:
            continue
        H    = homographies[t].astype(np.float64)
        gt_c = gt_lookup[t].astype(np.float64)

        proj  = (H @ ref_h.T).T
        proj /= proj[:, 2:3]
        pred_c = proj[:, :2]

        corner_errors.append(float(np.linalg.norm(pred_c - gt_c, axis=1).mean()))
        pred_cen = pred_c.mean(axis=0)
        gt_cen   = gt_c.mean(axis=0)
        centroid_errors.append(float(np.linalg.norm(pred_cen - gt_cen)))
        pred_centroids.append(pred_cen)

    if not centroid_errors:
        return None

    pt  = np.array(centroid_errors)
    ce  = np.array(corner_errors)
    ctr = np.array(pred_centroids)

    jitter = 0.0
    if len(ctr) >= 4:
        jerk   = np.diff(ctr, n=3, axis=0)
        jitter = float(np.sqrt((jerk ** 2).sum(axis=1).mean()))

    return {
        "pt_mean_px":       float(pt.mean()),
        "pt_median_px":     float(np.median(pt)),
        "pt_max_px":        float(pt.max()),
        "pt_p95_px":        float(np.percentile(pt, 95)),
        "corner_mean_px":   float(ce.mean()),
        "corner_median_px": float(np.median(ce)),
        "corner_max_px":    float(ce.max()),
        "corner_p95_px":    float(np.percentile(ce, 95)),
        "jitter_score":     jitter,
    }


def project_corners_through_homographies(ref_corners, homographies):
    S = len(homographies)
    ref_h = np.concatenate(
        [ref_corners.astype(np.float64), np.ones((4, 1), dtype=np.float64)], axis=1
    )
    pred = np.zeros((S, 4, 2), dtype=np.float32)
    for t in range(S):
        proj  = (homographies[t].astype(np.float64) @ ref_h.T).T
        proj /= proj[:, 2:3]
        pred[t] = proj[:, :2].astype(np.float32)
    return pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_tracks_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh):
    gt_t   = gt_tracks_orig[1:]
    pred_t = pred_tracks_orig[1:]
    gt_mask = gt_vis_mask[1:]
    pred_v  = pred_vis[1:]

    diff = pred_t - gt_t
    l2   = np.sqrt((diff ** 2).sum(-1))
    valid_l2 = l2[gt_mask]

    if len(valid_l2) == 0:
        return None

    pred_vis_bool = pred_v > vis_thresh
    vis_acc = float((pred_vis_bool == gt_mask).mean())

    return {
        "ate":        float(valid_l2.mean()),
        "median_te":  float(np.median(valid_l2)),
        "delta_1px":  float((valid_l2 < 1.0).mean()),
        "delta_2px":  float((valid_l2 < 2.0).mean()),
        "delta_5px":  float((valid_l2 < 5.0).mean()),
        "delta_10px": float((valid_l2 < 10.0).mean()),
        "vis_acc":    vis_acc,
        "n_points":   int(len(valid_l2)),
        "n_tracks":   int(gt_vis_mask.shape[1]),
        "n_frames":   int(gt_vis_mask.shape[0]),
    }


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def eval_sequence(
    seq_name, dataset_dir, model, roi, chunk_size, vis_thresh, device, dtype,
    ransac_thresh=3.0, min_h_inliers=8,
    image_paths=None, W_orig=None, H_orig=None,
    gt_tracks_orig=None, gt_vis_mask=None,
):
    """
    Returns: (metrics, metrics_h, metrics_corners, mean_inlier_ratio, error, preds, preds_h)
    All track coordinates in metrics/preds are in ORIGINAL image pixels.
    """
    seq_dir = os.path.join(dataset_dir, seq_name)

    if image_paths is None or gt_tracks_orig is None:
        image_paths, W_orig, H_orig, gt_tracks_orig, gt_vis_mask, _tmp, err = \
            load_seq_data(seq_dir)
        if err:
            return None, None, None, 0.0, err, None, None

    S = len(image_paths)
    if S < 2:
        return None, None, None, 0.0, "need at least 2 frames", None, None

    if roi is None:
        return None, None, None, 0.0, "no corners.csv — ROI unavailable", None, None

    # Load images cropped to ROI
    try:
        images_tensor, W_model, H_model = load_and_preprocess_images_roi(image_paths, roi)
    except Exception as exc:
        return None, None, None, 0.0, f"ROI image load error: {exc}", None, None

    # Map query points from original coords → ROI model coords
    q_orig  = gt_tracks_orig[0].copy()
    q_model = roi_orig_to_model(q_orig, roi, W_model, H_model)

    try:
        import time
        _t0 = time.perf_counter()
        pred_model, pred_vis = run_track_head(
            model, images_tensor, q_model, chunk_size, device, dtype
        )
        runtime_s = time.perf_counter() - _t0
    except RuntimeError as exc:
        return None, None, None, 0.0, f"runtime error: {exc}", None, None

    # Map predictions back to original image coords
    pred_orig = roi_model_to_orig(pred_model, roi, W_model, H_model)

    metrics = compute_metrics(pred_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh)
    if metrics is not None:
        metrics["runtime_s"] = runtime_s

    pred_h_orig, inlier_ratios, homographies = estimate_homography_refined_tracks(
        q_orig, pred_orig, min_inliers=min_h_inliers, ransac_thresh=ransac_thresh,
    )
    metrics_h = compute_metrics(pred_h_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh)
    mean_inlier_ratio = float(inlier_ratios[1:].mean()) if S > 1 else 1.0
    if metrics_h is not None:
        metrics_h["inlier_ratio"] = mean_inlier_ratio

    gt_frame_indices, gt_corners = load_gt_corners(seq_dir)
    metrics_corners = None
    if gt_frame_indices is not None:
        metrics_corners = compute_corner_metrics(homographies, gt_frame_indices, gt_corners)

    return (
        metrics, metrics_h, metrics_corners, mean_inlier_ratio,
        None,
        (pred_orig, pred_vis),
        (pred_h_orig, pred_vis, homographies, inlier_ratios),
    )


# ---------------------------------------------------------------------------
# Visualization (identical logic to eval_track_head.py)
# ---------------------------------------------------------------------------

_GT_COLOR  = (0, 220, 0)
_PANEL_DEFS = [
    ("vanilla",     (220, 60,  60),  "Vanilla"),
    ("vanilla_h",   (255, 150, 50),  "Vanilla+H"),
    ("finetuned",   (60,  130, 255), "Finetuned ROI"),
    ("finetuned_h", (60,  210, 160), "Finetuned ROI+H"),
]


def _render_frame(image_path, fi, gt_tracks_orig, gt_vis_mask,
                  preds_by_tag, sampled_ids, sx, sy, W_disp, H_disp,
                  draw_tracks=True, draw_quads=False,
                  gt_frame_lookup=None, pred_corners_by_tag=None,
                  single_tag=None, gt_only=False):
    def load_bg(path):
        arr = np.array(Image.open(path).convert("RGB"))
        H_raw, W_raw = arr.shape[:2]
        if W_raw == W_disp and H_raw == H_disp:
            return arr
        out = np.zeros((H_disp, W_disp, 3), dtype=np.uint8)
        out[:min(H_raw, H_disp), :min(W_raw, W_disp)] = arr[:H_disp, :W_disp]
        return out

    bg = load_bg(image_path)
    _SHIFT = 4
    _S16   = 1 << _SHIFT
    dot_r  = 4
    font   = cv2.FONT_HERSHEY_SIMPLEX
    bar_h  = 20
    _QUAD_ORDER = [0, 1, 3, 2]

    if gt_only:
        panel = bg.copy()
        if draw_tracks:
            for tid in sampled_ids:
                if gt_vis_mask[fi, tid]:
                    gx_s = round(float(gt_tracks_orig[fi, tid, 0]) * sx * _S16)
                    gy_s = round(float(gt_tracks_orig[fi, tid, 1]) * sy * _S16)
                    cv2.circle(panel, (gx_s, gy_s), dot_r * _S16,
                               _GT_COLOR[::-1], -1, cv2.LINE_AA, _SHIFT)
        cv2.putText(panel, "GT", (8, 22), font, 0.7, _GT_COLOR[::-1], 2, cv2.LINE_AA)
        panel[-bar_h:] = (20, 20, 20)
        cv2.circle(panel[-bar_h:], (10, bar_h // 2), 5, _GT_COLOR[::-1], -1)
        cv2.putText(panel[-bar_h:], "GT", (18, bar_h - 5), font, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
        return panel

    def _scale_corners(corners_4x2):
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

        if draw_tracks:
            pred_tracks = preds_by_tag[tag][0]
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

        if draw_quads:
            if gt_frame_lookup is not None and fi in gt_frame_lookup:
                pts = _scale_corners(gt_frame_lookup[fi])
                cv2.polylines(panel, [pts], True, _GT_COLOR[::-1], 2, cv2.LINE_AA, _SHIFT)
            if pred_corners_by_tag is not None and tag in pred_corners_by_tag:
                pts = _scale_corners(pred_corners_by_tag[tag][fi])
                cv2.polylines(panel, [pts], True, color[::-1], 2, cv2.LINE_AA, _SHIFT)

        cv2.putText(panel, label, (8, 22), font, 0.7, color[::-1], 2, cv2.LINE_AA)
        cv2.putText(panel, f"frame {fi}", (8, H_disp - 8), font, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        panel[-bar_h:] = (20, 20, 20)
        cv2.circle(panel[-bar_h:], (10, bar_h // 2), 5, _GT_COLOR[::-1], -1)
        cv2.putText(panel[-bar_h:], "GT", (18, bar_h - 5), font, 0.4,
                    (220, 220, 220), 1, cv2.LINE_AA)
        lx = 18 + 3 * 7 + 10
        cv2.circle(panel[-bar_h:], (lx, bar_h // 2), 5, color[::-1], 2)
        cv2.putText(panel[-bar_h:], "Pred", (lx + 8, bar_h - 5), font, 0.4,
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
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%06d.jpg"),
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    [viz] ffmpeg failed (code {result.returncode}):\n{result.stderr[-400:]}")
        return False
    return True


def visualize_sequence(seq_name, image_paths, gt_tracks_orig, gt_vis_mask,
                       preds_by_tag, split_name, vis_dir, n_tracks, fps=8,
                       W_orig=None, H_orig=None, seq_dir=None):
    import tempfile, shutil

    S = len(image_paths)
    if S < 2 or not preds_by_tag:
        return

    valid_ids = np.where(gt_vis_mask[1])[0]
    if len(valid_ids) == 0:
        return

    rng = np.random.RandomState(42)
    sampled_ids = rng.choice(valid_ids, size=min(n_tracks, len(valid_ids)), replace=False)

    if W_orig is None or H_orig is None:
        W_orig, H_orig, _, _ = get_model_resolution(image_paths[0])
    W_disp = W_orig + (W_orig % 2)
    H_disp = H_orig + (H_orig % 2)
    sx = sy = 1.0

    if seq_dir is not None:
        fps = get_sequence_fps(seq_dir, fallback=fps)

    out_dir = os.path.join(vis_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    def _save_video(render_kwargs, suffix):
        out_path = os.path.join(out_dir, f"{seq_name}_{suffix}.mp4")
        tmp_dir = tempfile.mkdtemp(prefix="roi_viz_")
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
                print(f"    [viz] -> {out_path}  ({S} frames @ {fps:.1f}fps)")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    _save_video({}, "tracks_composed")
    for tag in [t for t in preds_by_tag if not t.endswith("_h")]:
        _save_video({"single_tag": tag}, f"tracks_{tag}")


def visualize_planes(
    seq_name, image_paths, gt_tracks_orig, gt_vis_mask,
    preds_by_tag, split_name, vis_dir, n_tracks, fps=8,
    gt_frame_lookup=None, pred_corners_by_tag=None,
    W_orig=None, H_orig=None, seq_dir=None,
):
    """
    Save plane-quad comparison videos: GT + per-model + composed.
    Skipped silently if no GT corners are available (corners.csv missing).
    """
    import tempfile, shutil

    if gt_frame_lookup is None or pred_corners_by_tag is None:
        return

    S = len(image_paths)
    if S < 2 or not preds_by_tag:
        return

    valid_ids = np.where(gt_vis_mask[1])[0]
    if len(valid_ids) == 0:
        return

    rng = np.random.RandomState(42)
    sampled_ids = rng.choice(valid_ids, size=min(n_tracks, len(valid_ids)), replace=False)

    if W_orig is None or H_orig is None:
        W_orig, H_orig, _, _ = get_model_resolution(image_paths[0])
    W_disp = W_orig + (W_orig % 2)
    H_disp = H_orig + (H_orig % 2)
    sx = sy = 1.0

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
    split_name, sequences, dataset_dir, models, roi_pad,
    chunk_size, vis_thresh, device, dtype,
    vis_max_seqs=3, vis_n_tracks=50, vis_fps=8, vis_dir="roi_eval_viz",
    ransac_thresh=3.0, min_h_inliers=8,
    homography_dir=None, ae_output_dir=None,
):
    results         = {tag: {} for tag in models}
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

        import shutil as _shutil
        image_paths_seq, W_orig_seq, H_orig_seq, gt_tracks_seq, gt_vis_seq, tmp_dir_seq, load_err = \
            load_seq_data(seq_dir)

        if load_err:
            print(f"    SKIP ({load_err})")
            continue

        # Compute ROI once per sequence (shared across models)
        roi = load_roi_from_corners(seq_dir, roi_pad, W_orig_seq, H_orig_seq)
        if roi is None:
            print(f"    SKIP (no corners.csv — ROI required)")
            if tmp_dir_seq:
                _shutil.rmtree(tmp_dir_seq, ignore_errors=True)
            continue
        print(f"    ROI: ({roi[0]:.0f},{roi[1]:.0f}) - ({roi[2]:.0f},{roi[3]:.0f})"
              f"  size: {roi[2]-roi[0]:.0f}x{roi[3]-roi[1]:.0f}  pad={roi_pad}")

        gt_frame_indices_seq, gt_corners_seq = load_gt_corners(seq_dir)
        gt_frame_lookup = (
            {int(gt_frame_indices_seq[i]): gt_corners_seq[i]
             for i in range(len(gt_frame_indices_seq))}
            if gt_frame_indices_seq is not None else None
        )

        preds_by_tag      = {}
        pred_corners_by_tag = {}

        for tag, model in models.items():
            metrics, metrics_h, metrics_corners, mean_inlier, err, preds, preds_h = eval_sequence(
                seq, dataset_dir, model, roi,
                chunk_size, vis_thresh, device, dtype,
                ransac_thresh=ransac_thresh,
                min_h_inliers=min_h_inliers,
                image_paths=image_paths_seq,
                W_orig=W_orig_seq, H_orig=H_orig_seq,
                gt_tracks_orig=gt_tracks_seq, gt_vis_mask=gt_vis_seq,
            )
            if err:
                print(f"    {tag:12s}  SKIP ({err})")
                continue

            results[tag][seq] = metrics
            if metrics_h is not None:
                results[f"{tag}_h"][seq] = metrics_h

            if preds is not None:
                preds_by_tag[tag] = preds
            if preds_h is not None:
                pred_h_orig, pred_vis_h, homographies, _ = preds_h
                preds_by_tag[f"{tag}_h"] = (pred_h_orig, pred_vis_h)

                if gt_frame_lookup is not None and gt_frame_indices_seq is not None:
                    frame0_mask = gt_frame_indices_seq == 0
                    if frame0_mask.any():
                        ref_corners = gt_corners_seq[frame0_mask][0]
                        pc = project_corners_through_homographies(ref_corners, homographies)
                        pred_corners_by_tag[tag]        = pc
                        pred_corners_by_tag[f"{tag}_h"] = pc

            if homography_dir is not None and preds_h is not None:
                _, _, homographies, inlier_ratios = preds_h
                h_seq_dir = os.path.join(homography_dir, split_name, seq)
                os.makedirs(h_seq_dir, exist_ok=True)
                np.save(os.path.join(h_seq_dir, f"{tag}_H.npy"), homographies)
                np.save(os.path.join(h_seq_dir, f"{tag}_inliers.npy"), inlier_ratios)

            print(
                f"    {tag:12s}  "
                f"ATE={metrics['ate']:6.2f}px  "
                f"med={metrics['median_te']:6.2f}px  "
                f"d1={metrics['delta_1px']:.3f}  "
                f"d2={metrics['delta_2px']:.3f}  "
                f"d5={metrics['delta_5px']:.3f}  "
                f"vis={metrics['vis_acc']:.3f}  "
                f"N={metrics['n_points']}  "
                f"t={metrics.get('runtime_s', 0.0):.2f}s"
            )
            if metrics_h is not None:
                print(
                    f"    {'':12s}  "
                    f"[+H] ATE={metrics_h['ate']:6.2f}px  "
                    f"med={metrics_h['median_te']:6.2f}px  "
                    f"d5={metrics_h['delta_5px']:.3f}  "
                    f"inlier={mean_inlier:.2f}  "
                    f"ΔATE={metrics_h['ate']-metrics['ate']:+.2f}px"
                )
            if metrics_corners is not None:
                results_corners[tag][seq] = metrics_corners
                print(
                    f"    {'':12s}  "
                    f"[corners] "
                    f"pt={metrics_corners['pt_mean_px']:.2f}px  "
                    f"corner={metrics_corners['corner_mean_px']:.2f}px  "
                    f"jitter={metrics_corners['jitter_score']:.3f}"
                )

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

        if (
            vis_max_seqs != 0
            and (vis_max_seqs < 0 or viz_count < vis_max_seqs)
            and preds_by_tag and gt_tracks_seq is not None and image_paths_seq
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

        if tmp_dir_seq is not None:
            _shutil.rmtree(tmp_dir_seq, ignore_errors=True)

    # Aggregate stats
    print(f"\n  --- {split_name.upper()} aggregate ---")
    agg_by_tag = {}
    for tag in models:
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
                f"d5={agg['delta_5px']:.3f}  "
                f"vis={agg['vis_acc']:.3f}"
            )

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
                f"d5={agg_h['delta_5px']:.3f}"
                f"{inlier_str}"
            )

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
    if "vanilla" in agg_by_tag and "finetuned" in agg_by_tag:
        for tag_a, tag_b, label in [
            ("vanilla",   "finetuned",   "vanilla -> finetuned ROI"),
            ("vanilla",   "vanilla_h",   "vanilla -> vanilla+H"),
            ("finetuned", "finetuned_h", "finetuned -> finetuned+H"),
        ]:
            if tag_a not in agg_by_tag or tag_b not in agg_by_tag:
                continue
            a, b = agg_by_tag[tag_a], agg_by_tag[tag_b]
            ate_d = b["ate"] - a["ate"]
            d5_d  = b["delta_5px"] - a["delta_5px"]
            pct   = ate_d / a["ate"] * 100
            print(f"\n  IMPROVEMENT {label}:")
            print(f"    ATE: {a['ate']:.2f} -> {b['ate']:.2f} px  ({ate_d:+.2f}px, {pct:+.1f}%)")
            print(f"    d5:  {a['delta_5px']:.3f} -> {b['delta_5px']:.3f}  ({d5_d*100:+.1f}pp)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    capable = torch.cuda.get_device_capability()[0] >= 8
    dtype  = torch.bfloat16 if capable else torch.float16

    def load_sequences(path):
        with open(path) as fh:
            return [ln.strip() for ln in fh if ln.strip()]

    if args.train_split_file or args.val_split_file:
        train_seqs = load_sequences(args.train_split_file) if args.train_split_file else []
        val_seqs   = load_sequences(args.val_split_file)   if args.val_split_file   else []

        n_total = len(train_seqs)
        max_tr  = args.train_max_seqs
        if max_tr > 0 and n_total > max_tr:
            train_seqs = random.sample(train_seqs, max_tr)
            print(f"Train split: sampled {max_tr} / {n_total} sequences")
        else:
            print(f"Train split: {n_total} sequences")
        print(f"Val   split: {len(val_seqs)} sequences")
        splits = [("train", train_seqs), ("val", val_seqs)]
    else:
        raise ValueError("Provide --train-split-file and/or --val-split-file")

    print(f"ROI pad: {args.roi_pad}  (sequences without corners.csv will be skipped)")
    print(
        f"Homography refinement: "
        f"ransac_thresh={args.ransac_thresh}px  min_inliers={args.min_h_inliers}"
    )

    print(f"\nLoading fine-tuned model: {args.finetuned_ckpt}")
    ft_model = load_model(
        args.finetuned_ckpt,
        lora=args.lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_targets=args.lora_target_modules, device=device,
    )
    models = {"finetuned": ft_model}

    if args.run_vanilla:
        print(f"Loading vanilla model: {args.vanilla_ckpt}")
        vanilla_model = load_model(args.vanilla_ckpt, device=device)
        models["vanilla"] = vanilla_model

    all_results = {}
    for split_name, seqs in splits:
        all_results[split_name] = eval_split(
            split_name, seqs, args.dataset_dir, models, args.roi_pad,
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

    sep = "=" * 70
    print(f"\n{sep}")
    print("SUMMARY")
    print(sep)
    for tag in ("finetuned", "finetuned_h", "vanilla", "vanilla_h"):
        for split_name in all_results:
            agg = all_results[split_name].get(f"{tag}_mean")
            if agg is None:
                continue
            inlier_str = (
                f"  inlier={agg['inlier_ratio']:.2f}"
                if "inlier_ratio" in agg else ""
            )
            print(
                f"  {tag:14s} {split_name:5s}  "
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
