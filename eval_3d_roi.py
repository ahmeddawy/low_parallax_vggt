#!/usr/bin/env python3
"""
eval_3d_roi.py - 3D reprojection evaluation with plane-ROI inference.

Same protocol as eval_3d.py, but the model receives only the cropped region
around the advertisement plane (derived from ae_data/corners.csv) instead of
the full image.  The hypothesis is that higher effective resolution on the
plane improves depth and camera predictions for that region.

Key differences vs eval_3d.py:
  - Requires ae_data/corners.csv per sequence; sequences without it are skipped.
  - Images are cropped to the ROI and resized to 518 x H_roi (VGGT-compatible).
  - Depth/camera inference runs on the cropped view.
  - Reprojection errors are computed and reported in ORIGINAL image pixels so
    metrics are directly comparable to eval_3d.py results.
  - COLMAP cameras are written at full (W_orig x H_orig) resolution with the
    principal point shifted by the ROI offset, keeping 3D consistent.
  - Only the fine-tuned model is run by default; add --run-vanilla to compare.

Usage:
    python eval_3d_roi.py \\
        --vanilla-ckpt     /workspace/model.pt \\
        --finetuned-ckpt   /workspace/ckpts/checkpoint.pt \\
        --dataset-dir      /mnt/.../dataset \\
        --val-split-file   /mnt/.../val_split.txt \\
        --roi-pad          0.3 \\
        [--run-vanilla] \\
        [--run-ba] [--ba-max-iter 100] \\
        [--max-frames 230] \\
        [--colmap-dir roi_colmap_out] \\
        [--vis-dir roi_3d_viz] [--vis-max-seqs 3] \\
        [--output-json roi_3d_results.json]
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import torch
import torchvision.transforms as T_vis
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="3D reprojection evaluation of VGGT on plane ROI"
    )
    p.add_argument("--vanilla-ckpt", required=True)
    p.add_argument("--finetuned-ckpt", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--train-split-file", default=None)
    p.add_argument("--val-split-file", required=True)
    p.add_argument(
        "--train-max-seqs", type=int, default=20,
        help="Randomly sample this many train sequences (-1 = all)",
    )
    p.add_argument(
        "--roi-pad", type=float, default=0.3,
        help=(
            "Expand the plane bounding box by this fraction on each side. "
            "Default: 0.3"
        ),
    )
    p.add_argument(
        "--run-vanilla", action="store_true", default=False,
        help="Also evaluate the vanilla model (disabled by default)",
    )
    p.add_argument("--lora", action="store_true", default=False)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-target-modules", default="qkv,proj")
    p.add_argument(
        "--track-num", type=int, default=256,
        help="Max GT tracks per sequence (random sample)",
    )
    p.add_argument("--colmap-dir", default=None)
    p.add_argument(
        "--run-ba", action="store_true", default=False,
        help="Run COLMAP bundle adjustment on predicted cameras",
    )
    p.add_argument("--ba-colmap-bin", default="colmap")
    p.add_argument("--ba-max-iter", type=int, default=100)
    p.add_argument("--vis-max-seqs", type=int, default=3)
    p.add_argument("--vis-n-tracks", type=int, default=50)
    p.add_argument("--vis-fps", type=int, default=8)
    p.add_argument("--vis-dir", default="roi_3d_viz")
    p.add_argument("--output-json", default="roi_3d_results.json")
    p.add_argument(
        "--max-frames", type=int, default=0,
        help="Cap sequences to this many frames (evenly subsampled). 0 = no limit.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path, lora=False, lora_r=16, lora_alpha=32.0,
               lora_targets="qkv,proj", device="cuda", enable_track=False):
    model = VGGT(enable_point=False, enable_track=enable_track)

    if lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=lora_targets.split(","),
            lora_dropout=0.0, bias="none",
        )
        model.aggregator = get_peft_model(model.aggregator, lora_config)
        print(f"  LoRA applied (r={lora_r}, alpha={lora_alpha})")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys (expected for LoRA/partial ckpt)")

    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def get_model_resolution(image_path):
    img = Image.open(image_path)
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
# ROI utilities
# ---------------------------------------------------------------------------

def load_roi_from_corners(seq_dir, pad_factor, W_orig, H_orig):
    """
    Returns (x1, y1, x2, y2) in original image pixels, or None if missing.
    """
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
    x1 = max(0.0,    xs.min() - w * pad_factor)
    y1 = max(0.0,    ys.min() - h * pad_factor)
    x2 = min(W_orig, xs.max() + w * pad_factor)
    y2 = min(H_orig, ys.max() + h * pad_factor)
    return (x1, y1, x2, y2)


def load_and_preprocess_images_roi(image_paths, roi):
    """
    Crop each image to roi and resize to 518 x H_model (nearest multiple of 14).

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


def intrinsics_roi_to_full(intrinsics, roi, W_model, H_model):
    """
    Convert intrinsics from ROI model space → full original image space.

    The focal lengths scale by crop_size / W_model (inverse of the ROI
    preprocessing scale).  The principal point also shifts by the ROI offset.

    intrinsics: (S, 3, 3) at ROI model resolution
    Returns:    (S, 3, 3) at full original image resolution
    """
    x1, y1, x2, y2 = roi
    crop_w = x2 - x1
    crop_h = y2 - y1
    sx = W_model / crop_w   # ROI model → ROI original
    sy = H_model / crop_h
    out = intrinsics.copy()
    # fx, fy
    out[:, 0, 0] = intrinsics[:, 0, 0] / sx
    out[:, 1, 1] = intrinsics[:, 1, 1] / sy
    # cx, cy: ROI-relative original + ROI offset = full image original
    out[:, 0, 2] = intrinsics[:, 0, 2] / sx + x1
    out[:, 1, 2] = intrinsics[:, 1, 2] / sy + y1
    return out


# ---------------------------------------------------------------------------
# Camera + depth inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_camera_and_depth(model, images_tensor, H_model, W_model, device, dtype):
    images_batch = images_tensor.unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(dtype=dtype):
        agg_tokens, patch_start_idx = model.aggregator(images_batch)

    with torch.cuda.amp.autocast(enabled=False):
        pose_enc_list = model.camera_head(agg_tokens)
        depth, _ = model.depth_head(
            agg_tokens, images=images_batch, patch_start_idx=patch_start_idx,
        )

    pose_enc = pose_enc_list[-1]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(H_model, W_model)
    )
    extrinsics = extrinsics.squeeze(0).float().cpu().numpy()
    intrinsics = intrinsics.squeeze(0).float().cpu().numpy()
    pred_depth  = depth.squeeze(0).squeeze(-1).float().cpu().numpy()
    return extrinsics, intrinsics, pred_depth


# ---------------------------------------------------------------------------
# 3D reprojection (ROI-aware)
# ---------------------------------------------------------------------------

def compute_reproj_metrics_roi(
    gt_tracks_orig, gt_vis_mask,
    extrinsics, intrinsics_roi,
    pred_depth, roi, W_model, H_model,
    track_num=256,
):
    """
    Same as compute_reproj_metrics in eval_3d.py but coordinates go through
    the ROI coordinate system.

    Inputs:
      gt_tracks_orig   : (S, N, 2) float32  GT tracks in ORIGINAL image pixels
      extrinsics       : (S, 3, 4)          world-to-cam [R|t] (ROI model space)
      intrinsics_roi   : (S, 3, 3)          K at ROI model resolution
      pred_depth       : (S, H_model, W_model) depth in ROI model space
      roi              : (x1, y1, x2, y2)  in original image pixels

    All reported errors are in ORIGINAL image pixels.
    """
    x1, y1, x2, y2 = roi
    sx = W_model / (x2 - x1)   # original-ROI → model
    sy = H_model / (y2 - y1)

    vis0 = gt_vis_mask[0]
    valid_ids = np.where(vis0)[0]
    rng = np.random.RandomState(42)
    if len(valid_ids) > track_num:
        valid_ids = rng.choice(valid_ids, size=track_num, replace=False)
    N_use = len(valid_ids)
    if N_use == 0:
        return None, None, None, None, None

    d0 = pred_depth[0]
    K0 = intrinsics_roi[0]
    R0 = extrinsics[0, :3, :3]
    T0 = extrinsics[0, :3, 3]

    # Map GT frame-0 points from original to ROI model space
    u0 = (gt_tracks_orig[0, valid_ids, 0] - x1) * sx
    v0 = (gt_tracks_orig[0, valid_ids, 1] - y1) * sy

    u0_c = np.clip(u0, 0, W_model - 1)
    v0_c = np.clip(v0, 0, H_model - 1)
    u0i  = np.floor(u0_c).astype(int)
    v0i  = np.floor(v0_c).astype(int)
    d_val = d0[v0i, u0i]

    depth_valid = d_val > 0.0

    fx0, fy0 = K0[0, 0], K0[1, 1]
    cx0, cy0 = K0[0, 2], K0[1, 2]
    X_cam0 = (u0 - cx0) / fx0 * d_val
    Y_cam0 = (v0 - cy0) / fy0 * d_val
    Z_cam0 = d_val
    P_cam0 = np.stack([X_cam0, Y_cam0, Z_cam0], axis=-1)
    P_world = (P_cam0 - T0[None, :]) @ R0

    S = gt_tracks_orig.shape[0]
    all_errors = []
    reproj_tracks_orig = np.zeros((S, N_use, 2), dtype=np.float32)
    reproj_tracks_orig[0] = gt_tracks_orig[0, valid_ids, :]

    for j in range(1, S):
        R_j = extrinsics[j, :3, :3]
        T_j = extrinsics[j, :3, 3]
        K_j = intrinsics_roi[j]

        P_camj   = P_world @ R_j.T + T_j[None, :]
        in_front = P_camj[:, 2] > 0.0

        fx_j, fy_j = K_j[0, 0], K_j[1, 1]
        cx_j, cy_j = K_j[0, 2], K_j[1, 2]
        denom = np.maximum(P_camj[:, 2], 1e-6)
        u_j   = P_camj[:, 0] / denom * fx_j + cx_j   # ROI model coords
        v_j   = P_camj[:, 1] / denom * fy_j + cy_j

        # Map back to original image coords for metric computation
        u_j_orig = u_j / sx + x1
        v_j_orig = v_j / sy + y1

        reproj_tracks_orig[j, :, 0] = u_j_orig
        reproj_tracks_orig[j, :, 1] = v_j_orig

        gt_u   = gt_tracks_orig[j, valid_ids, 0]
        gt_v   = gt_tracks_orig[j, valid_ids, 1]
        gt_vis_j = gt_vis_mask[j, valid_ids]

        mask = gt_vis_j & depth_valid & in_front
        err  = np.sqrt((u_j_orig - gt_u) ** 2 + (v_j_orig - gt_v) ** 2)
        all_errors.append(err[mask])

    errors = np.concatenate(all_errors) if all_errors else np.array([])
    if len(errors) == 0:
        return None, None, None, None, None

    metrics = {
        "reproj_ate":  float(errors.mean()),
        "reproj_median": float(np.median(errors)),
        "delta_1px":   float((errors < 1.0).mean()),
        "delta_2px":   float((errors < 2.0).mean()),
        "delta_5px":   float((errors < 5.0).mean()),
        "delta_10px":  float((errors < 10.0).mean()),
        "n_points":    int(N_use),
        "n_valid_depth": int(depth_valid.sum()),
    }
    return metrics, P_world, depth_valid, reproj_tracks_orig, valid_ids


# ---------------------------------------------------------------------------
# COLMAP export (full-image space for compatibility)
# ---------------------------------------------------------------------------

def _rot_to_colmap_quat(R):
    try:
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])
    except ImportError:
        pass
    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s; x = (m[2,1]-m[1,2])*s; y = (m[0,2]-m[2,0])*s; z = (m[1,0]-m[0,1])*s
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = 2.0 * np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
        w = (m[2,1]-m[1,2])/s; x = 0.25*s; y = (m[0,1]+m[1,0])/s; z = (m[0,2]+m[2,0])/s
    elif m[1,1] > m[2,2]:
        s = 2.0 * np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
        w = (m[0,2]-m[2,0])/s; x = (m[0,1]+m[1,0])/s; y = 0.25*s; z = (m[1,2]+m[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
        w = (m[1,0]-m[0,1])/s; x = (m[0,2]+m[2,0])/s; y = (m[1,2]+m[2,1])/s; z = 0.25*s
    return np.array([w, x, y, z])


def write_colmap(
    out_dir, image_paths,
    extrinsics, intrinsics_full,
    W_orig, H_orig,
    pts3d_world=None, depth_valid=None,
    gt_tracks_orig=None, gt_vis_mask=None,
):
    """Write COLMAP sparse model. intrinsics_full must already be in full-image space."""
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    S = len(image_paths)

    # Average intrinsics across frames (single camera model)
    fx_orig = float(intrinsics_full[:, 0, 0].mean())
    fy_orig = float(intrinsics_full[:, 1, 1].mean())
    cx_orig = float(intrinsics_full[:, 0, 2].mean())
    cy_orig = float(intrinsics_full[:, 1, 2].mean())

    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as fh:
        fh.write("# Camera list\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fh.write(
            f"1 PINHOLE {W_orig} {H_orig} "
            f"{fx_orig:.6f} {fy_orig:.6f} {cx_orig:.6f} {cy_orig:.6f}\n"
        )

    have_pts = (pts3d_world is not None and depth_valid is not None
                and gt_tracks_orig is not None)
    if have_pts:
        valid_pt_idx = np.where(depth_valid)[0]
        pt3d_id_map = {int(i): int(k + 1) for k, i in enumerate(valid_pt_idx)}
    else:
        valid_pt_idx = np.array([], dtype=int)
        pt3d_id_map = {}

    with open(os.path.join(sparse_dir, "images.txt"), "w") as fh:
        fh.write("# Image list\n#   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, img_path in enumerate(image_paths):
            R = extrinsics[i, :3, :3]; T = extrinsics[i, :3, 3]
            qw, qx, qy, qz = _rot_to_colmap_quat(R)
            fh.write(
                f"{i+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{T[0]:.9f} {T[1]:.9f} {T[2]:.9f} 1 {os.path.basename(img_path)}\n"
            )
            if have_pts and gt_vis_mask is not None:
                parts = []
                for k, pt_idx in enumerate(valid_pt_idx):
                    if gt_vis_mask[i, pt_idx]:
                        u = float(gt_tracks_orig[i, pt_idx, 0])
                        v = float(gt_tracks_orig[i, pt_idx, 1])
                        parts.append(f"{u:.3f} {v:.3f} {pt3d_id_map[int(pt_idx)]}")
                fh.write(" ".join(parts) + "\n" if parts else "\n")
            else:
                fh.write("\n")

    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as fh:
        fh.write("# 3D point list\n#   POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        if have_pts:
            for k, pt_idx in enumerate(valid_pt_idx):
                X, Y, Z = pts3d_world[pt_idx]
                track_parts = []
                if gt_vis_mask is not None:
                    for i in range(S):
                        if gt_vis_mask[i, pt_idx]:
                            track_parts.append(f"{i+1} {k}")
                fh.write(
                    f"{pt3d_id_map[int(pt_idx)]} {X:.6f} {Y:.6f} {Z:.6f} "
                    f"200 200 0 0.0 {' '.join(track_parts)}\n"
                )
    print(f"    [colmap] -> {sparse_dir} ({S} images, {len(valid_pt_idx)} 3D pts)")


# ---------------------------------------------------------------------------
# BA helpers (same as eval_3d.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_track_head_ba(model, images_tensor, query_pts_model, device, dtype, chunk_size=128):
    N = query_pts_model.shape[0]
    images_batch = images_tensor.unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(dtype=dtype):
        agg_tokens, patch_start_idx = model.aggregator(images_batch)
        feature_maps = model.track_head.feature_extractor(
            agg_tokens, images_batch, patch_start_idx
        )
    all_tracks, all_vis = [], []
    query_tensor = torch.from_numpy(query_pts_model)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        qpts = query_tensor[start:end].unsqueeze(0).to(device)
        with torch.cuda.amp.autocast(dtype=dtype):
            coord_preds, vis, _ = model.track_head.tracker(
                query_points=qpts, fmaps=feature_maps, iters=model.track_head.iters,
            )
        all_tracks.append(coord_preds[-1].squeeze(0).cpu().float())
        all_vis.append(vis.squeeze(0).cpu().float())
    pred_tracks = torch.cat(all_tracks, dim=1).numpy()
    pred_vis    = torch.cat(all_vis,    dim=1).numpy()
    return pred_tracks, pred_vis


def write_colmap_from_pred_tracks(
    out_dir, image_paths,
    extrinsics, intrinsics_full,
    W_orig, H_orig,
    pts3d_world, depth_valid,
    pred_tracks_roi, pred_vis,
    roi, W_model, H_model,
    vis_thresh=0.5,
):
    """
    Write COLMAP sparse model using predicted track positions as 2D observations.
    pred_tracks_roi: (S, N, 2) float32 in ROI model coords.
    Observations are converted to full-image original coords.
    """
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)
    S = len(image_paths)

    x1, y1, x2, y2 = roi
    sx_roi = W_model / (x2 - x1)
    sy_roi = H_model / (y2 - y1)

    # Per-frame intrinsics in full-image space
    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as fh:
        fh.write("# Camera list\n#   CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        for i in range(S):
            K = intrinsics_full[i]
            fh.write(
                f"{i+1} PINHOLE {W_orig} {H_orig} "
                f"{K[0,0]:.6f} {K[1,1]:.6f} {K[0,2]:.6f} {K[1,2]:.6f}\n"
            )

    valid_pt_idx = np.where(depth_valid)[0]
    pt3d_id_map  = {int(ii): int(k + 1) for k, ii in enumerate(valid_pt_idx)}

    with open(os.path.join(sparse_dir, "images.txt"), "w") as fh:
        fh.write("# Image list\n#   IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n")
        fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, img_path in enumerate(image_paths):
            R = extrinsics[i, :3, :3]; T = extrinsics[i, :3, 3]
            qw, qx, qy, qz = _rot_to_colmap_quat(R)
            fh.write(
                f"{i+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{T[0]:.9f} {T[1]:.9f} {T[2]:.9f} {i+1} {os.path.basename(img_path)}\n"
            )
            parts = []
            for k, pt_idx in enumerate(valid_pt_idx):
                if float(pred_vis[i, pt_idx]) > vis_thresh:
                    # Convert ROI model → full image original
                    u_orig = float(pred_tracks_roi[i, pt_idx, 0]) / sx_roi + x1
                    v_orig = float(pred_tracks_roi[i, pt_idx, 1]) / sy_roi + y1
                    parts.append(f"{u_orig:.3f} {v_orig:.3f} {pt3d_id_map[int(pt_idx)]}")
            fh.write(" ".join(parts) + "\n" if parts else "\n")

    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as fh:
        fh.write("# 3D point list\n#   POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for k, pt_idx in enumerate(valid_pt_idx):
            X, Y, Z = pts3d_world[pt_idx]
            track_parts = [
                f"{i+1} {k}" for i in range(S)
                if float(pred_vis[i, pt_idx]) > vis_thresh
            ]
            if track_parts:
                fh.write(
                    f"{pt3d_id_map[int(pt_idx)]} {X:.6f} {Y:.6f} {Z:.6f} "
                    f"200 200 0 0.0 {' '.join(track_parts)}\n"
                )
    print(f"    [ba-colmap] -> {sparse_dir} ({S} images, {len(valid_pt_idx)} pts)")
    return sparse_dir


def run_bundle_adjustment(sparse_in, work_dir, colmap_bin="colmap", max_iter=100):
    ba_bin_dir = os.path.join(work_dir, "ba_bin")
    ba_txt_dir = os.path.join(work_dir, "ba_txt")
    os.makedirs(ba_bin_dir, exist_ok=True)
    os.makedirs(ba_txt_dir, exist_ok=True)

    r = subprocess.run([
        colmap_bin, "bundle_adjuster",
        "--input_path", sparse_in,
        "--output_path", ba_bin_dir,
        "--BundleAdjustment.max_num_iterations", str(max_iter),
        "--BundleAdjustment.refine_focal_length", "1",
        "--BundleAdjustment.refine_principal_point", "0",
        "--BundleAdjustment.refine_extra_params", "0",
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [BA] bundle_adjuster failed:\n{r.stderr[-400:]}")
        return None

    r2 = subprocess.run([
        colmap_bin, "model_converter",
        "--input_path", ba_bin_dir,
        "--output_path", ba_txt_dir,
        "--output_type", "TXT",
    ], capture_output=True, text=True)
    if r2.returncode != 0:
        print(f"    [BA] model_converter failed:\n{r2.stderr[-400:]}")
        return None

    print(f"    [BA] refined model -> {ba_txt_dir}")
    return ba_txt_dir


def read_colmap_refined_cameras(sparse_txt_dir, image_paths, W_orig, H_orig):
    """
    Parse BA-refined cameras and return (extrinsics, intrinsics_full)
    in full original image space.
    """
    try:
        from scipy.spatial.transform import Rotation as ScipyRot
    except ImportError:
        print("    [BA] scipy not available")
        return None, None

    S = len(image_paths)
    cam_params = {}

    cameras_txt = os.path.join(sparse_txt_dir, "cameras.txt")
    if not os.path.isfile(cameras_txt):
        return None, None
    with open(cameras_txt) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            # PINHOLE: WIDTH HEIGHT FX FY CX CY — already in full-image space
            fx, fy, cx, cy = (
                float(parts[4]), float(parts[5]),
                float(parts[6]), float(parts[7]),
            )
            cam_params[cam_id] = (fx, fy, cx, cy)

    images_txt = os.path.join(sparse_txt_dir, "images.txt")
    if not os.path.isfile(images_txt):
        return None, None

    img_name_to_pose = {}
    with open(images_txt) as fh:
        raw = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    i = 0
    while i < len(raw):
        parts = raw[i].split()
        if len(parts) < 10:
            i += 1; continue
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        cam_id   = int(parts[8])
        img_name = parts[9]
        rot = ScipyRot.from_quat([qx, qy, qz, qw]).as_matrix()
        img_name_to_pose[img_name] = (rot, np.array([tx, ty, tz], dtype=np.float32), cam_id)
        i += 2

    if not img_name_to_pose:
        return None, None

    extrinsics      = np.zeros((S, 3, 4), dtype=np.float32)
    intrinsics_full = np.zeros((S, 3, 3), dtype=np.float32)

    for idx, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        if img_name not in img_name_to_pose:
            print(f"    [BA] {img_name} not in refined model")
            return None, None
        rot, t, cam_id = img_name_to_pose[img_name]
        extrinsics[idx, :3, :3] = rot
        extrinsics[idx, :3, 3]  = t
        if cam_id not in cam_params:
            return None, None
        fx, fy, cx, cy = cam_params[cam_id]
        intrinsics_full[idx] = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1 ],
        ], dtype=np.float32)

    return extrinsics, intrinsics_full


def reproj_metrics_from_full_intrinsics(
    gt_tracks_orig, gt_vis_mask,
    extrinsics, intrinsics_full,
    pred_depth, roi, W_model, H_model,
    track_num=256,
):
    """
    Compute reproj metrics after BA, using full-image intrinsics.
    We still need to look up depth in ROI model space.
    """
    x1, y1, x2, y2 = roi
    sx = W_model / (x2 - x1)
    sy = H_model / (y2 - y1)

    vis0 = gt_vis_mask[0]
    valid_ids = np.where(vis0)[0]
    rng = np.random.RandomState(42)
    if len(valid_ids) > track_num:
        valid_ids = rng.choice(valid_ids, size=track_num, replace=False)
    N_use = len(valid_ids)
    if N_use == 0:
        return None, None

    d0 = pred_depth[0]
    K0_full = intrinsics_full[0]
    R0 = extrinsics[0, :3, :3]
    T0 = extrinsics[0, :3, 3]

    # Query depth at ROI model coords
    u0_orig = gt_tracks_orig[0, valid_ids, 0]
    v0_orig = gt_tracks_orig[0, valid_ids, 1]
    u0_roi  = (u0_orig - x1) * sx
    v0_roi  = (v0_orig - y1) * sy
    u0i = np.clip(np.floor(u0_roi).astype(int), 0, W_model - 1)
    v0i = np.clip(np.floor(v0_roi).astype(int), 0, H_model - 1)
    d_val = d0[v0i, u0i]
    depth_valid = d_val > 0.0

    # Unproject using full-image intrinsics (with gt tracks in original px)
    fx0, fy0 = K0_full[0, 0], K0_full[1, 1]
    cx0, cy0 = K0_full[0, 2], K0_full[1, 2]
    X_cam0 = (u0_orig - cx0) / fx0 * d_val
    Y_cam0 = (v0_orig - cy0) / fy0 * d_val
    Z_cam0 = d_val
    P_cam0  = np.stack([X_cam0, Y_cam0, Z_cam0], axis=-1)
    P_world = (P_cam0 - T0[None, :]) @ R0

    S = gt_tracks_orig.shape[0]
    all_errors = []
    for j in range(1, S):
        R_j = extrinsics[j, :3, :3]
        T_j = extrinsics[j, :3, 3]
        K_j = intrinsics_full[j]
        P_camj   = P_world @ R_j.T + T_j[None, :]
        in_front = P_camj[:, 2] > 0.0
        denom = np.maximum(P_camj[:, 2], 1e-6)
        u_j = P_camj[:, 0] / denom * K_j[0, 0] + K_j[0, 2]
        v_j = P_camj[:, 1] / denom * K_j[1, 1] + K_j[1, 2]
        gt_u = gt_tracks_orig[j, valid_ids, 0]
        gt_v = gt_tracks_orig[j, valid_ids, 1]
        gt_vis_j = gt_vis_mask[j, valid_ids]
        mask = gt_vis_j & depth_valid & in_front
        err  = np.sqrt((u_j - gt_u) ** 2 + (v_j - gt_v) ** 2)
        all_errors.append(err[mask])

    errors = np.concatenate(all_errors) if all_errors else np.array([])
    if len(errors) == 0:
        return None, None

    metrics = {
        "reproj_ate":    float(errors.mean()),
        "reproj_median": float(np.median(errors)),
        "delta_1px":     float((errors < 1.0).mean()),
        "delta_2px":     float((errors < 2.0).mean()),
        "delta_5px":     float((errors < 5.0).mean()),
        "delta_10px":    float((errors < 10.0).mean()),
        "n_points":      int(N_use),
        "n_valid_depth": int(depth_valid.sum()),
    }
    return metrics, P_world


# ---------------------------------------------------------------------------
# Visualization (same as eval_3d.py, tracks already in original px)
# ---------------------------------------------------------------------------

_GT_COLOR  = (0, 220, 0)
_PANEL_DEFS = [
    ("vanilla",      (220, 60,  60),  "Vanilla"),
    ("finetuned",    (60,  130, 255), "Finetuned ROI"),
    ("vanilla_ba",   (255, 150, 50),  "Vanilla+BA"),
    ("finetuned_ba", (60,  210, 160), "Finetuned ROI+BA"),
]
_SHIFT = 4
_S16   = 1 << _SHIFT


def _encode_video(frames_dir, out_path, fps):
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "%06d.jpg"),
        "-c:v", "libx264", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p", out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    [viz] ffmpeg failed:\n{r.stderr[-400:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# AE plane metrics from 3D ROI predictions
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


def get_plane_point_ids(gt_tracks_orig, gt_vis_mask, ref_corners):
    """Return indices of GT track points visible in frame 0 and inside plane bounding box."""
    vis0  = gt_vis_mask[0]
    pts0  = gt_tracks_orig[0]
    x_min, y_min = ref_corners[:, 0].min(), ref_corners[:, 1].min()
    x_max, y_max = ref_corners[:, 0].max(), ref_corners[:, 1].max()
    inside = (
        vis0
        & (pts0[:, 0] >= x_min) & (pts0[:, 0] <= x_max)
        & (pts0[:, 1] >= y_min) & (pts0[:, 1] <= y_max)
    )
    return np.where(inside)[0]


def compute_plane_homographies_from_3d_roi(
    pts0_orig, P_world,
    extrinsics, intrinsics_roi,
    roi, W_model, H_model, S,
    ransac_thresh=3.0,
):
    """
    For each frame j = 1..S-1:
      Project P_world into ROI model space, convert back to original image coords,
      RANSAC findHomography(pts0_orig → pts_j_orig).
    Returns (S, 3, 3) homographies; H[0] = identity.
    """
    x1, y1, x2, y2 = roi
    sx = W_model / (x2 - x1)
    sy = H_model / (y2 - y1)

    homographies = np.zeros((S, 3, 3), dtype=np.float32)
    homographies[0] = np.eye(3, dtype=np.float32)

    for j in range(1, S):
        R_j = extrinsics[j, :3, :3]
        T_j = extrinsics[j, :3, 3]
        K_j = intrinsics_roi[j]

        P_camj   = P_world @ R_j.T + T_j[None, :]
        in_front = P_camj[:, 2] > 0.0

        denom    = np.maximum(P_camj[:, 2], 1e-6)
        u_j_roi  = P_camj[:, 0] / denom * K_j[0, 0] + K_j[0, 2]
        v_j_roi  = P_camj[:, 1] / denom * K_j[1, 1] + K_j[1, 2]

        # ROI model → original image
        u_j_orig = u_j_roi / sx + x1
        v_j_orig = v_j_roi / sy + y1
        pts_j    = np.stack([u_j_orig, v_j_orig], axis=-1)

        valid = in_front
        if valid.sum() < 4:
            homographies[j] = np.eye(3, dtype=np.float32)
            continue

        H, _ = cv2.findHomography(
            pts0_orig[valid].reshape(-1, 1, 2).astype(np.float32),
            pts_j[valid].reshape(-1, 1, 2).astype(np.float32),
            cv2.RANSAC, ransac_thresh,
        )
        homographies[j] = H if H is not None else np.eye(3, dtype=np.float32)

    return homographies


def compute_corner_metrics(homographies, gt_frame_indices, gt_corners):
    frame0_mask = gt_frame_indices == 0
    if not frame0_mask.any():
        return None
    ref_corners = gt_corners[frame0_mask][0].astype(np.float64)
    ref_h = np.concatenate([ref_corners, np.ones((4, 1), dtype=np.float64)], axis=1)
    gt_lookup = {int(gt_frame_indices[i]): gt_corners[i] for i in range(len(gt_frame_indices))}

    centroid_errors, corner_errors, pred_centroids = [], [], []
    for t in range(len(homographies)):
        if t not in gt_lookup:
            continue
        H    = homographies[t].astype(np.float64)
        gt_c = gt_lookup[t].astype(np.float64)
        proj = (H @ ref_h.T).T
        proj /= proj[:, 2:3]
        pred_c = proj[:, :2]
        corner_errors.append(float(np.linalg.norm(pred_c - gt_c, axis=1).mean()))
        pred_cen = pred_c.mean(axis=0)
        centroid_errors.append(float(np.linalg.norm(pred_cen - gt_c.mean(axis=0))))
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
        "plane_pt_mean_px":       float(pt.mean()),
        "plane_pt_median_px":     float(np.median(pt)),
        "plane_pt_p95_px":        float(np.percentile(pt, 95)),
        "plane_corner_mean_px":   float(ce.mean()),
        "plane_corner_median_px": float(np.median(ce)),
        "plane_corner_p95_px":    float(np.percentile(ce, 95)),
        "plane_jitter_score":     jitter,
    }


def project_corners_through_homographies(ref_corners, homographies):
    S     = len(homographies)
    ref_h = np.concatenate(
        [ref_corners.astype(np.float64), np.ones((4, 1), dtype=np.float64)], axis=1
    )
    pred = np.zeros((S, 4, 2), dtype=np.float32)
    for t in range(S):
        proj = (homographies[t].astype(np.float64) @ ref_h.T).T
        proj /= proj[:, 2:3]
        pred[t] = proj[:, :2].astype(np.float32)
    return pred


def _render_frame_planes(
    image_path, fi, W_disp, H_disp,
    gt_frame_lookup, pred_corners_by_tag,
    single_tag=None, gt_only=False,
):
    arr = np.array(__import__("PIL").Image.open(image_path).convert("RGB"))
    bg  = np.zeros((H_disp, W_disp, 3), dtype=np.uint8)
    bg[:min(arr.shape[0], H_disp), :min(arr.shape[1], W_disp)] = arr[:H_disp, :W_disp]

    _SHIFT = 4; _S16 = 1 << _SHIFT; _QUAD_ORDER = [0, 1, 3, 2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    def _scale(c4x2):
        return (c4x2.astype(np.float64)[_QUAD_ORDER] * _S16).astype(np.int32).reshape(-1, 1, 2)

    if gt_only:
        panel = bg.copy()
        if gt_frame_lookup is not None and fi in gt_frame_lookup:
            cv2.polylines(panel, [_scale(gt_frame_lookup[fi])],
                          True, _GT_COLOR[::-1], 2, cv2.LINE_AA, _SHIFT)
        cv2.putText(panel, "GT", (8, 22), font, 0.7, _GT_COLOR[::-1], 2, cv2.LINE_AA)
        return panel

    panel_defs = (
        [(t, c, l) for t, c, l in _PANEL_DEFS if t == single_tag]
        if single_tag is not None else _PANEL_DEFS
    )
    panels = []
    for tag, color, label in panel_defs:
        if tag not in pred_corners_by_tag:
            continue
        panel = bg.copy()
        if gt_frame_lookup is not None and fi in gt_frame_lookup:
            cv2.polylines(panel, [_scale(gt_frame_lookup[fi])],
                          True, _GT_COLOR[::-1], 2, cv2.LINE_AA, _SHIFT)
        cv2.polylines(panel, [_scale(pred_corners_by_tag[tag][fi])],
                      True, color[::-1], 2, cv2.LINE_AA, _SHIFT)
        cv2.putText(panel, label, (8, 22), font, 0.7, color[::-1], 2, cv2.LINE_AA)
        panels.append(panel)

    if not panels:
        return np.zeros((H_disp, W_disp, 3), dtype=np.uint8)
    divider = np.full((H_disp, 2, 3), 180, dtype=np.uint8)
    frame = panels[0]
    for p in panels[1:]:
        frame = np.concatenate([frame, divider, p], axis=1)
    return frame


def visualize_planes(
    seq_name, image_paths, W_orig, H_orig,
    gt_frame_lookup, pred_corners_by_tag,
    split_name, vis_dir, fps=8, seq_dir=None,
):
    """Save plane quad videos: planes_gt, planes_{tag}, planes_composed."""
    import tempfile, shutil as _shutil

    if not pred_corners_by_tag:
        return
    S = len(image_paths)
    if S < 2:
        return

    if seq_dir is not None:
        fps = get_sequence_fps(seq_dir, fallback=fps)

    W_disp = W_orig + (W_orig % 2)
    H_disp = H_orig + (H_orig % 2)
    out_dir = os.path.join(vis_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    def _save_video(render_kwargs, suffix):
        out_path = os.path.join(out_dir, f"{seq_name}_{suffix}.mp4")
        tmp_dir  = tempfile.mkdtemp(prefix="planes3d_roi_viz_")
        try:
            for fi in range(S):
                frame_rgb = _render_frame_planes(
                    image_paths[fi], fi, W_disp, H_disp,
                    gt_frame_lookup, pred_corners_by_tag,
                    **render_kwargs,
                )
                cv2.imwrite(os.path.join(tmp_dir, f"{fi:06d}.jpg"),
                            frame_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if _encode_video(tmp_dir, out_path, fps):
                print(f"    [viz planes] -> {out_path}  ({S} frames @ {fps:.1f}fps)")
        finally:
            _shutil.rmtree(tmp_dir, ignore_errors=True)

    _save_video({"gt_only": True}, "planes_gt")
    for tag in [t for t, _, _ in _PANEL_DEFS if t in pred_corners_by_tag]:
        _save_video({"single_tag": tag}, f"planes_{tag}")
    _save_video({}, "planes_composed")


def compute_plane_homographies_from_3d_full_intrinsics(
    pts0_orig, P_world,
    extrinsics, intrinsics_full,
    S, ransac_thresh=3.0,
):
    """
    BA variant: intrinsics_full is in full original image space.
    Project P_world → original image coords directly (no ROI offset needed).
    """
    homographies = np.zeros((S, 3, 3), dtype=np.float32)
    homographies[0] = np.eye(3, dtype=np.float32)

    for j in range(1, S):
        R_j = extrinsics[j, :3, :3]
        T_j = extrinsics[j, :3, 3]
        K_j = intrinsics_full[j]

        P_camj   = P_world @ R_j.T + T_j[None, :]
        in_front = P_camj[:, 2] > 0.0
        denom    = np.maximum(P_camj[:, 2], 1e-6)
        u_j      = P_camj[:, 0] / denom * K_j[0, 0] + K_j[0, 2]  # original image coords
        v_j      = P_camj[:, 1] / denom * K_j[1, 1] + K_j[1, 2]
        pts_j    = np.stack([u_j, v_j], axis=-1)

        valid = in_front
        if valid.sum() < 4:
            homographies[j] = np.eye(3, dtype=np.float32)
            continue

        H, _ = cv2.findHomography(
            pts0_orig[valid].reshape(-1, 1, 2).astype(np.float32),
            pts_j[valid].reshape(-1, 1, 2).astype(np.float32),
            cv2.RANSAC, ransac_thresh,
        )
        homographies[j] = H if H is not None else np.eye(3, dtype=np.float32)

    return homographies


def _render_frame_reproj(image_path, fi, gt_tracks_orig, gt_vis_mask,
                          reproj_by_tag, valid_ids_by_tag,
                          sampled_ids, W_disp, H_disp,
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
    dot_r = 4; font = cv2.FONT_HERSHEY_SIMPLEX; bar_h = 20

    if gt_only:
        panel = bg.copy()
        for tid in sampled_ids:
            if gt_vis_mask[fi, tid]:
                gx_s = round(float(gt_tracks_orig[fi, tid, 0]) * _S16)
                gy_s = round(float(gt_tracks_orig[fi, tid, 1]) * _S16)
                cv2.circle(panel, (gx_s, gy_s), dot_r * _S16,
                           _GT_COLOR[::-1], -1, cv2.LINE_AA, _SHIFT)
        cv2.putText(panel, "GT", (8, 22), font, 0.7, _GT_COLOR[::-1], 2, cv2.LINE_AA)
        panel[-bar_h:] = (20, 20, 20)
        return panel

    panel_defs = (
        [(t, c, l) for t, c, l in _PANEL_DEFS if t == single_tag]
        if single_tag is not None else _PANEL_DEFS
    )
    panels = []
    for tag, color, label in panel_defs:
        if tag not in reproj_by_tag:
            continue
        reproj_tracks = reproj_by_tag[tag]
        valid_ids     = valid_ids_by_tag[tag]
        panel = bg.copy()
        for tid in sampled_ids:
            if gt_vis_mask[fi, tid]:
                gx_s = round(float(gt_tracks_orig[fi, tid, 0]) * _S16)
                gy_s = round(float(gt_tracks_orig[fi, tid, 1]) * _S16)
                cv2.circle(panel, (gx_s, gy_s), dot_r * _S16,
                           _GT_COLOR[::-1], -1, cv2.LINE_AA, _SHIFT)
            loc = np.where(valid_ids == tid)[0]
            if len(loc):
                rx_s = round(float(reproj_tracks[fi, loc[0], 0]) * _S16)
                ry_s = round(float(reproj_tracks[fi, loc[0], 1]) * _S16)
                cv2.circle(panel, (rx_s, ry_s), dot_r * _S16,
                           color[::-1], 2, cv2.LINE_AA, _SHIFT)
        cv2.putText(panel, label, (8, 22), font, 0.7, color[::-1], 2, cv2.LINE_AA)
        cv2.putText(panel, f"frame {fi}", (8, H_disp - 8), font, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        panel[-bar_h:] = (20, 20, 20)
        panels.append(panel)

    if not panels:
        return np.zeros((H_disp, W_disp, 3), dtype=np.uint8)
    divider = np.full((H_disp, 2, 3), 180, dtype=np.uint8)
    frame = panels[0]
    for p in panels[1:]:
        frame = np.concatenate([frame, divider, p], axis=1)
    return frame


def visualize_sequence_reproj(seq_name, image_paths, gt_tracks_orig, gt_vis_mask,
                               reproj_by_tag, valid_ids_by_tag,
                               split_name, vis_dir, n_tracks=50, fps=8,
                               W_orig=None, H_orig=None, seq_dir=None):
    S = len(image_paths)
    if S < 2 or not reproj_by_tag:
        return

    all_valid = set(np.where(gt_vis_mask[0])[0].tolist())
    for vids in valid_ids_by_tag.values():
        all_valid &= set(vids.tolist())
    all_valid_arr = np.array(sorted(all_valid))
    if len(all_valid_arr) == 0:
        return

    rng = np.random.RandomState(42)
    sampled_ids = rng.choice(all_valid_arr, size=min(n_tracks, len(all_valid_arr)), replace=False)

    if W_orig is None or H_orig is None:
        W_orig, H_orig, _, _ = get_model_resolution(image_paths[0])
    W_disp = W_orig + (W_orig % 2)
    H_disp = H_orig + (H_orig % 2)

    if seq_dir is not None:
        fps = get_sequence_fps(seq_dir, fallback=fps)

    out_dir = os.path.join(vis_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)

    def _save_video(render_kwargs, suffix):
        out_path = os.path.join(out_dir, f"{seq_name}_{suffix}.mp4")
        tmp_dir = tempfile.mkdtemp(prefix="roi3d_viz_")
        try:
            for fi in range(S):
                frame_rgb = _render_frame_reproj(
                    image_paths[fi], fi, gt_tracks_orig, gt_vis_mask,
                    reproj_by_tag, valid_ids_by_tag, sampled_ids, W_disp, H_disp,
                    **render_kwargs,
                )
                cv2.imwrite(os.path.join(tmp_dir, f"{fi:06d}.jpg"),
                            frame_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
            if _encode_video(tmp_dir, out_path, fps):
                print(f"    [viz] -> {out_path}  ({S} frames @ {fps:.1f}fps)")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    _save_video({"gt_only": True}, "reproj_gt")
    for tag in reproj_by_tag:
        _save_video({"single_tag": tag}, f"reproj_{tag}")
    _save_video({}, "reproj_composed")


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "reproj_ate", "reproj_median",
    "delta_1px", "delta_2px", "delta_5px", "delta_10px",
]

PLANE_METRIC_KEYS = [
    "plane_pt_mean_px", "plane_pt_median_px", "plane_pt_p95_px",
    "plane_corner_mean_px", "plane_corner_median_px", "plane_corner_p95_px",
    "plane_jitter_score",
]


def eval_sequence_3d_roi(
    seq_name, dataset_dir, models, roi_pad,
    track_num, device, dtype,
    colmap_dir=None,
    run_ba=False, ba_colmap_dir=None, ba_colmap_bin="colmap", ba_max_iter=100,
    vis_n_tracks=50, vis_fps=8, vis_dir=None, split_name="val",
    max_frames=0,
):
    seq_dir   = os.path.join(dataset_dir, seq_name)
    image_dir = os.path.join(seq_dir, "images")
    _skip = (None, None, None, None, None)

    if not os.path.isdir(image_dir):
        print(f"  [{seq_name}] SKIP -- no images dir")
        return _skip

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not image_paths:
        print(f"  [{seq_name}] SKIP -- no images")
        return _skip

    tracks_path = os.path.join(seq_dir, "tracks.npy")
    masks_path  = os.path.join(seq_dir, "track_masks.npy")
    if not os.path.isfile(tracks_path):
        print(f"  [{seq_name}] SKIP -- no tracks.npy")
        return _skip

    gt_tracks_orig = np.load(tracks_path).astype(np.float32)
    gt_vis_mask = (
        np.load(masks_path).astype(bool) if os.path.isfile(masks_path)
        else np.ones(gt_tracks_orig.shape[:2], dtype=bool)
    )

    S = len(image_paths)
    gt_tracks_orig = gt_tracks_orig[:S]
    gt_vis_mask    = gt_vis_mask[:S]

    if max_frames > 0 and S > max_frames:
        indices = np.linspace(0, S - 1, max_frames, dtype=int)
        image_paths    = [image_paths[i] for i in indices]
        gt_tracks_orig = gt_tracks_orig[indices]
        gt_vis_mask    = gt_vis_mask[indices]
        S = max_frames

    W_orig, H_orig, _, _ = get_model_resolution(image_paths[0])

    # Compute ROI
    roi = load_roi_from_corners(seq_dir, roi_pad, W_orig, H_orig)
    if roi is None:
        print(f"  [{seq_name}] SKIP -- no corners.csv (ROI required)")
        return _skip
    x1, y1, x2, y2 = roi
    print(
        f"  [{seq_name}] ROI: ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})"
        f"  size: {x2-x1:.0f}x{y2-y1:.0f}  pad={roi_pad}"
    )

    # Load images cropped to ROI
    try:
        images_tensor, W_model, H_model = load_and_preprocess_images_roi(image_paths, roi)
    except Exception as exc:
        print(f"  [{seq_name}] SKIP -- ROI image load error: {exc}")
        return _skip

    x1, y1, x2, y2 = roi
    sx_roi = W_model / (x2 - x1)
    sy_roi = H_model / (y2 - y1)

    # Load GT corners once per sequence for plane AE metrics
    gt_frame_indices, gt_corners = load_gt_corners(seq_dir)
    gt_frame_lookup     = None
    ref_corners         = None
    plane_ids           = None
    pred_corners_by_tag = {}

    if gt_frame_indices is not None:
        gt_frame_lookup = {
            int(gt_frame_indices[i]): gt_corners[i]
            for i in range(len(gt_frame_indices))
        }
        frame0_mask = gt_frame_indices == 0
        if frame0_mask.any():
            ref_corners = gt_corners[frame0_mask][0]
            plane_ids   = get_plane_point_ids(gt_tracks_orig, gt_vis_mask, ref_corners)
            if len(plane_ids) < 4:
                plane_ids = None

    seq_results      = {}
    reproj_by_tag    = {}
    valid_ids_by_tag = {}

    for tag, model in models.items():
        extrinsics, intrinsics_roi, pred_depth = run_camera_and_depth(
            model, images_tensor, H_model, W_model, device, dtype
        )
        # Convert intrinsics to full-image space (needed for COLMAP and BA reproj)
        intrinsics_full = intrinsics_roi_to_full(intrinsics_roi, roi, W_model, H_model)

        out = compute_reproj_metrics_roi(
            gt_tracks_orig, gt_vis_mask,
            extrinsics, intrinsics_roi,
            pred_depth, roi, W_model, H_model,
            track_num=track_num,
        )
        metrics, pts3d_world, depth_valid, reproj_tracks, valid_ids = out

        if metrics is None:
            print(f"  [{seq_name}][{tag}] SKIP -- no valid points")
            continue

        seq_results[tag] = metrics
        reproj_by_tag[tag]    = reproj_tracks
        valid_ids_by_tag[tag] = valid_ids

        if colmap_dir is not None:
            write_colmap(
                os.path.join(colmap_dir, seq_name, tag),
                image_paths, extrinsics, intrinsics_full,
                W_orig, H_orig,
                pts3d_world=pts3d_world,
                depth_valid=depth_valid,
                gt_tracks_orig=gt_tracks_orig[:, valid_ids, :],
                gt_vis_mask=gt_vis_mask[:, valid_ids],
            )

        print(
            f"  [{seq_name}][{tag}]"
            f"  reproj_ATE={metrics['reproj_ate']:.2f}px"
            f"  d5={metrics['delta_5px']:.3f}"
        )

        # --- Plane AE metrics from 3D ROI ---
        if plane_ids is not None:
            pts0_orig = gt_tracks_orig[0, plane_ids, :]
            K0 = intrinsics_roi[0]
            R0 = extrinsics[0, :3, :3]
            T0 = extrinsics[0, :3, 3]

            u0_roi = (pts0_orig[:, 0] - x1) * sx_roi
            v0_roi = (pts0_orig[:, 1] - y1) * sy_roi
            u0i = np.floor(np.clip(u0_roi, 0, W_model - 1)).astype(int)
            v0i = np.floor(np.clip(v0_roi, 0, H_model - 1)).astype(int)
            d_val = pred_depth[0, v0i, u0i]

            X0 = (u0_roi - K0[0, 2]) / K0[0, 0] * d_val
            Y0 = (v0_roi - K0[1, 2]) / K0[1, 1] * d_val
            P_cam0 = np.stack([X0, Y0, d_val], axis=-1)
            P_world_plane = (P_cam0 - T0[None, :]) @ R0

            homographies = compute_plane_homographies_from_3d_roi(
                pts0_orig, P_world_plane,
                extrinsics, intrinsics_roi,
                roi, W_model, H_model, S,
            )
            plane_metrics = compute_corner_metrics(homographies, gt_frame_indices, gt_corners)
            if plane_metrics is not None:
                seq_results[tag].update(plane_metrics)
                print(
                    f"  [{seq_name}][{tag}]"
                    f"  plane_pt={plane_metrics['plane_pt_mean_px']:.2f}px"
                    f"  plane_corner={plane_metrics['plane_corner_mean_px']:.2f}px"
                    f"  jitter={plane_metrics['plane_jitter_score']:.3f}"
                )
            pred_corners_by_tag[tag] = project_corners_through_homographies(
                ref_corners, homographies
            )

        # --- Bundle adjustment ---
        if run_ba and hasattr(model, "track_head") and model.track_head is not None:
            try:
                # Query BA correspondences: map GT frame-0 points to ROI model space
                q0 = gt_tracks_orig[0, valid_ids, :].copy()
                q0_roi_model = roi_orig_to_model(q0, roi, W_model, H_model)

                print(f"  [{seq_name}][{tag}] running track head for BA ({len(valid_ids)} pts)...")
                pred_tracks_roi, pred_vis = _run_track_head_ba(
                    model, images_tensor, q0_roi_model, device, dtype
                )

                ba_sparse_in = None
                if ba_colmap_dir is not None:
                    ba_write_dir = os.path.join(ba_colmap_dir, seq_name, tag)
                    ba_sparse_in = write_colmap_from_pred_tracks(
                        ba_write_dir, image_paths,
                        extrinsics, intrinsics_full,
                        W_orig, H_orig,
                        pts3d_world, depth_valid,
                        pred_tracks_roi, pred_vis,
                        roi, W_model, H_model,
                    )
                else:
                    _ba_tmp = tempfile.mkdtemp(prefix="vggt_ba_in_")
                    ba_sparse_in = write_colmap_from_pred_tracks(
                        _ba_tmp, image_paths,
                        extrinsics, intrinsics_full,
                        W_orig, H_orig,
                        pts3d_world, depth_valid,
                        pred_tracks_roi, pred_vis,
                        roi, W_model, H_model,
                    )

                if ba_sparse_in is not None:
                    ba_work_tmp = tempfile.mkdtemp(prefix="vggt_ba_work_")
                    try:
                        ba_txt_dir = run_bundle_adjustment(
                            ba_sparse_in, ba_work_tmp,
                            colmap_bin=ba_colmap_bin, max_iter=ba_max_iter,
                        )
                        if ba_txt_dir is not None:
                            ext_ba, int_ba_full = read_colmap_refined_cameras(
                                ba_txt_dir, image_paths, W_orig, H_orig
                            )
                            if ext_ba is not None:
                                ba_metrics, _ = reproj_metrics_from_full_intrinsics(
                                    gt_tracks_orig, gt_vis_mask,
                                    ext_ba, int_ba_full,
                                    pred_depth, roi, W_model, H_model,
                                    track_num=track_num,
                                )
                                if ba_metrics is not None:
                                    seq_results[f"{tag}_ba"] = ba_metrics
                                    print(
                                        f"  [{seq_name}][{tag}_ba]"
                                        f"  reproj_ATE={ba_metrics['reproj_ate']:.2f}px"
                                        f"  d5={ba_metrics['delta_5px']:.3f}"
                                    )

                                    # Plane AE metrics using BA-refined cameras (full-image space)
                                    if plane_ids is not None:
                                        pts0_orig = gt_tracks_orig[0, plane_ids, :]
                                        K0_full   = int_ba_full[0]
                                        R0_ba     = ext_ba[0, :3, :3]
                                        T0_ba     = ext_ba[0, :3, 3]

                                        u0_roi = (pts0_orig[:, 0] - x1) * sx_roi
                                        v0_roi = (pts0_orig[:, 1] - y1) * sy_roi
                                        u0i = np.floor(np.clip(u0_roi, 0, W_model - 1)).astype(int)
                                        v0i = np.floor(np.clip(v0_roi, 0, H_model - 1)).astype(int)
                                        d_val_ba = pred_depth[0, v0i, u0i]

                                        X0 = (pts0_orig[:, 0] - K0_full[0, 2]) / K0_full[0, 0] * d_val_ba
                                        Y0 = (pts0_orig[:, 1] - K0_full[1, 2]) / K0_full[1, 1] * d_val_ba
                                        P_cam0_ba = np.stack([X0, Y0, d_val_ba], axis=-1)
                                        P_world_ba = (P_cam0_ba - T0_ba[None, :]) @ R0_ba

                                        hom_ba = compute_plane_homographies_from_3d_full_intrinsics(
                                            pts0_orig, P_world_ba,
                                            ext_ba, int_ba_full, S,
                                        )
                                        pm_ba = compute_corner_metrics(
                                            hom_ba, gt_frame_indices, gt_corners
                                        )
                                        if pm_ba is not None:
                                            seq_results[f"{tag}_ba"].update(pm_ba)
                                            print(
                                                f"  [{seq_name}][{tag}_ba]"
                                                f"  plane_pt={pm_ba['plane_pt_mean_px']:.2f}px"
                                                f"  plane_corner={pm_ba['plane_corner_mean_px']:.2f}px"
                                                f"  jitter={pm_ba['plane_jitter_score']:.3f}"
                                            )
                                        pred_corners_by_tag[f"{tag}_ba"] = \
                                            project_corners_through_homographies(
                                                ref_corners, hom_ba
                                            )
                    finally:
                        shutil.rmtree(ba_work_tmp, ignore_errors=True)
            except Exception as e:
                print(f"  [{seq_name}][{tag}] BA failed: {e}")

    if vis_dir is not None and reproj_by_tag:
        visualize_sequence_reproj(
            seq_name, image_paths, gt_tracks_orig, gt_vis_mask,
            reproj_by_tag, valid_ids_by_tag,
            split_name, vis_dir,
            n_tracks=vis_n_tracks, fps=vis_fps,
            W_orig=W_orig, H_orig=H_orig, seq_dir=seq_dir,
        )

    if vis_dir is not None and gt_frame_lookup is not None and pred_corners_by_tag:
        visualize_planes(
            seq_name, image_paths, W_orig, H_orig,
            gt_frame_lookup, pred_corners_by_tag,
            split_name, vis_dir, fps=vis_fps, seq_dir=seq_dir,
        )

    return (
        seq_results if seq_results else None,
        image_paths, gt_tracks_orig, gt_vis_mask, reproj_by_tag,
    )


# ---------------------------------------------------------------------------
# Split-level evaluation
# ---------------------------------------------------------------------------

def mean_over_seqs(seq_results, tag):
    all_keys = METRIC_KEYS + PLANE_METRIC_KEYS
    vals = {k: [] for k in all_keys}
    for sr in seq_results.values():
        if sr is None or tag not in sr:
            continue
        for k in all_keys:
            if k in sr[tag]:
                vals[k].append(sr[tag][k])
    return {
        k: float(np.mean(v)) if v else float("nan")
        for k, v in vals.items()
    }


def eval_split(
    split_name, sequences, dataset_dir, models, roi_pad,
    track_num, device, dtype, colmap_dir=None,
    run_ba=False, ba_colmap_dir=None, ba_colmap_bin="colmap", ba_max_iter=100,
    vis_max_seqs=3, vis_n_tracks=50, vis_fps=8, vis_dir=None,
    max_frames=0,
):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"SPLIT: {split_name.upper()}  ({len(sequences)} sequences)")
    print(sep)

    seq_results = {}
    viz_count   = 0
    for seq_name in sequences:
        do_viz = (
            vis_dir is not None
            and vis_max_seqs != 0
            and (vis_max_seqs < 0 or viz_count < vis_max_seqs)
        )
        try:
            r, *_ = eval_sequence_3d_roi(
                seq_name, dataset_dir, models, roi_pad,
                track_num, device, dtype,
                colmap_dir=colmap_dir,
                run_ba=run_ba,
                ba_colmap_dir=ba_colmap_dir,
                ba_colmap_bin=ba_colmap_bin,
                ba_max_iter=ba_max_iter,
                vis_n_tracks=vis_n_tracks,
                vis_fps=vis_fps,
                vis_dir=vis_dir if do_viz else None,
                split_name=split_name,
                max_frames=max_frames,
            )
        except RuntimeError as exc:
            print(f"  [{seq_name}] SKIP -- RuntimeError: {exc}")
            torch.cuda.empty_cache()
            r = None

        seq_results[seq_name] = r
        if r is not None and do_viz:
            viz_count += 1

    base_tags = list(models.keys())
    all_tags_seen = set()
    for sr in seq_results.values():
        if sr:
            all_tags_seen.update(sr.keys())
    tags = [t for t in base_tags if t in all_tags_seen]
    tags += [t for t in sorted(all_tags_seen) if t not in base_tags]
    if not tags:
        tags = base_tags
    means = {tag: mean_over_seqs(seq_results, tag) for tag in tags}

    print(f"\n--- {split_name.upper()} MEAN METRICS ---")
    for tag in tags:
        m = means[tag]
        print(
            f"  [{tag}]"
            f"  reproj_ATE={m['reproj_ate']:.3f}px"
            f"  median={m['reproj_median']:.3f}px"
            f"  d1={m['delta_1px']:.3f}"
            f"  d5={m['delta_5px']:.3f}"
            f"  d10={m['delta_10px']:.3f}"
        )
        if not np.isnan(m.get("plane_pt_mean_px", float("nan"))):
            print(
                f"  [{tag}] plane:"
                f"  pt_mean={m['plane_pt_mean_px']:.3f}px"
                f"  corner_mean={m['plane_corner_mean_px']:.3f}px"
                f"  corner_p95={m['plane_corner_p95_px']:.3f}px"
                f"  jitter={m['plane_jitter_score']:.3f}"
            )

    return {"per_seq": seq_results, "mean": means}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    print(f"\nLoading fine-tuned model: {args.finetuned_ckpt}")
    finetuned_model = load_model(
        args.finetuned_ckpt,
        lora=args.lora, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_targets=args.lora_target_modules,
        device=device, enable_track=args.run_ba,
    )
    models = {"finetuned": finetuned_model}

    if args.run_vanilla:
        print(f"Loading vanilla model: {args.vanilla_ckpt}")
        vanilla_model = load_model(
            args.vanilla_ckpt, lora=False, device=device, enable_track=args.run_ba,
        )
        models["vanilla"] = vanilla_model

    if args.run_ba:
        print("BA mode enabled: models loaded with track head")

    print(f"ROI pad: {args.roi_pad}  (sequences without corners.csv skipped)")

    def read_split(path):
        with open(path) as fh:
            return [line.strip() for line in fh if line.strip()]

    all_results = {}

    val_seqs = read_split(args.val_split_file)
    all_results["val"] = eval_split(
        "val", val_seqs, args.dataset_dir, models, args.roi_pad,
        args.track_num, device, dtype,
        colmap_dir=os.path.join(args.colmap_dir, "val") if args.colmap_dir else None,
        run_ba=args.run_ba,
        ba_colmap_dir=(
            os.path.join(args.colmap_dir, "val_ba")
            if args.colmap_dir and args.run_ba else None
        ),
        ba_colmap_bin=args.ba_colmap_bin,
        ba_max_iter=args.ba_max_iter,
        vis_max_seqs=args.vis_max_seqs,
        vis_n_tracks=args.vis_n_tracks,
        vis_fps=args.vis_fps,
        vis_dir=args.vis_dir,
        max_frames=args.max_frames,
    )

    if args.train_split_file:
        train_seqs = read_split(args.train_split_file)
        if 0 < args.train_max_seqs < len(train_seqs):
            random.seed(0)
            train_seqs = random.sample(train_seqs, args.train_max_seqs)
            print(f"\n(train subsampled to {len(train_seqs)} sequences)")
        all_results["train"] = eval_split(
            "train", train_seqs, args.dataset_dir, models, args.roi_pad,
            args.track_num, device, dtype,
            colmap_dir=os.path.join(args.colmap_dir, "train") if args.colmap_dir else None,
            run_ba=args.run_ba,
            ba_colmap_dir=(
                os.path.join(args.colmap_dir, "train_ba")
                if args.colmap_dir and args.run_ba else None
            ),
            ba_colmap_bin=args.ba_colmap_bin,
            ba_max_iter=args.ba_max_iter,
            vis_max_seqs=args.vis_max_seqs,
            vis_n_tracks=args.vis_n_tracks,
            vis_fps=args.vis_fps,
            vis_dir=args.vis_dir,
            max_frames=args.max_frames,
        )

    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(args.output_json, "w") as fh:
        json.dump(to_python(all_results), fh, indent=2)
    print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
