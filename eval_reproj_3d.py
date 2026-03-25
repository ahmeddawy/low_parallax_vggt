#!/usr/bin/env python3
"""
eval_reproj_3d.py - 3D reprojection evaluation.

Do predicted depth + cameras jointly explain GT track motion?

Protocol:
  For each GT track point (u0, v0) visible in frame 0:
    1. Look up VGGT-predicted depth d0 at (u0, v0) in model resolution.
    2. Unproject to 3D using predicted camera 0:
         P_world = R0^T @ (K0^-1 @ [u,v,1]*d - T0)
    3. Project to frame j using predicted camera j:
         [u,v] = Kj @ (Rj @ P_world + Tj)
    4. Scale back to original resolution; compare with GT track.
  Metrics: reproj_ATE, median_te, delta@1/2/5/10px (original px, frames 1..S)

COLMAP export (--colmap-dir):
  Predicted cameras + GT-track-derived 3D points in COLMAP sparse format.
    {colmap_dir}/{seq_name}/{vanilla|finetuned}/sparse/0/
      cameras.txt  - PINHOLE intrinsics (original image resolution)
      images.txt   - extrinsics per frame
      points3D.txt - GT tracks unprojected via predicted depth + cameras

Videos (--vis-dir):
  Side-by-side per sequence: left=vanilla reproj vs GT, right=finetuned vs GT.

Usage:
    python eval_reproj_3d.py \\
        --vanilla-ckpt     /workspace/model.pt \\
        --finetuned-ckpt   /workspace/ckpts/checkpoint.pt \\
        --dataset-dir      /mnt/.../dataset \\
        --val-split-file   /mnt/.../val_split.txt \\
        [--train-split-file /mnt/.../train_split.txt --train-max-seqs 20] \\
        [--lora] [--lora-r 16] [--lora-alpha 32] \\
        [--colmap-dir reproj_colmap_out] \\
        [--vis-dir reproj_eval_viz] \\
        [--output-json reproj_eval_results.json]
"""

import argparse
import glob
import json
import os
import random
import shutil
import subprocess
import tempfile

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="3D reprojection evaluation of VGGT predictions"
    )
    p.add_argument("--vanilla-ckpt", required=True)
    p.add_argument("--finetuned-ckpt", required=True)
    p.add_argument("--dataset-dir", required=True)
    p.add_argument(
        "--train-split-file", default=None,
        help="Train split (one seq per line). Omit to skip train eval.",
    )
    p.add_argument("--val-split-file", required=True)
    p.add_argument(
        "--train-max-seqs", type=int, default=20,
        help="Randomly sample this many train sequences (-1 = all)",
    )
    p.add_argument("--lora", action="store_true", default=False)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-target-modules", default="qkv,proj")
    p.add_argument(
        "--track-num", type=int, default=256,
        help="Max GT tracks per sequence (random sample)",
    )
    p.add_argument(
        "--colmap-dir", default=None,
        help="If set, write COLMAP sparse reconstructions here",
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
    p.add_argument("--vis-fps", type=int, default=8)
    p.add_argument("--vis-dir", default="reproj_eval_viz")
    p.add_argument("--output-json", default="reproj_eval_results.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    ckpt_path, lora=False, lora_r=16, lora_alpha=32.0,
    lora_targets="qkv,proj", device="cuda",
):
    model = VGGT(enable_point=False, enable_track=False)

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
        print(f"  LoRA applied (r={lora_r}, alpha={lora_alpha})")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"  [WARN] {len(missing)} missing keys "
            f"(expected for LoRA / partial ckpt)"
        )

    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Resolution helper
# ---------------------------------------------------------------------------

def get_model_resolution(image_path):
    """Return (W_orig, H_orig, W_model, H_model)."""
    img = Image.open(image_path)
    W_orig, H_orig = img.size
    W_model = 518
    H_model = round(H_orig * (W_model / W_orig) / 14) * 14
    return W_orig, H_orig, W_model, H_model


# ---------------------------------------------------------------------------
# Camera + depth inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_camera_and_depth(
    model, images_tensor, H_model, W_model, device, dtype
):
    """
    Run aggregator + camera head + depth head.

    Returns:
        extrinsics: (S, 3, 4) float32 numpy  [R|t], world-to-cam, OpenCV
        intrinsics: (S, 3, 3) float32 numpy  PINHOLE at model resolution
        pred_depth: (S, H_model, W_model) float32 numpy
    """
    images_batch = images_tensor.unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(dtype=dtype):
        agg_tokens, patch_start_idx = model.aggregator(images_batch)

    with torch.cuda.amp.autocast(enabled=False):
        pose_enc_list = model.camera_head(agg_tokens)
        depth, _ = model.depth_head(
            agg_tokens,
            images=images_batch,
            patch_start_idx=patch_start_idx,
        )

    pose_enc = pose_enc_list[-1]  # (1, S, 9)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=(H_model, W_model)
    )
    extrinsics = extrinsics.squeeze(0).float().cpu().numpy()  # (S, 3, 4)
    intrinsics = intrinsics.squeeze(0).float().cpu().numpy()  # (S, 3, 3)
    # depth: (1, S, H, W, 1) -> (S, H, W)
    pred_depth = depth.squeeze(0).squeeze(-1).float().cpu().numpy()

    return extrinsics, intrinsics, pred_depth


# ---------------------------------------------------------------------------
# 3D reprojection computation
# ---------------------------------------------------------------------------

def compute_reproj_metrics(
    gt_tracks_orig, gt_vis_mask,
    extrinsics, intrinsics,
    pred_depth, W_orig, H_orig, W_model, H_model,
    track_num=256,
):
    """
    Unproject GT tracks from frame 0 via predicted depth + cameras,
    project to frames 1..S, measure pixel error vs GT tracks.

    gt_tracks_orig : (S, N, 2) float32 in original px space
    gt_vis_mask    : (S, N) bool
    extrinsics     : (S, 3, 4) predicted [R|t] world-to-cam
    intrinsics     : (S, 3, 3) predicted K at model resolution
    pred_depth     : (S, H_model, W_model) predicted depth

    Returns:
        metrics           - dict or None
        P_world           - (N_use, 3) 3D points for COLMAP
        depth_valid       - (N_use,) bool
        reproj_tracks_orig - (S, N_use, 2) reprojected positions in orig px
        valid_ids         - (N_use,) indices into original track array
    """
    sx = W_model / W_orig
    sy = H_model / H_orig

    vis0 = gt_vis_mask[0]
    valid_ids = np.where(vis0)[0]
    rng = np.random.RandomState(42)
    if len(valid_ids) > track_num:
        valid_ids = rng.choice(valid_ids, size=track_num, replace=False)
    N_use = len(valid_ids)

    if N_use == 0:
        return None, None, None, None, None

    d0 = pred_depth[0]  # (H_model, W_model)

    K0 = intrinsics[0]
    R0 = extrinsics[0, :3, :3]
    T0 = extrinsics[0, :3, 3]

    u0 = gt_tracks_orig[0, valid_ids, 0] * sx
    v0 = gt_tracks_orig[0, valid_ids, 1] * sy

    u0_c = np.clip(u0, 0, W_model - 1)
    v0_c = np.clip(v0, 0, H_model - 1)
    u0i = np.floor(u0_c).astype(int)
    v0i = np.floor(v0_c).astype(int)
    d_val = d0[v0i, u0i]

    depth_valid = d_val > 0.0

    fx0, fy0 = K0[0, 0], K0[1, 1]
    cx0, cy0 = K0[0, 2], K0[1, 2]
    X_cam0 = (u0 - cx0) / fx0 * d_val
    Y_cam0 = (v0 - cy0) / fy0 * d_val
    Z_cam0 = d_val
    P_cam0 = np.stack([X_cam0, Y_cam0, Z_cam0], axis=-1)  # (N_use, 3)

    # P_world = R0^T @ (P_cam0 - T0)  (row-vector form)
    P_world = (P_cam0 - T0[None, :]) @ R0  # (N_use, 3)

    S = gt_tracks_orig.shape[0]
    all_errors = []

    reproj_tracks_orig = np.zeros((S, N_use, 2), dtype=np.float32)
    reproj_tracks_orig[0] = gt_tracks_orig[0, valid_ids, :]

    for j in range(1, S):
        R_j = extrinsics[j, :3, :3]
        T_j = extrinsics[j, :3, 3]
        K_j = intrinsics[j]

        P_camj = P_world @ R_j.T + T_j[None, :]  # (N_use, 3)
        in_front = P_camj[:, 2] > 0.0

        fx_j, fy_j = K_j[0, 0], K_j[1, 1]
        cx_j, cy_j = K_j[0, 2], K_j[1, 2]
        denom = np.maximum(P_camj[:, 2], 1e-6)
        u_j = P_camj[:, 0] / denom * fx_j + cx_j
        v_j = P_camj[:, 1] / denom * fy_j + cy_j

        u_j_orig = u_j / sx
        v_j_orig = v_j / sy

        reproj_tracks_orig[j, :, 0] = u_j_orig
        reproj_tracks_orig[j, :, 1] = v_j_orig

        gt_u = gt_tracks_orig[j, valid_ids, 0]
        gt_v = gt_tracks_orig[j, valid_ids, 1]
        gt_vis_j = gt_vis_mask[j, valid_ids]

        mask = gt_vis_j & depth_valid & in_front
        err = np.sqrt((u_j_orig - gt_u) ** 2 + (v_j_orig - gt_v) ** 2)
        all_errors.append(err[mask])

    errors = np.concatenate(all_errors) if all_errors else np.array([])

    if len(errors) == 0:
        return None, None, None, None, None

    metrics = {
        "reproj_ate": float(errors.mean()),
        "reproj_median": float(np.median(errors)),
        "delta_1px": float((errors < 1.0).mean()),
        "delta_2px": float((errors < 2.0).mean()),
        "delta_5px": float((errors < 5.0).mean()),
        "delta_10px": float((errors < 10.0).mean()),
        "n_points": int(N_use),
        "n_valid_depth": int(depth_valid.sum()),
    }

    return metrics, P_world, depth_valid, reproj_tracks_orig, valid_ids


# ---------------------------------------------------------------------------
# COLMAP export
# ---------------------------------------------------------------------------

def _rot_to_colmap_quat(R):
    """Rotation matrix (3x3) -> COLMAP quaternion [QW, QX, QY, QZ]."""
    try:
        from scipy.spatial.transform import Rotation
        q = Rotation.from_matrix(R).as_quat()  # [qx, qy, qz, qw]
        return np.array([q[3], q[0], q[1], q[2]])
    except ImportError:
        pass

    m = R
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def write_colmap(
    out_dir, image_paths,
    extrinsics, intrinsics,
    W_orig, H_orig, W_model, H_model,
    pts3d_world=None, depth_valid=None,
    gt_tracks_orig=None, gt_vis_mask=None,
):
    """Write COLMAP sparse reconstruction to out_dir/sparse/0/."""
    sparse_dir = os.path.join(out_dir, "sparse", "0")
    os.makedirs(sparse_dir, exist_ok=True)

    S = len(image_paths)
    sx = W_model / W_orig
    sy = H_model / H_orig

    fx_orig = float(intrinsics[:, 0, 0].mean()) / sx
    fy_orig = float(intrinsics[:, 1, 1].mean()) / sy
    cx_orig = W_orig / 2.0
    cy_orig = H_orig / 2.0

    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_path, "w") as fh:
        fh.write("# Camera list with one line of data per camera:\n")
        fh.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        fh.write(
            f"1 PINHOLE {W_orig} {H_orig} "
            f"{fx_orig:.6f} {fy_orig:.6f} "
            f"{cx_orig:.6f} {cy_orig:.6f}\n"
        )

    have_pts = (
        pts3d_world is not None
        and depth_valid is not None
        and gt_tracks_orig is not None
    )

    if have_pts:
        valid_pt_idx = np.where(depth_valid)[0]
        pt3d_id_map = {
            int(i): int(k + 1) for k, i in enumerate(valid_pt_idx)
        }
    else:
        pt3d_id_map = {}
        valid_pt_idx = np.array([], dtype=int)

    images_path = os.path.join(sparse_dir, "images.txt")
    with open(images_path, "w") as fh:
        fh.write("# Image list with two lines of data per image:\n")
        fh.write(
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )
        fh.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, img_path in enumerate(image_paths):
            img_name = os.path.basename(img_path)
            R = extrinsics[i, :3, :3]
            T = extrinsics[i, :3, 3]
            qw, qx, qy, qz = _rot_to_colmap_quat(R)
            tx, ty, tz = T
            fh.write(
                f"{i+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{tx:.9f} {ty:.9f} {tz:.9f} 1 {img_name}\n"
            )
            if have_pts and gt_vis_mask is not None:
                parts = []
                for k, pt_idx in enumerate(valid_pt_idx):
                    if gt_vis_mask[i, pt_idx]:
                        u = float(gt_tracks_orig[i, pt_idx, 0])
                        v = float(gt_tracks_orig[i, pt_idx, 1])
                        parts.append(
                            f"{u:.3f} {v:.3f} {pt3d_id_map[int(pt_idx)]}"
                        )
                fh.write(" ".join(parts) + "\n" if parts else "\n")
            else:
                fh.write("\n")

    points_path = os.path.join(sparse_dir, "points3D.txt")
    with open(points_path, "w") as fh:
        fh.write("# 3D point list with one line of data per point:\n")
        fh.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )
        if have_pts:
            for k, pt_idx in enumerate(valid_pt_idx):
                pt3d_id = pt3d_id_map[int(pt_idx)]
                X, Y, Z = pts3d_world[pt_idx]
                track_parts = []
                if gt_vis_mask is not None:
                    for i in range(S):
                        if gt_vis_mask[i, pt_idx]:
                            track_parts.append(f"{i+1} {k}")
                track_str = " ".join(track_parts)
                fh.write(
                    f"{pt3d_id} {X:.6f} {Y:.6f} {Z:.6f} "
                    f"200 200 0 0.0 {track_str}\n"
                )

    print(
        f"    [colmap] -> {sparse_dir} "
        f"({S} images, {len(valid_pt_idx)} 3D points)"
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_GT_COLOR = (0, 220, 0)
_VAN_COLOR = (220, 60, 60)
_FT_COLOR = (60, 130, 255)

_SHIFT = 4
_S16 = 1 << _SHIFT  # 16 subpixel precision


def _render_frame_reproj(
    image_path, fi, gt_tracks_orig, gt_vis_mask,
    reproj_by_tag, sampled_ids, sx, sy, W_disp, H_disp,
):
    """
    One video frame: left=vanilla reproj vs GT, right=finetuned vs GT.
    Subpixel-accurate cv2 circle rendering (shift=4).
    """
    import cv2

    bg = np.array(
        Image.open(image_path).convert("RGB").resize(
            (W_disp, H_disp), Image.BILINEAR
        )
    )
    left = bg.copy()
    right = bg.copy()

    dot_r_gt = 4
    dot_r_pred = 4

    for tid in sampled_ids:
        if gt_vis_mask[fi, tid]:
            gx = int(float(gt_tracks_orig[fi, tid, 0]) * sx * _S16)
            gy = int(float(gt_tracks_orig[fi, tid, 1]) * sy * _S16)
            cv2.circle(
                left, (gx, gy), dot_r_gt * _S16,
                _GT_COLOR[::-1], -1, cv2.LINE_AA, _SHIFT,
            )
            cv2.circle(
                right, (gx, gy), dot_r_gt * _S16,
                _GT_COLOR[::-1], -1, cv2.LINE_AA, _SHIFT,
            )

        if "vanilla" in reproj_by_tag:
            van_reproj, van_vids = reproj_by_tag["vanilla"]
            loc = np.where(van_vids == tid)[0]
            if len(loc):
                vx = int(float(van_reproj[fi, loc[0], 0]) * sx * _S16)
                vy = int(float(van_reproj[fi, loc[0], 1]) * sy * _S16)
                cv2.circle(
                    left, (vx, vy), dot_r_pred * _S16,
                    _VAN_COLOR[::-1], 2, cv2.LINE_AA, _SHIFT,
                )

        if "finetuned" in reproj_by_tag:
            ft_reproj, ft_vids = reproj_by_tag["finetuned"]
            loc = np.where(ft_vids == tid)[0]
            if len(loc):
                fx_ = int(float(ft_reproj[fi, loc[0], 0]) * sx * _S16)
                fy_ = int(float(ft_reproj[fi, loc[0], 1]) * sy * _S16)
                cv2.circle(
                    right, (fx_, fy_), dot_r_pred * _S16,
                    _FT_COLOR[::-1], 2, cv2.LINE_AA, _SHIFT,
                )

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        left, "Vanilla (reproj)", (8, 22),
        font, 0.7, _VAN_COLOR[::-1], 2, cv2.LINE_AA,
    )
    cv2.putText(
        right, "Finetuned (reproj)", (8, 22),
        font, 0.7, _FT_COLOR[::-1], 2, cv2.LINE_AA,
    )
    label = f"frame {fi}"
    cv2.putText(
        left, label, (8, H_disp - 8), font, 0.5, (200, 200, 200), 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        right, label, (8, H_disp - 8), font, 0.5, (200, 200, 200), 1,
        cv2.LINE_AA,
    )

    bar_h = 20
    for panel in (left, right):
        panel[-bar_h:] = (20, 20, 20)
    cv2.circle(left[-bar_h:], (10, bar_h // 2), 5, _GT_COLOR[::-1], -1)
    cv2.putText(
        left[-bar_h:], "GT", (18, bar_h - 5),
        font, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.circle(left[-bar_h:], (50, bar_h // 2), 5, _VAN_COLOR[::-1], 2)
    cv2.putText(
        left[-bar_h:], "Reproj", (58, bar_h - 5),
        font, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.circle(right[-bar_h:], (10, bar_h // 2), 5, _GT_COLOR[::-1], -1)
    cv2.putText(
        right[-bar_h:], "GT", (18, bar_h - 5),
        font, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.circle(right[-bar_h:], (50, bar_h // 2), 5, _FT_COLOR[::-1], 2)
    cv2.putText(
        right[-bar_h:], "Reproj", (58, bar_h - 5),
        font, 0.4, (220, 220, 220), 1, cv2.LINE_AA,
    )

    divider = np.full((H_disp, 2, 3), 180, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def visualize_sequence_reproj(
    seq_name, image_paths, gt_tracks_orig, gt_vis_mask,
    reproj_by_tag, split_name, vis_dir, n_tracks, fps=8,
):
    """Save a side-by-side reprojection comparison video."""
    try:
        import cv2
    except ImportError:
        print("    [viz] SKIP -- opencv not installed")
        return

    S = len(image_paths)
    if S < 2:
        return

    vis0 = gt_vis_mask[0]
    all_valid = set(np.where(vis0)[0].tolist())
    for _, (_, vids) in reproj_by_tag.items():
        all_valid &= set(vids.tolist())
    valid_ids = np.array(sorted(all_valid))

    if len(valid_ids) == 0:
        return

    rng = np.random.RandomState(42)
    sampled_ids = rng.choice(
        valid_ids, size=min(n_tracks, len(valid_ids)), replace=False
    )

    W_orig, H_orig, W_disp, H_disp = get_model_resolution(image_paths[0])
    sx = W_disp / W_orig
    sy = H_disp / H_orig
    W_disp = W_disp + (W_disp % 2)
    H_disp = H_disp + (H_disp % 2)

    out_dir = os.path.join(vis_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq_name}.mp4")

    tmp_dir = tempfile.mkdtemp(prefix="reproj_viz_")
    try:
        for fi in range(S):
            frame_rgb = _render_frame_reproj(
                image_paths[fi], fi,
                gt_tracks_orig, gt_vis_mask,
                reproj_by_tag, sampled_ids,
                sx, sy, W_disp, H_disp,
            )
            frame_path = os.path.join(tmp_dir, f"{fi:06d}.jpg")
            cv2.imwrite(
                frame_path, frame_rgb[:, :, ::-1],
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmp_dir, "%06d.jpg"),
            "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-pix_fmt", "yuv420p",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    [viz] ffmpeg failed (code {result.returncode}):")
            print(result.stderr[-600:])
            return
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"    [viz] -> {out_path}  ({S} frames @ {fps}fps)")


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "reproj_ate", "reproj_median",
    "delta_1px", "delta_2px", "delta_5px", "delta_10px",
]


def eval_sequence_3d(
    seq_name, dataset_dir, models,
    track_num, device, dtype,
    colmap_dir=None,
    vis_n_tracks=50, vis_fps=8, vis_dir=None, split_name="val",
):
    """
    Run 3D reprojection eval for one sequence across all model tags.

    Returns:
        (seq_results, image_paths, gt_tracks_orig, gt_vis_mask, reproj_by_tag)
    """
    seq_dir = os.path.join(dataset_dir, seq_name)
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
    masks_path = os.path.join(seq_dir, "track_masks.npy")
    if not os.path.isfile(tracks_path):
        print(f"  [{seq_name}] SKIP -- no tracks.npy")
        return _skip

    gt_tracks_orig = np.load(tracks_path).astype(np.float32)
    if os.path.isfile(masks_path):
        gt_vis_mask = np.load(masks_path).astype(bool)
    else:
        gt_vis_mask = np.ones(gt_tracks_orig.shape[:2], dtype=bool)

    S = len(image_paths)
    gt_tracks_orig = gt_tracks_orig[:S]
    gt_vis_mask = gt_vis_mask[:S]

    W_orig, H_orig, W_model, H_model = get_model_resolution(image_paths[0])
    images_tensor = load_and_preprocess_images(image_paths, mode="crop")

    seq_results = {}
    reproj_by_tag = {}

    for tag, model in models.items():
        extrinsics, intrinsics, pred_depth = run_camera_and_depth(
            model, images_tensor, H_model, W_model, device, dtype
        )
        out = compute_reproj_metrics(
            gt_tracks_orig, gt_vis_mask,
            extrinsics, intrinsics,
            pred_depth, W_orig, H_orig, W_model, H_model,
            track_num=track_num,
        )
        metrics, pts3d_world, depth_valid, reproj_tracks, valid_ids = out

        if metrics is None:
            print(f"  [{seq_name}][{tag}] SKIP -- no valid points")
            continue

        seq_results[tag] = metrics
        reproj_by_tag[tag] = (reproj_tracks, valid_ids)

        if colmap_dir is not None:
            write_colmap(
                os.path.join(colmap_dir, seq_name, tag),
                image_paths, extrinsics, intrinsics,
                W_orig, H_orig, W_model, H_model,
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

    if vis_dir is not None and len(reproj_by_tag) > 0:
        visualize_sequence_reproj(
            seq_name, image_paths, gt_tracks_orig, gt_vis_mask,
            reproj_by_tag, split_name, vis_dir,
            n_tracks=vis_n_tracks, fps=vis_fps,
        )

    return (
        seq_results if seq_results else None,
        image_paths, gt_tracks_orig, gt_vis_mask, reproj_by_tag,
    )


# ---------------------------------------------------------------------------
# Split-level evaluation
# ---------------------------------------------------------------------------

def mean_over_seqs(seq_results, tag):
    vals = {k: [] for k in METRIC_KEYS}
    for sr in seq_results.values():
        if sr is None or tag not in sr:
            continue
        m = sr[tag]
        for k in METRIC_KEYS:
            if k in m:
                vals[k].append(m[k])
    return {
        k: float(np.mean(v)) if v else float("nan")
        for k, v in vals.items()
    }


def eval_split(
    split_name, sequences, dataset_dir, models,
    track_num, device, dtype, colmap_dir=None,
    vis_max_seqs=3, vis_n_tracks=50, vis_fps=8, vis_dir=None,
):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"SPLIT: {split_name.upper()}  ({len(sequences)} sequences)")
    print(sep)

    seq_results = {}
    viz_count = 0
    for seq_name in sequences:
        do_viz = vis_dir is not None and viz_count < vis_max_seqs
        r, *_ = eval_sequence_3d(
            seq_name, dataset_dir, models,
            track_num, device, dtype,
            colmap_dir=colmap_dir,
            vis_n_tracks=vis_n_tracks,
            vis_fps=vis_fps,
            vis_dir=vis_dir if do_viz else None,
            split_name=split_name,
        )
        seq_results[seq_name] = r
        if r is not None and do_viz:
            viz_count += 1

    tags = list(models.keys())
    means = {tag: mean_over_seqs(seq_results, tag) for tag in tags}

    print(f"\n--- {split_name.upper()} MEAN METRICS ---")
    for tag in tags:
        m = means[tag]
        print(
            f"  [{tag}]"
            f"  reproj_ATE={m['reproj_ate']:.3f}px"
            f"  median={m['reproj_median']:.3f}px"
            f"  d1={m['delta_1px']:.3f}"
            f"  d2={m['delta_2px']:.3f}"
            f"  d5={m['delta_5px']:.3f}"
            f"  d10={m['delta_10px']:.3f}"
        )

    if len(tags) == 2:
        v, f = tags[0], tags[1]
        mv, mf = means[v], means[f]
        diff = mf["reproj_ate"] - mv["reproj_ate"]
        pct = diff / (mv["reproj_ate"] + 1e-9) * 100
        diff5 = mf["delta_5px"] - mv["delta_5px"]
        print(f"\n  IMPROVEMENT {v} -> {f}:")
        print(
            f"    reproj_ATE: {mv['reproj_ate']:.2f} -> "
            f"{mf['reproj_ate']:.2f} px ({diff:+.2f}px, {pct:+.1f}%)"
        )
        print(
            f"    d5:         {mv['delta_5px']:.3f} -> "
            f"{mf['delta_5px']:.3f} ({diff5:+.3f})"
        )

    return {"per_seq": seq_results, "mean": means}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("\nLoading vanilla model ...")
    vanilla_model = load_model(
        args.vanilla_ckpt, lora=False, device=device
    )
    print("Loading fine-tuned model ...")
    finetuned_model = load_model(
        args.finetuned_ckpt,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_targets=args.lora_target_modules,
        device=device,
    )
    models = {"vanilla": vanilla_model, "finetuned": finetuned_model}

    def read_split(path):
        with open(path) as fh:
            return [line.strip() for line in fh if line.strip()]

    all_results = {}

    val_seqs = read_split(args.val_split_file)
    all_results["val"] = eval_split(
        "val", val_seqs, args.dataset_dir,
        models, args.track_num, device, dtype,
        colmap_dir=(
            os.path.join(args.colmap_dir, "val")
            if args.colmap_dir else None
        ),
        vis_max_seqs=args.vis_max_seqs,
        vis_n_tracks=args.vis_n_tracks,
        vis_fps=args.vis_fps,
        vis_dir=args.vis_dir,
    )

    if args.train_split_file:
        train_seqs = read_split(args.train_split_file)
        if 0 < args.train_max_seqs < len(train_seqs):
            random.seed(0)
            train_seqs = random.sample(train_seqs, args.train_max_seqs)
            print(f"\n(train subsampled to {len(train_seqs)} sequences)")
        all_results["train"] = eval_split(
            "train", train_seqs, args.dataset_dir,
            models, args.track_num, device, dtype,
            colmap_dir=(
                os.path.join(args.colmap_dir, "train")
                if args.colmap_dir else None
            ),
            vis_max_seqs=args.vis_max_seqs,
            vis_n_tracks=args.vis_n_tracks,
            vis_fps=args.vis_fps,
            vis_dir=args.vis_dir,
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

    if args.colmap_dir:
        print(f"COLMAP reconstructions saved to {args.colmap_dir}/")
    if args.vis_max_seqs > 0:
        print(f"Videos saved to {args.vis_dir}/")


if __name__ == "__main__":
    main()
