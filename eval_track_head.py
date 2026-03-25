#!/usr/bin/env python3
"""
eval_track_head.py — Evaluate VGGT track head against GT AE tracks.

Tests the model's raw feedforward track head (no BA, no post-processing).
Compares vanilla VGGT vs fine-tuned checkpoint.

Protocol:
  - Images loaded at 16:9 aspect ratio (crop mode) to match fine-tuning distribution.
  - GT frame-0 track positions used as query points.
  - Predicted track positions for frames 1..S compared against GT on visible frames only.
  - Aggregator runs once per sequence; feature maps cached; tracker runs per chunk (OOM safe).

Metrics (reported in original image resolution):
  - ATE:        mean L2 pixel error over all visible GT track points
  - median_te:  median L2 pixel error
  - delta_Npx:  fraction of predictions within N pixels of GT (1, 2, 5, 10)
  - vis_acc:    visibility prediction accuracy vs GT track_masks

Usage:
    python eval_track_head.py \
        --vanilla-ckpt    /workspace/model.pt \
        --finetuned-ckpt  /workspace/ckpts/checkpoint.pt \
        --dataset-dir     /mnt/bucket/.../tracking_whisper_sample_dataset \
        --train-split-file /mnt/bucket/.../train_split.txt \
        --val-split-file   /mnt/bucket/.../val_split.txt \
        [--lora] [--lora-r 16] [--lora-alpha 32] \
        [--track-chunk    256] \
        [--output-json    track_eval_results.json]
"""

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VGGT track head vs GT AE tracks")
    p.add_argument("--vanilla-ckpt",          required=True,  help="Path to vanilla VGGT checkpoint")
    p.add_argument("--finetuned-ckpt",        required=True,  help="Path to fine-tuned checkpoint")
    p.add_argument("--dataset-dir",           required=True,  help="Root dataset directory")
    p.add_argument("--train-split-file",      required=True,  help="Train split: one sequence name per line")
    p.add_argument("--val-split-file",        required=True,  help="Val split: one sequence name per line")
    p.add_argument("--lora",                  action="store_true", default=False,
                   help="Fine-tuned ckpt is LoRA (wrap aggregator with PEFT before loading)")
    p.add_argument("--lora-r",                type=int,   default=16)
    p.add_argument("--lora-alpha",            type=float, default=32.0)
    p.add_argument("--lora-target-modules",   type=str,   default="qkv,proj")
    p.add_argument("--track-chunk",           type=int,   default=256,
                   help="Number of tracks per tracker forward pass (reduce if OOM)")
    p.add_argument("--vis-thresh",            type=float, default=0.5,
                   help="Threshold on predicted vis score to call a point visible")
    p.add_argument("--output-json",           default="track_eval_results.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path, lora=False, lora_r=16, lora_alpha=32.0,
               lora_targets="qkv,proj", device="cuda"):
    """Load VGGT with optional LoRA wrapping."""
    # point_head not needed for track eval — skip to save memory
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

    ckpt  = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys (expected for LoRA or partial ckpt)")

    model.eval()
    return model.to(device)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def get_model_resolution(image_path):
    """
    Return (W_orig, H_orig, W_model, H_model) for load_and_preprocess_images(mode='crop').
    crop mode: sets width = 518, height = round(H * 518/W / 14) * 14.
    """
    img      = Image.open(image_path)
    W_orig, H_orig = img.size
    W_model  = 518
    H_model  = round(H_orig * (W_model / W_orig) / 14) * 14
    return W_orig, H_orig, W_model, H_model


# ---------------------------------------------------------------------------
# Track head inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_track_head(model, images_tensor, query_points_np, chunk_size, device, dtype):
    """
    Run the VGGT track head for all query tracks, chunked to avoid OOM.

    Args:
        model:             VGGT model on device
        images_tensor:     (S, 3, H, W) float32 [0,1] on CPU
        query_points_np:   (N, 2) float32 in MODEL pixel coords (frame-0 positions)
        chunk_size:        max tracks per forward pass
        device, dtype:     inference device/dtype

    Returns:
        pred_tracks: (S, N, 2) float32 in model pixel coords
        pred_vis:    (S, N)    float32  [0,1]
    """
    S = images_tensor.shape[0]
    N = query_points_np.shape[0]

    # Add batch dim: (1, S, 3, H, W)
    images_batch = images_tensor.unsqueeze(0).to(device)

    # Run aggregator once — expensive, shared across all chunks
    with torch.cuda.amp.autocast(dtype=dtype):
        agg_tokens, patch_start_idx = model.aggregator(images_batch)
        # Pre-compute feature maps once — avoids re-running DPTHead per chunk
        feature_maps = model.track_head.feature_extractor(agg_tokens, images_batch, patch_start_idx)
        # feature_maps: (1, S, C, H//2, W//2)

    all_tracks = []
    all_vis    = []

    query_tensor = torch.from_numpy(query_points_np)  # (N, 2)

    for start in range(0, N, chunk_size):
        end   = min(start + chunk_size, N)
        qpts  = query_tensor[start:end].unsqueeze(0).to(device)  # (1, chunk, 2)

        with torch.cuda.amp.autocast(dtype=dtype):
            coord_preds, vis, conf = model.track_head.tracker(
                query_points=qpts,
                fmaps=feature_maps,
                iters=model.track_head.iters,
            )

        # coord_preds: list of (1, S, chunk, 2) per iteration — take last
        pred_t = coord_preds[-1].squeeze(0).cpu().float()  # (S, chunk, 2)
        pred_v = vis.squeeze(0).cpu().float()               # (S, chunk)

        all_tracks.append(pred_t)
        all_vis.append(pred_v)

    pred_tracks = torch.cat(all_tracks, dim=1).numpy()  # (S, N, 2)
    pred_vis    = torch.cat(all_vis,    dim=1).numpy()   # (S, N)

    return pred_tracks, pred_vis


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_tracks_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh):
    """
    All coordinates in original image resolution.

    pred_tracks_orig: (S, N, 2)
    gt_tracks_orig:   (S, N, 2)
    gt_vis_mask:      (S, N) bool   — True where GT track is visible
    pred_vis:         (S, N) float  — model's visibility score
    vis_thresh:       float

    Frame 0 is the query frame — skipped from error computation.
    Returns dict or None if no visible points.
    """
    # Skip frame 0 (query)
    gt_t    = gt_tracks_orig[1:]   # (S-1, N, 2)
    pred_t  = pred_tracks_orig[1:] # (S-1, N, 2)
    gt_mask = gt_vis_mask[1:]      # (S-1, N) bool
    pred_v  = pred_vis[1:]         # (S-1, N) float

    # Per-point L2 in original pixel space
    diff     = pred_t - gt_t                          # (S-1, N, 2)
    l2       = np.sqrt((diff ** 2).sum(-1))           # (S-1, N)
    valid_l2 = l2[gt_mask]

    if len(valid_l2) == 0:
        return None

    # Visibility accuracy
    pred_vis_bool = pred_v > vis_thresh
    vis_acc = float((pred_vis_bool == gt_mask).mean())

    return {
        "ate":         float(valid_l2.mean()),
        "median_te":   float(np.median(valid_l2)),
        "delta_1px":   float((valid_l2 < 1.0).mean()),
        "delta_2px":   float((valid_l2 < 2.0).mean()),
        "delta_5px":   float((valid_l2 < 5.0).mean()),
        "delta_10px":  float((valid_l2 < 10.0).mean()),
        "vis_acc":     vis_acc,
        "n_points":    int(len(valid_l2)),
        "n_tracks":    int(gt_vis_mask.shape[1]),
        "n_frames":    int(gt_vis_mask.shape[0]),
    }


# ---------------------------------------------------------------------------
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def eval_sequence(seq_name, dataset_dir, model, chunk_size, vis_thresh, device, dtype):
    seq_dir   = os.path.join(dataset_dir, seq_name)
    image_dir = os.path.join(seq_dir, "images")

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not image_paths:
        return None, f"no images in {image_dir}"

    tracks_path = os.path.join(seq_dir, "tracks.npy")
    masks_path  = os.path.join(seq_dir, "track_masks.npy")
    if not os.path.isfile(tracks_path):
        return None, "no tracks.npy"

    # GT tracks in original pixel space: (N_frames_total, N_tracks, 2)
    gt_tracks_orig = np.load(tracks_path).astype(np.float32)
    gt_vis_mask    = (np.load(masks_path).astype(bool)
                      if os.path.isfile(masks_path)
                      else np.ones(gt_tracks_orig.shape[:2], dtype=bool))

    S_total = gt_tracks_orig.shape[0]
    S       = len(image_paths)

    if S_total < S:
        return None, f"tracks.npy has {S_total} frames but {S} images found"

    # Align: use only the first S frames (sorted image list = frame order)
    gt_tracks_orig = gt_tracks_orig[:S]   # (S, N, 2)
    gt_vis_mask    = gt_vis_mask[:S]      # (S, N)

    # Need at least 2 frames to evaluate (frame 0 = query, rest = eval)
    if S < 2:
        return None, "need at least 2 frames"

    # --- Resolution mapping ---
    W_orig, H_orig, W_model, H_model = get_model_resolution(image_paths[0])
    scale_x = W_model / W_orig
    scale_y = H_model / H_orig

    # --- Load images at model resolution (16:9 crop) ---
    images_tensor = load_and_preprocess_images(image_paths, mode="crop")  # (S, 3, H_model, W_model)

    # --- Query points: frame-0 GT positions scaled to model space ---
    q_orig  = gt_tracks_orig[0].copy()                 # (N, 2) original coords
    q_model = np.stack([q_orig[:, 0] * scale_x,
                        q_orig[:, 1] * scale_y], axis=-1).astype(np.float32)  # (N, 2)

    # --- Run track head ---
    try:
        pred_tracks_model, pred_vis = run_track_head(
            model, images_tensor, q_model, chunk_size, device, dtype
        )
    except RuntimeError as e:
        return None, f"runtime error: {e}"

    # --- Scale predictions back to original space ---
    pred_tracks_orig = np.stack([
        pred_tracks_model[:, :, 0] / scale_x,
        pred_tracks_model[:, :, 1] / scale_y,
    ], axis=-1)  # (S, N, 2)

    metrics = compute_metrics(pred_tracks_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh)
    return metrics, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def mean_over_seqs(seq_results, keys):
    vals = {k: [] for k in keys}
    for m in seq_results.values():
        for k in keys:
            vals[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in vals.items()}


def eval_split(split_name, sequences, dataset_dir, models, chunk_size, vis_thresh, device, dtype):
    """
    Run evaluation for all sequences in one split.
    models: dict of tag -> model
    Returns: dict of tag -> {seq -> metrics}
    """
    metric_keys = ["ate", "median_te", "delta_1px", "delta_2px", "delta_5px", "delta_10px", "vis_acc"]
    results = {tag: {} for tag in models}

    print(f"\n{'='*70}")
    print(f"SPLIT: {split_name.upper()} ({len(sequences)} sequences)")
    print('='*70)

    for seq in sequences:
        print(f"\n  [{seq}]")
        for tag, model in models.items():
            metrics, err = eval_sequence(
                seq, dataset_dir, model,
                chunk_size, vis_thresh, device, dtype,
            )
            if err:
                print(f"    {tag:10s}  SKIP ({err})")
                continue
            results[tag][seq] = metrics
            print(
                f"    {tag:10s}  "
                f"ATE={metrics['ate']:6.2f}px  "
                f"med={metrics['median_te']:6.2f}px  "
                f"δ1={metrics['delta_1px']:.3f}  "
                f"δ2={metrics['delta_2px']:.3f}  "
                f"δ5={metrics['delta_5px']:.3f}  "
                f"vis={metrics['vis_acc']:.3f}  "
                f"N={metrics['n_points']}"
            )

    # Per-split aggregate + comparison
    print(f"\n  --- {split_name.upper()} aggregate ---")
    agg_by_tag = {}
    for tag in models:
        seqs = results[tag]
        if not seqs:
            print(f"  {tag.upper()}: no sequences evaluated")
            continue
        agg = mean_over_seqs(seqs, metric_keys)
        agg_by_tag[tag] = agg
        results[f"{tag}_mean"] = agg
        print(
            f"  {tag.upper()} ({len(seqs)} seqs)  "
            f"ATE={agg['ate']:.2f}px  "
            f"med={agg['median_te']:.2f}px  "
            f"δ1={agg['delta_1px']:.3f}  "
            f"δ2={agg['delta_2px']:.3f}  "
            f"δ5={agg['delta_5px']:.3f}  "
            f"vis={agg['vis_acc']:.3f}"
        )

    if "vanilla" in agg_by_tag and "finetuned" in agg_by_tag:
        v  = agg_by_tag["vanilla"]
        ft = agg_by_tag["finetuned"]
        ate_delta = ft["ate"] - v["ate"]
        d5_delta  = ft["delta_5px"] - v["delta_5px"]
        print(f"\n  IMPROVEMENT vanilla → finetuned:")
        print(f"    ATE:    {v['ate']:.2f} → {ft['ate']:.2f} px  ({ate_delta:+.2f}px, {ate_delta/v['ate']*100:+.1f}%)")
        print(f"    δ5:     {v['delta_5px']:.3f} → {ft['delta_5px']:.3f}  ({d5_delta*100:+.1f}pp)")
        print(f"    vis_acc:{v['vis_acc']:.3f} → {ft['vis_acc']:.3f}")

    return results


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def load_sequences(path):
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]

    train_seqs = load_sequences(args.train_split_file)
    val_seqs   = load_sequences(args.val_split_file)
    print(f"Train split: {args.train_split_file} — {len(train_seqs)} sequences")
    print(f"Val   split: {args.val_split_file}   — {len(val_seqs)} sequences")

    # Load models once, reuse for both splits
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
    for split_name, seqs in [("train", train_seqs), ("val", val_seqs)]:
        split_results = eval_split(
            split_name, seqs, args.dataset_dir, models,
            args.track_chunk, args.vis_thresh, device, dtype,
        )
        all_results[split_name] = split_results

    # Final cross-split summary
    print(f"\n{'='*70}")
    print("SUMMARY  (train vs val — checks for overfitting)")
    print('='*70)
    for tag in ("vanilla", "finetuned"):
        for split_name in ("train", "val"):
            key = f"{tag}_mean"
            if key in all_results.get(split_name, {}):
                agg = all_results[split_name][key]
                print(
                    f"  {tag:10s} {split_name:5s}  "
                    f"ATE={agg['ate']:.2f}px  "
                    f"δ5={agg['delta_5px']:.3f}  "
                    f"vis={agg['vis_acc']:.3f}"
                )

    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
