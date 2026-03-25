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

Usage:
    python eval_track_head.py \\
        --vanilla-ckpt     /workspace/model.pt \\
        --finetuned-ckpt   /workspace/ckpts/checkpoint.pt \\
        --dataset-dir      /mnt/bucket/.../tracking_whisper_sample_dataset \\
        --train-split-file /mnt/bucket/.../train_split.txt \\
        --val-split-file   /mnt/bucket/.../val_split.txt \\
        [--train-max-seqs 20] \\
        [--lora] [--lora-r 16] [--lora-alpha 32] \\
        [--track-chunk 256] \\
        [--output-json track_eval_results.json]
"""

import argparse
import glob
import json
import os
import random

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
        "--train-split-file", required=True,
        help="Train split: one sequence name per line",
    )
    p.add_argument(
        "--val-split-file", required=True,
        help="Val split: one sequence name per line",
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
    p.add_argument("--output-json", default="track_eval_results.json")
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

def get_model_resolution(image_path):
    """
    Return (W_orig, H_orig, W_model, H_model).
    crop mode: width=518, height=round(H*518/W/14)*14.
    """
    img = Image.open(image_path)
    W_orig, H_orig = img.size
    W_model = 518
    H_model = round(H_orig * (W_model / W_orig) / 14) * 14
    return W_orig, H_orig, W_model, H_model


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

    Frame 0 is the query frame and is skipped from error computation.
    Returns dict or None if no visible GT points in frames 1..S.
    """
    gt_t = gt_tracks_orig[1:]    # (S-1, N, 2)
    pred_t = pred_tracks_orig[1:]
    gt_mask = gt_vis_mask[1:]    # (S-1, N) bool
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
# Per-sequence evaluation
# ---------------------------------------------------------------------------

def eval_sequence(
    seq_name, dataset_dir, model, chunk_size, vis_thresh, device, dtype
):
    seq_dir = os.path.join(dataset_dir, seq_name)
    image_dir = os.path.join(seq_dir, "images")

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not image_paths:
        return None, f"no images in {image_dir}"

    tracks_path = os.path.join(seq_dir, "tracks.npy")
    masks_path = os.path.join(seq_dir, "track_masks.npy")
    if not os.path.isfile(tracks_path):
        return None, "no tracks.npy"

    # GT tracks in original pixel space: (N_frames_total, N_tracks, 2)
    gt_tracks_orig = np.load(tracks_path).astype(np.float32)
    if os.path.isfile(masks_path):
        gt_vis_mask = np.load(masks_path).astype(bool)
    else:
        gt_vis_mask = np.ones(gt_tracks_orig.shape[:2], dtype=bool)

    S = len(image_paths)
    if gt_tracks_orig.shape[0] < S:
        return None, (
            f"tracks.npy has {gt_tracks_orig.shape[0]} frames "
            f"but {S} images found"
        )

    gt_tracks_orig = gt_tracks_orig[:S]
    gt_vis_mask = gt_vis_mask[:S]

    if S < 2:
        return None, "need at least 2 frames"

    W_orig, H_orig, W_model, H_model = get_model_resolution(image_paths[0])
    scale_x = W_model / W_orig
    scale_y = H_model / H_orig

    # Load images at model resolution (16:9 crop)
    images_tensor = load_and_preprocess_images(
        image_paths, mode="crop"
    )  # (S, 3, H_model, W_model)

    # Frame-0 GT positions scaled to model space as query points
    q_orig = gt_tracks_orig[0].copy()  # (N, 2)
    q_model = np.stack(
        [q_orig[:, 0] * scale_x, q_orig[:, 1] * scale_y], axis=-1
    ).astype(np.float32)

    try:
        pred_model, pred_vis = run_track_head(
            model, images_tensor, q_model, chunk_size, device, dtype
        )
    except RuntimeError as exc:
        return None, f"runtime error: {exc}"

    # Scale predictions back to original space
    pred_orig = np.stack(
        [pred_model[:, :, 0] / scale_x, pred_model[:, :, 1] / scale_y],
        axis=-1,
    )  # (S, N, 2)

    metrics = compute_metrics(
        pred_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh
    )
    return metrics, None


# ---------------------------------------------------------------------------
# Split-level evaluation
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "ate", "median_te",
    "delta_1px", "delta_2px", "delta_5px", "delta_10px",
    "vis_acc",
]


def mean_over_seqs(seq_results):
    vals = {k: [] for k in METRIC_KEYS}
    for m in seq_results.values():
        for k in METRIC_KEYS:
            vals[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in vals.items()}


def eval_split(
    split_name, sequences, dataset_dir, models,
    chunk_size, vis_thresh, device, dtype,
):
    """
    Evaluate all sequences in one split for every model in `models`.

    models: dict of tag -> model
    Returns: dict  (includes per-seq results and '<tag>_mean' aggregates)
    """
    results = {tag: {} for tag in models}

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"SPLIT: {split_name.upper()} ({len(sequences)} sequences)")
    print(sep)

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
                f"d1={metrics['delta_1px']:.3f}  "
                f"d2={metrics['delta_2px']:.3f}  "
                f"d5={metrics['delta_5px']:.3f}  "
                f"vis={metrics['vis_acc']:.3f}  "
                f"N={metrics['n_points']}"
            )

    # Aggregate stats + vanilla vs finetuned comparison
    print(f"\n  --- {split_name.upper()} aggregate ---")
    agg_by_tag = {}
    for tag in models:
        seqs = results[tag]
        if not seqs:
            print(f"  {tag.upper()}: no sequences evaluated")
            continue
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

    if "vanilla" in agg_by_tag and "finetuned" in agg_by_tag:
        v = agg_by_tag["vanilla"]
        ft = agg_by_tag["finetuned"]
        ate_d = ft["ate"] - v["ate"]
        d5_d = ft["delta_5px"] - v["delta_5px"]
        pct = ate_d / v["ate"] * 100
        print(f"\n  IMPROVEMENT vanilla -> finetuned:")
        print(
            f"    ATE:     {v['ate']:.2f} -> {ft['ate']:.2f} px  "
            f"({ate_d:+.2f}px, {pct:+.1f}%)"
        )
        print(
            f"    d5:      {v['delta_5px']:.3f} -> {ft['delta_5px']:.3f}  "
            f"({d5_d * 100:+.1f}pp)"
        )
        print(f"    vis_acc: {v['vis_acc']:.3f} -> {ft['vis_acc']:.3f}")

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

    train_seqs = load_sequences(args.train_split_file)
    val_seqs = load_sequences(args.val_split_file)

    # Subsample train split
    max_tr = args.train_max_seqs
    n_total = len(train_seqs)
    if max_tr > 0 and n_total > max_tr:
        train_seqs = random.sample(train_seqs, max_tr)
        print(f"Train split: sampled {max_tr} / {n_total} sequences")
    else:
        print(f"Train split: {n_total} sequences")
    print(f"Val   split: {len(val_seqs)} sequences")

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
    for split_name, seqs in [("train", train_seqs), ("val", val_seqs)]:
        all_results[split_name] = eval_split(
            split_name, seqs, args.dataset_dir, models,
            args.track_chunk, args.vis_thresh, device, dtype,
        )

    # Cross-split summary table
    sep = "=" * 70
    print(f"\n{sep}")
    print("SUMMARY  (train vs val splits — check for overfitting)")
    print(sep)
    for tag in ("vanilla", "finetuned"):
        for split_name in ("train", "val"):
            key = f"{tag}_mean"
            agg = all_results.get(split_name, {}).get(key)
            if agg is None:
                continue
            print(
                f"  {tag:10s} {split_name:5s}  "
                f"ATE={agg['ate']:.2f}px  "
                f"d5={agg['delta_5px']:.3f}  "
                f"vis={agg['vis_acc']:.3f}"
            )

    with open(args.output_json, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
