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

Visualization (--vis-max-seqs, default 3 per split):
  - One image per sequence saved to --vis-dir/{split}/{seq}.jpg
  - Columns = sampled frames; dots: GT=green, vanilla=red, finetuned=blue
  - Legend strip appended at bottom

Usage:
    python eval_track_head.py \\
        --vanilla-ckpt     /workspace/model.pt \\
        --finetuned-ckpt   /workspace/ckpts/checkpoint.pt \\
        --dataset-dir      /mnt/bucket/.../tracking_whisper_sample_dataset \\
        --train-split-file /mnt/bucket/.../train_split.txt \\
        --val-split-file   /mnt/bucket/.../val_split.txt \\
        [--train-max-seqs 20] \\
        [--vis-max-seqs 3] [--vis-n-tracks 50] [--vis-dir track_eval_viz] \\
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
from PIL import Image, ImageDraw, ImageFont

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
        "--vis-n-frames", type=int, default=6,
        help="Number of frames to show per sequence strip",
    )
    p.add_argument(
        "--vis-dir", default="track_eval_viz",
        help="Output directory for visualization images",
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
    seq_name, dataset_dir, model, chunk_size, vis_thresh, device, dtype
):
    """
    Returns: (metrics, error, preds)
      metrics: dict or None
      error:   str or None
      preds:   (pred_tracks_orig, pred_vis) in original pixel space, or None
    """
    seq_dir = os.path.join(dataset_dir, seq_name)
    image_dir = os.path.join(seq_dir, "images")

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not image_paths:
        return None, f"no images in {image_dir}", None

    tracks_path = os.path.join(seq_dir, "tracks.npy")
    masks_path = os.path.join(seq_dir, "track_masks.npy")
    if not os.path.isfile(tracks_path):
        return None, "no tracks.npy", None

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
        ), None

    gt_tracks_orig = gt_tracks_orig[:S]
    gt_vis_mask = gt_vis_mask[:S]

    if S < 2:
        return None, "need at least 2 frames", None

    W_orig, H_orig, W_model, H_model = get_model_resolution(image_paths[0])
    scale_x = W_model / W_orig
    scale_y = H_model / H_orig

    images_tensor = load_and_preprocess_images(image_paths, mode="crop")

    q_orig = gt_tracks_orig[0].copy()  # (N, 2)
    q_model = np.stack(
        [q_orig[:, 0] * scale_x, q_orig[:, 1] * scale_y], axis=-1
    ).astype(np.float32)

    try:
        pred_model, pred_vis = run_track_head(
            model, images_tensor, q_model, chunk_size, device, dtype
        )
    except RuntimeError as exc:
        return None, f"runtime error: {exc}", None

    # Scale predictions back to original space
    pred_orig = np.stack(
        [pred_model[:, :, 0] / scale_x, pred_model[:, :, 1] / scale_y],
        axis=-1,
    )

    metrics = compute_metrics(
        pred_orig, gt_tracks_orig, gt_vis_mask, pred_vis, vis_thresh
    )
    return metrics, None, (pred_orig, pred_vis)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Dot colors: (filled GT, outline vanilla, outline finetuned)
_GT_COLOR = (0, 220, 0)
_VAN_COLOR = (220, 60, 60)
_FT_COLOR = (60, 130, 255)
_LEGEND_BG = (20, 20, 20)


def _draw_dot(draw, x, y, r, color, filled=True):
    """Draw a circle; filled or outline only."""
    box = [x - r, y - r, x + r, y + r]
    if filled:
        draw.ellipse(box, fill=color)
    else:
        draw.ellipse(box, outline=color, width=2)


def _make_legend(width, font):
    """Return a small PIL image with the color legend."""
    h = 22
    img = Image.new("RGB", (width, h), _LEGEND_BG)
    draw = ImageDraw.Draw(img)
    items = [
        (_GT_COLOR,  True,  "GT (green)"),
        (_VAN_COLOR, False, "Vanilla (red)"),
        (_FT_COLOR,  False, "Finetuned (blue)"),
    ]
    x = 6
    for color, filled, label in items:
        _draw_dot(draw, x + 5, h // 2, 5, color, filled=filled)
        draw.text((x + 14, 4), label, fill=(220, 220, 220), font=font)
        x += 140
    return img


def visualize_sequence(
    seq_name, image_paths, gt_tracks_orig, gt_vis_mask,
    preds_by_tag, split_name, vis_dir, n_tracks, n_frames,
):
    """
    Save a frame-strip visualization for one sequence.

    preds_by_tag: dict  tag -> (pred_tracks_orig, pred_vis)
                  Expected tags: "vanilla", "finetuned" (either may be absent)
    Layout: columns = sampled frames (1..S-1)
            dots:  GT=green filled, vanilla=red outline, finetuned=blue outline
    """
    S = len(image_paths)
    if S < 2:
        return

    # Sample frames evenly from 1..S-1
    n_show = min(n_frames, S - 1)
    frame_ids = np.linspace(1, S - 1, n_show, dtype=int)

    # Sample tracks visible in frame 1
    gt_vis_f1 = gt_vis_mask[1] if S > 1 else gt_vis_mask[0]
    valid_ids = np.where(gt_vis_f1)[0]
    if len(valid_ids) == 0:
        return
    rng = np.random.RandomState(42)
    sampled_ids = rng.choice(
        valid_ids, size=min(n_tracks, len(valid_ids)), replace=False
    )

    # Load images and resize to display resolution
    W_orig, H_orig, W_disp, H_disp = get_model_resolution(image_paths[0])
    sx = W_disp / W_orig
    sy = H_disp / H_orig

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    dot_r_gt = 4    # GT: slightly larger filled dot
    dot_r_pred = 4  # predictions: outline circle same size

    cell_w, cell_h = W_disp, H_disp
    n_cols = len(frame_ids)

    legend = _make_legend(cell_w * n_cols, font)
    canvas = Image.new(
        "RGB", (cell_w * n_cols, cell_h + legend.height), _LEGEND_BG
    )
    canvas.paste(legend, (0, cell_h))

    for col, fi in enumerate(frame_ids):
        frame_img = Image.open(image_paths[fi]).convert("RGB")
        frame_img = frame_img.resize((W_disp, H_disp), Image.BILINEAR)
        draw = ImageDraw.Draw(frame_img)

        for tid in sampled_ids:
            # --- GT dot ---
            if gt_vis_mask[fi, tid]:
                gx = float(gt_tracks_orig[fi, tid, 0]) * sx
                gy = float(gt_tracks_orig[fi, tid, 1]) * sy
                _draw_dot(draw, gx, gy, dot_r_gt, _GT_COLOR, filled=True)

            # --- Vanilla prediction ---
            if "vanilla" in preds_by_tag:
                van_tracks, _ = preds_by_tag["vanilla"]
                vx = float(van_tracks[fi, tid, 0]) * sx
                vy = float(van_tracks[fi, tid, 1]) * sy
                _draw_dot(draw, vx, vy, dot_r_pred, _VAN_COLOR, filled=False)

            # --- Finetuned prediction ---
            if "finetuned" in preds_by_tag:
                ft_tracks, _ = preds_by_tag["finetuned"]
                fx = float(ft_tracks[fi, tid, 0]) * sx
                fy = float(ft_tracks[fi, tid, 1]) * sy
                _draw_dot(draw, fx, fy, dot_r_pred, _FT_COLOR, filled=False)

        # Frame label
        draw.text((4, 3), f"frame {fi}", fill=(255, 255, 255), font=font)

        canvas.paste(frame_img, (col * cell_w, 0))

    out_dir = os.path.join(vis_dir, split_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq_name}.jpg")
    canvas.save(out_path, quality=92)
    print(f"    [viz] -> {out_path}")


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
    vis_max_seqs=3, vis_n_tracks=50, vis_n_frames=6, vis_dir="track_eval_viz",
):
    """
    Evaluate all sequences in one split for every model in `models`.

    Loops sequence-first so predictions from all models are available
    together for visualization before moving to the next sequence.

    models: dict  tag -> model
    Returns: dict  (per-seq results + '<tag>_mean' aggregates)
    """
    results = {tag: {} for tag in models}
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

        for tag, model in models.items():
            metrics, err, preds = eval_sequence(
                seq, dataset_dir, model,
                chunk_size, vis_thresh, device, dtype,
            )
            if err:
                print(f"    {tag:10s}  SKIP ({err})")
                continue
            results[tag][seq] = metrics
            if preds is not None:
                preds_by_tag[tag] = preds
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

        # Visualization for first vis_max_seqs sequences in this split
        if (
            vis_max_seqs > 0
            and viz_count < vis_max_seqs
            and preds_by_tag
            and gt_tracks_orig is not None
            and image_paths
        ):
            visualize_sequence(
                seq, image_paths,
                gt_tracks_orig, gt_vis_mask,
                preds_by_tag, split_name, vis_dir,
                n_tracks=vis_n_tracks, n_frames=vis_n_frames,
            )
            viz_count += 1

    # Aggregate stats + comparison
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
        print("\n  IMPROVEMENT vanilla -> finetuned:")
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
    n_total = len(train_seqs)
    max_tr = args.train_max_seqs
    if max_tr > 0 and n_total > max_tr:
        train_seqs = random.sample(train_seqs, max_tr)
        print(f"Train split: sampled {max_tr} / {n_total} sequences")
    else:
        print(f"Train split: {n_total} sequences")
    print(f"Val   split: {len(val_seqs)} sequences")

    if args.vis_max_seqs > 0:
        print(
            f"Visualization: first {args.vis_max_seqs} seqs per split "
            f"-> {args.vis_dir}/"
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
    for split_name, seqs in [("train", train_seqs), ("val", val_seqs)]:
        all_results[split_name] = eval_split(
            split_name, seqs, args.dataset_dir, models,
            args.track_chunk, args.vis_thresh, device, dtype,
            vis_max_seqs=args.vis_max_seqs,
            vis_n_tracks=args.vis_n_tracks,
            vis_n_frames=args.vis_n_frames,
            vis_dir=args.vis_dir,
        )

    # Cross-split summary table
    sep = "=" * 70
    print(f"\n{sep}")
    print("SUMMARY  (train vs val splits — check for overfitting)")
    print(sep)
    for tag in ("vanilla", "finetuned"):
        for split_name in ("train", "val"):
            agg = all_results.get(split_name, {}).get(f"{tag}_mean")
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
