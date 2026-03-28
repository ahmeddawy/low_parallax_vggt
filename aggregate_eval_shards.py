#!/usr/bin/env python3
"""
aggregate_eval_shards.py — merge per-shard eval JSONs from indexed job.

Reads shard_0.json .. shard_N.json from a bucket eval_shard/ dir,
merges per-seq results, recomputes aggregate means, prints summary.

Usage:
    python aggregate_eval_shards.py \
        --shard-dir /path/to/eval_shard \
        --output    /path/to/eval_results_full.json
"""

import argparse
import glob
import json
import os

import numpy as np


METRIC_KEYS = [
    "ate", "median_te",
    "delta_1px", "delta_2px", "delta_5px", "delta_10px",
    "vis_acc",
]
CORNER_KEYS = [
    "pt_mean_px", "pt_median_px", "pt_max_px", "pt_p95_px",
    "corner_mean_px", "corner_median_px", "corner_max_px", "corner_p95_px",
    "jitter_score",
]


def mean_over_seqs(per_seq, keys):
    vals = {k: [] for k in keys}
    for m in per_seq.values():
        if m is None:
            continue
        for k in keys:
            if k in m:
                vals[k].append(m[k])
    return {k: float(np.mean(v)) if v else float("nan") for k, v in vals.items()}


def merge_split(shards, split_name):
    """Merge shard results for one split. Returns unified per_seq + means."""
    per_seq = {}
    for shard in shards:
        split_data = shard.get(split_name, {})
        per_seq_shard = split_data.get("per_seq", {})
        for seq, result in per_seq_shard.items():
            if seq in per_seq:
                print(f"  [WARN] duplicate sequence {seq} in {split_name} — keeping first")
                continue
            per_seq[seq] = result

    if not per_seq:
        return None

    # Collect all model tags from per_seq values
    tags = set()
    for sr in per_seq.values():
        if sr is not None:
            tags.update(sr.keys())
    tags = sorted(tags)

    means = {}
    for tag in tags:
        tag_seqs = {seq: sr[tag] for seq, sr in per_seq.items()
                    if sr is not None and tag in sr}
        if tag_seqs:
            means[tag] = mean_over_seqs(tag_seqs, METRIC_KEYS + CORNER_KEYS)

    return {"per_seq": per_seq, "mean": means}


def print_summary(split_name, split_result):
    if split_result is None:
        return
    means = split_result.get("mean", {})
    n_seqs = len(split_result.get("per_seq", {}))
    print(f"\n--- {split_name.upper()} ({n_seqs} sequences) ---")
    base_tags = [t for t in means if not t.endswith("_h") and not t.endswith("_mean")]
    for tag in sorted(base_tags):
        m = means[tag]
        print(
            f"  [{tag}]"
            f"  ATE={m.get('ate', float('nan')):.3f}px"
            f"  med={m.get('median_te', float('nan')):.3f}px"
            f"  d1={m.get('delta_1px', float('nan')):.3f}"
            f"  d5={m.get('delta_5px', float('nan')):.3f}"
            f"  vis={m.get('vis_acc', float('nan')):.3f}"
        )
        if "pt_mean_px" in m:
            print(
                f"  {'':>8}  "
                f"pt_mean={m.get('pt_mean_px', float('nan')):.2f}px"
                f"  corner_mean={m.get('corner_mean_px', float('nan')):.2f}px"
                f"  jitter={m.get('jitter_score', float('nan')):.3f}"
            )

    # vanilla vs finetuned comparison
    if "vanilla" in means and "finetuned" in means:
        v, f = means["vanilla"], means["finetuned"]
        d_ate = f.get("ate", 0) - v.get("ate", 0)
        pct = d_ate / (v.get("ate", 1e-9)) * 100
        d5 = f.get("delta_5px", 0) - v.get("delta_5px", 0)
        print(f"\n  vanilla -> finetuned:")
        print(f"    ATE:  {v.get('ate'):.2f} -> {f.get('ate'):.2f}px  ({d_ate:+.2f}px, {pct:+.1f}%)")
        print(f"    d5:   {v.get('delta_5px'):.3f} -> {f.get('delta_5px'):.3f}  ({d5*100:+.1f}pp)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", required=True,
                    help="Directory containing shard_0.json..shard_N.json")
    ap.add_argument("--output", default=None,
                    help="Write merged JSON here (default: shard_dir/eval_results_full.json)")
    args = ap.parse_args()

    shard_files = sorted(glob.glob(os.path.join(args.shard_dir, "shard_*.json")))
    if not shard_files:
        print(f"No shard JSON files found in {args.shard_dir}")
        return

    print(f"Found {len(shard_files)} shard files:")
    shards = []
    for f in shard_files:
        print(f"  {f}")
        shards.append(json.load(open(f)))

    all_results = {}
    for split_name in ("val", "train"):
        if any(split_name in s for s in shards):
            merged = merge_split(shards, split_name)
            if merged:
                all_results[split_name] = merged
                print_summary(split_name, merged)

    out_path = args.output or os.path.join(args.shard_dir, "eval_results_full.json")
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\nMerged results -> {out_path}")


if __name__ == "__main__":
    main()
