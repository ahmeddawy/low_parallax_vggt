#!/usr/bin/env python3
"""
Experiment health monitor for EXP-03a/b.

Reads TensorBoard event files from GCS (/mnt/bucket) and reports training health.
Checks for NaN losses, explosions, and stagnation. Optionally writes a JSON
status report to GCS so it can be read from anywhere.

Usage (run from inside either pod):
    python /workspace/vggt/scripts/monitor.py            # single report
    python /workspace/vggt/scripts/monitor.py --watch 10 # re-check every 10 min
    python /workspace/vggt/scripts/monitor.py --watch 10 --write-status
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

# ── Experiment registry ───────────────────────────────────────────────────────

EXPERIMENTS = {
    "EXP-03a": {
        "name": "rembrand_lora_r16_plane_rigidity_v1",
        "config": "rembrand",
        "tb_dir": (
            "/mnt/bucket/dawy/law_parallax_vggt_exp1"
            "/rembrand_lora_r16_plane_rigidity_v1/tensorboard"
        ),
        "log_file": (
            "/mnt/bucket/dawy/law_parallax_vggt_exp1"
            "/rembrand_lora_r16_plane_rigidity_v1/log.txt"
        ),
    },
    "EXP-03b": {
        "name": "rembrand_lora_r16_plane_rigidity_v2",
        "config": "rembrand_exp03b",
        "tb_dir": (
            "/mnt/bucket/dawy/law_parallax_vggt_exp1"
            "/rembrand_lora_r16_plane_rigidity_v2/tensorboard"
        ),
        "log_file": (
            "/mnt/bucket/dawy/law_parallax_vggt_exp1"
            "/rembrand_lora_r16_plane_rigidity_v2/log.txt"
        ),
    },
}

# Per-experiment status files (one per pod, keyed by exp_id slug)
STATUS_DIR = "/mnt/bucket/dawy/monitoring"
STATUS_FILES = {
    "EXP-03a": f"{STATUS_DIR}/exp03a_status.json",
    "EXP-03b": f"{STATUS_DIR}/exp03b_status.json",
}

# ── What to watch ─────────────────────────────────────────────────────────────

WATCHED_KEYS = [
    "objective",
    "loss_track",
    "loss_plane_rigidity",
    "loss_conf_depth",
    "loss_reg_depth",
]

# Gradient health tags written by trainer (Grad/{module})
GRAD_KEYS = ["aggregator", "depth", "camera", "track"]

STEPS_PER_EPOCH = 28        # limit_train_batches from config
STALE_WINDOW = 10           # epochs with <1% relative improvement → stagnant
EXPLOSION_THRESHOLD = 500.0 # loss above this = explosion


# ── TensorBoard reader ────────────────────────────────────────────────────────

def read_tb_scalars(tb_dir: str) -> dict:
    """Returns {tag: [(step, value), ...]} for all scalar summaries."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    result = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        result[tag] = [(e.step, e.value) for e in events]
    return result


# ── Health assessment ─────────────────────────────────────────────────────────

def assess(exp_id: str, cfg: dict) -> dict:
    report = {
        "exp_id": exp_id,
        "name": cfg["name"],
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "status": "unknown",
        "last_epoch": None,
        "last_step": None,
        "issues": [],
        "metrics": {},
        "grads": {},
    }

    tb_dir = cfg["tb_dir"]
    if not os.path.isdir(tb_dir):
        report["status"] = "not_started"
        report["issues"].append(f"TB dir not found: {tb_dir}")
        return report

    event_files = list(Path(tb_dir).glob("events.out.tfevents.*"))
    if not event_files:
        report["status"] = "not_started"
        report["issues"].append("No TensorBoard event files yet")
        return report

    try:
        scalars = read_tb_scalars(tb_dir)
    except Exception as exc:
        report["status"] = "error"
        report["issues"].append(f"TB read failed: {exc}")
        return report

    if not scalars:
        report["status"] = "no_data"
        report["issues"].append("Event files exist but contain no scalar data yet")
        return report

    issues = []

    # ── Per-key checks ────────────────────────────────────────────────────────
    for key in WATCHED_KEYS:
        for phase in ("train", "val"):
            tag = f"Values/{phase}/{key}"
            if tag not in scalars or not scalars[tag]:
                continue
            pts = scalars[tag]
            last_step, last_val = pts[-1]

            report["metrics"][f"{phase}/{key}"] = {
                "last_step": last_step,
                "last_value": round(last_val, 6),
                "n_points": len(pts),
            }

            if math.isnan(last_val) or math.isinf(last_val):
                issues.append(f"NaN/Inf  {phase}/{key}  step={last_step}")
                continue

            if abs(last_val) > EXPLOSION_THRESHOLD:
                issues.append(
                    f"Explosion  {phase}/{key} = {last_val:.2f}  step={last_step}"
                )

            # Stagnation: compare mean of last two STALE_WINDOW epochs
            if len(pts) >= STALE_WINDOW * 2 * STEPS_PER_EPOCH:
                window_size = STALE_WINDOW * STEPS_PER_EPOCH
                early_vals = [v for _, v in pts[-(2 * window_size):-window_size]]
                late_vals  = [v for _, v in pts[-window_size:]]
                early_mean = sum(early_vals) / len(early_vals)
                late_mean  = sum(late_vals)  / len(late_vals)
                if early_mean > 1e-8:
                    rel_improvement = (early_mean - late_mean) / early_mean
                    if rel_improvement < 0.01:
                        issues.append(
                            f"Stagnation  {phase}/{key}: "
                            f"early={early_mean:.5f} → late={late_mean:.5f} "
                            f"({rel_improvement*100:.1f}% change)"
                        )

    # ── plane_rigidity zero check (only after 5 epochs) ──────────────────────
    pr_tag = "Values/train/loss_plane_rigidity"
    if pr_tag in scalars and scalars[pr_tag]:
        pts = scalars[pr_tag]
        if len(pts) >= 5 * STEPS_PER_EPOCH:
            recent_vals = [v for _, v in pts[-5 * STEPS_PER_EPOCH:]]
            if max(recent_vals) < 1e-8:
                issues.append(
                    "loss_plane_rigidity stuck at zero — constraint not activating"
                )

    # ── Gradient health ───────────────────────────────────────────────────────
    for mod in GRAD_KEYS:
        tag = f"Grad/{mod}"
        if tag in scalars and scalars[tag]:
            pts = scalars[tag]
            last_step, last_val = pts[-1]
            report["grads"][mod] = round(last_val, 4)
            if last_val > 10.0:
                issues.append(
                    f"High grad  Grad/{mod} = {last_val:.2f}  step={last_step}"
                )

    # ── Epoch / step progress ─────────────────────────────────────────────────
    obj_train = scalars.get("Values/train/objective", [])
    if obj_train:
        last_step = obj_train[-1][0]
        report["last_step"] = last_step
        report["last_epoch"] = last_step // STEPS_PER_EPOCH

    report["issues"] = issues
    report["status"] = "healthy" if not issues else "needs_attention"
    return report


# ── Pretty printer ────────────────────────────────────────────────────────────

SYMBOLS = {
    "healthy": "✓",
    "needs_attention": "⚠",
    "not_started": "○",
    "no_data": "⋯",
    "error": "✗",
    "unknown": "?",
}


def print_report(r: dict) -> None:
    sym = SYMBOLS.get(r["status"], "?")
    print(f"\n{'─' * 62}")
    print(f"  {sym}  {r['exp_id']}  ({r['name']})")
    print(f"     Status : {r['status'].upper()}")
    if r["last_epoch"] is not None:
        print(f"     Epoch  : ~{r['last_epoch']} / 100  (step {r['last_step']})")
    print(f"     At     : {r['timestamp']} UTC")

    if r["metrics"]:
        print()
        print("  Losses (last logged value):")
        for k, v in sorted(r["metrics"].items()):
            print(f"    {k:<38s}  {v['last_value']:.6f}  (step {v['last_step']})")

    if r["grads"]:
        print()
        print("  Gradient norms (pre-clip, last step):")
        for mod, g in r["grads"].items():
            flag = "  ← HIGH" if g > 10.0 else ""
            print(f"    {mod:<12s}  {g:.4f}{flag}")

    print()
    if r["issues"]:
        print("  Issues:")
        for issue in r["issues"]:
            print(f"    ⚠  {issue}")
    else:
        print("  No issues detected.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor EXP-03a/b training health")
    parser.add_argument(
        "--watch", type=int, default=0, metavar="MINUTES",
        help="Re-check every N minutes (0 = run once and exit)",
    )
    parser.add_argument(
        "--write-status", action="store_true",
        help=f"Write JSON status to {STATUS_DIR}/<exp>_status.json after each check",
    )
    parser.add_argument(
        "--exp", choices=list(EXPERIMENTS.keys()), default=None,
        help="Monitor only one experiment (default: both)",
    )
    args = parser.parse_args()

    exps = (
        {args.exp: EXPERIMENTS[args.exp]}
        if args.exp
        else EXPERIMENTS
    )

    while True:
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n{'═' * 62}")
        print(f"  VGGT Experiment Monitor  —  {now}")
        print(f"{'═' * 62}")

        all_reports = []
        for exp_id, cfg in exps.items():
            report = assess(exp_id, cfg)
            print_report(report)
            all_reports.append(report)

        if args.write_status:
            for report in all_reports:
                out_path = Path(STATUS_FILES[report["exp_id"]])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as fh:
                    json.dump(report, fh, indent=2)
                print(f"\n  Status written → {out_path}")

        if args.watch <= 0:
            break

        print(f"\n  Next check in {args.watch} min  (Ctrl-C to stop)...")
        try:
            time.sleep(args.watch * 60)
        except KeyboardInterrupt:
            print("\n  Stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
