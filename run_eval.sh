#!/bin/bash
# run_eval.sh — run inside the sleep pod (SSH in, then: bash run_eval.sh)
# Evaluates EXP-06 ep95 / EXP-07 ep70 / EXP-08 ep35 sequentially.
# Estimated runtime: ~1.5h total on a single A100.
# Run in a screen/tmux session so SSH disconnect doesn't kill it.
set -e

# ── Setup ──────────────────────────────────────────────────────────────────
git clone https://github.com/ahmeddawy/low_parallax_vggt.git /workspace/vggt
cd /workspace/vggt && git checkout plane-rigidity
pip install -e /workspace/vggt -q
pip install peft -q   # not in research-cuda12 image

mkdir -p /workspace/low_parallax_vggt
wget -q -O /workspace/low_parallax_vggt/model.pt \
  "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"

DATASET=/mnt/bucket/dawy/vggt_finetune/tracking_whisper_clean_dataset
VANILLA=/workspace/low_parallax_vggt/model.pt
BASE=/mnt/bucket/dawy/law_parallax_vggt_exp2

# ── EXP-06  ep95  (track + reproj + plane_rigidity)  ← best val=0.0526 ───
OUT06=$BASE/reproj_plane/rembrand_lora_r16_reproj_plane_v1/eval_ep95
mkdir -p $OUT06
echo "[$(date)] EXP-06 starting..." | tee -a $OUT06/eval.log
python /workspace/vggt/eval_track_head.py \
  --vanilla-ckpt    $VANILLA \
  --finetuned-ckpt  $BASE/reproj_plane/rembrand_lora_r16_reproj_plane_v1/ckpts/checkpoint_95.pt \
  --lora \
  --dataset-dir     $DATASET \
  --train-split-file $DATASET/train_split.txt \
  --val-split-file   $DATASET/val_split.txt \
  --train-max-seqs  -1 \
  --vis-max-seqs    -1 \
  --vis-n-tracks    50 \
  --ransac-thresh   3.0 \
  --output-json     $OUT06/track_eval_results.json \
  --vis-dir         $OUT06/viz \
  --homography-dir  $OUT06/homographies \
  --ae-output-dir   $OUT06/ae_output \
  2>&1 | tee -a $OUT06/eval.log
echo "[$(date)] EXP-06 done." | tee -a $OUT06/eval.log

# ── EXP-07  ep70  (track-only baseline, depth+camera dead) ───────────────
OUT07=$BASE/depth_consistency/rembrand_lora_r16_depth_consistency_v1/eval_ep70
mkdir -p $OUT07
echo "[$(date)] EXP-07 starting..." | tee -a $OUT07/eval.log
python /workspace/vggt/eval_track_head.py \
  --vanilla-ckpt    $VANILLA \
  --finetuned-ckpt  $BASE/depth_consistency/rembrand_lora_r16_depth_consistency_v1/ckpts/checkpoint_70.pt \
  --lora \
  --dataset-dir     $DATASET \
  --train-split-file $DATASET/train_split.txt \
  --val-split-file   $DATASET/val_split.txt \
  --train-max-seqs  -1 \
  --vis-max-seqs    -1 \
  --vis-n-tracks    50 \
  --ransac-thresh   3.0 \
  --output-json     $OUT07/track_eval_results.json \
  --vis-dir         $OUT07/viz \
  --homography-dir  $OUT07/homographies \
  --ae-output-dir   $OUT07/ae_output \
  2>&1 | tee -a $OUT07/eval.log
echo "[$(date)] EXP-07 done." | tee -a $OUT07/eval.log

# ── EXP-08  ep35  (last clean ckpt before NaN) ───────────────────────────
OUT08=$BASE/track_depth_reproj/rembrand_lora_r16_track_depth_reproj_v1/eval_ep35
mkdir -p $OUT08
echo "[$(date)] EXP-08 starting..." | tee -a $OUT08/eval.log
python /workspace/vggt/eval_track_head.py \
  --vanilla-ckpt    $VANILLA \
  --finetuned-ckpt  $BASE/track_depth_reproj/rembrand_lora_r16_track_depth_reproj_v1/ckpts/checkpoint_35.pt \
  --lora \
  --dataset-dir     $DATASET \
  --train-split-file $DATASET/train_split.txt \
  --val-split-file   $DATASET/val_split.txt \
  --train-max-seqs  -1 \
  --vis-max-seqs    -1 \
  --vis-n-tracks    50 \
  --ransac-thresh   3.0 \
  --output-json     $OUT08/track_eval_results.json \
  --vis-dir         $OUT08/viz \
  --homography-dir  $OUT08/homographies \
  --ae-output-dir   $OUT08/ae_output \
  2>&1 | tee -a $OUT08/eval.log
echo "[$(date)] EXP-08 done." | tee -a $OUT08/eval.log

echo "[$(date)] All evals complete."
