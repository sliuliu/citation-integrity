#!/bin/bash
# PubMedBERT reranker from the team toolkit (external/team-toolkit), with the
# recipe of the midterm report (1 epoch, 4 negatives, 4 groups per batch, seed 0):
#   - one model on all train articles -> rankings for dev and test
#   - 5-fold CV over train articles -> out-of-fold rankings for train
# then builds data/topk/pubmedbert20 for MultiVerS training. From the project root:
#   mkdir -p logs
#   sbatch carc/rerank_pubmedbert.sbatch
#SBATCH --job-name=ci-rerank
#SBATCH --account=vsharan_1861
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv
PY=$PWD/.venv/bin/python
OUT=results/retrieval/pubmedbert-$SLURM_JOB_ID
BASE="--epochs 1 --negatives 4 --groups-per-batch 4 --seed 0 --log-every 200"

cd external/team-toolkit
mkdir -p "$OUT"
$PY scripts/train_reranker.py $BASE --out "$OUT/model" --predict test --rankings-out "$OUT/test.jsonl"
$PY scripts/train_reranker.py --no-train --init-from "$OUT/model" --predict dev --rankings-out "$OUT/dev.jsonl"
for k in 0 1 2 3 4; do
    $PY scripts/train_reranker.py $BASE --folds 5 --fold "$k" --train-split train \
        --out "$OUT/cv_f$k" --predict train --rankings-out "$OUT/train_f$k.jsonl"
    rm -rf "$OUT/cv_f$k"
done
cat "$OUT"/train_f*.jsonl > "$OUT/train_cv.jsonl"
for split in test dev; do
    $PY scripts/retrieval_harness.py --split $split --ranker file --rankings-in "$OUT/$split.jsonl" \
        --metrics-out "$OUT/$split.metrics.json" | tail -3
done
$PY scripts/retrieval_harness.py --split train --ranker file --rankings-in "$OUT/train_cv.jsonl" \
    --metrics-out "$OUT/train_cv.metrics.json" | tail -3

cd ../../scripts
$PY build_reranker_data.py --train "../external/team-toolkit/$OUT/train_cv.jsonl" \
    --dev "../external/team-toolkit/$OUT/dev.jsonl" --test "../external/team-toolkit/$OUT/test.jsonl" \
    --k 20 --out-dir ../data/topk/pubmedbert20
