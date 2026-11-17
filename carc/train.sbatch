#!/bin/bash
# Fine-tune MultiVerS from healthver.ckpt, one array task per row of an
# experiments table. From the project root:
#   mkdir -p logs                              # Slurm opens --output before this script runs
#   sbatch carc/train.sbatch                   # oracle rows (carc/experiments.tsv, 6), seed 76
#   sbatch --array=0-3 --export=ALL,EXPERIMENTS=carc/topk.tsv carc/train.sbatch
#                                              # top-k rows (4)
#   sbatch --array=0-3 --export=ALL,EXPERIMENTS=carc/topk.tsv,SEED=1,SAVE=none carc/train.sbatch
#                                              # another seed, keep predictions only
#SBATCH --job-name=ci-train
#SBATCH --account=vsharan_1861
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
EXPERIMENTS=${EXPERIMENTS:-carc/experiments.tsv}
SEED=${SEED:-76}
SAVE=${SAVE:-best}
mkdir -p logs runs
IFS=$'\t' read -r NAME DATA TRAIN SELECT < <(grep -v '^#' "$EXPERIMENTS" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
# Seed-76 runs keep their original directory names.
TAG="$NAME"
[ "$SEED" != 76 ] && TAG="$NAME-s$SEED"
echo "task $SLURM_ARRAY_TASK_ID: $TAG data=$DATA train=$TRAIN select=$SELECT seed=$SEED save=$SAVE on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
cd scripts
../.venv/bin/python train_multivers.py \
    --data-dir ../data/"$DATA" \
    --init ../checkpoints/multivers/healthver-state.ckpt \
    --train-splits "$TRAIN" --eval-splits dev,test --select-on "$SELECT" \
    --seed "$SEED" --save "$SAVE" \
    --out-dir ../runs/"$TAG"-"$SLURM_ARRAY_JOB_ID"
