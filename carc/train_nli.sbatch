#!/bin/bash
# Fine-tune the DeBERTa NLI model as the classifier (scripts/train_nli.py), one array
# task per row of carc/nli.tsv. From the project root:
#   mkdir -p logs
#   sbatch carc/train_nli.sbatch                              # seed 76
#   sbatch --export=ALL,SEED=1 carc/train_nli.sbatch          # another seed
#SBATCH --job-name=ci-nli
#SBATCH --account=vsharan_1861
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
EXPERIMENTS=${EXPERIMENTS:-carc/nli.tsv}
SEED=${SEED:-76}
SAVE=${SAVE:-none}
mkdir -p logs runs
IFS=$'\t' read -r NAME DATA TRAIN SELECT < <(grep -v '^#' "$EXPERIMENTS" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
# Seed-76 runs keep their original directory names.
TAG="$NAME"
[ "$SEED" != 76 ] && TAG="$NAME-s$SEED"
echo "task $SLURM_ARRAY_TASK_ID: $TAG data=$DATA train=$TRAIN select=$SELECT seed=$SEED save=$SAVE on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
cd scripts
../.venv/bin/python train_nli.py \
    --data-dir ../data/"$DATA" \
    --train-splits "$TRAIN" --eval-splits dev,test --select-on "$SELECT" \
    --seed "$SEED" --save "$SAVE" \
    --out-dir ../runs/"$TAG"-"$SLURM_ARRAY_JOB_ID"
