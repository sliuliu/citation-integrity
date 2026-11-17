#!/bin/bash
# OVERSIMPLIFY test with gpt-oss-20b (scripts/llm_local.py). From the project root:
#   bash carc/setup_gptoss_env.sh     # once, on a login node
#   mkdir -p logs
#   sbatch carc/llm_gptoss.sbatch
# Resubmitting continues where a stopped run left off.
#SBATCH --job-name=ci-gptoss
#SBATCH --account=vsharan_1861
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
cd scripts
../.venv-gptoss/bin/python llm_local.py
