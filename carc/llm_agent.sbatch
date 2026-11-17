#!/bin/bash
# Evidence-seeking gpt-oss-20b agent (scripts/llm_agent.py). From the project root:
#   sbatch --export=ALL,START=none,SPLIT=dev,LIMIT=20 carc/llm_agent.sbatch
# START is none or top20. LIMIT judges only the next N citations (a speed check). Resubmitting continues
# where a run left off.
#SBATCH --job-name=ci-agent
#SBATCH --account=vsharan_1861
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
echo "start=$START split=$SPLIT limit=${LIMIT:-all} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd scripts
../.venv-gptoss/bin/python llm_agent.py --start "$START" --split "$SPLIT" $LIMIT_ARG
