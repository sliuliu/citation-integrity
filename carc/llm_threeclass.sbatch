#!/bin/bash
# Three-way LLM test (scripts/llm_threeclass.py). From the project root:
#   sbatch --export=ALL,PROMPT=improved,EVIDENCE=topk/top20,SPLIT=dev carc/llm_threeclass.sbatch
#   sbatch --export=ALL,MODEL=Qwen/Qwen3.8-27B,PROMPT=improved,EVIDENCE=topk/top20,SPLIT=dev,LIMIT=20 carc/llm_threeclass.sbatch
# MODEL defaults to openai/gpt-oss-20b; other models run in .venv-llm5 (carc/setup_llm5_env.sh).
# LIMIT judges only the next N citations (a speed check). Resubmitting continues where a run left off.
#SBATCH --job-name=ci-llm3
#SBATCH --account=vsharan_1861
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --mail-type=END,FAIL
set -euo pipefail
MODEL=${MODEL:-openai/gpt-oss-20b}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
if [ "$MODEL" = openai/gpt-oss-20b ]; then
    PY=../.venv-gptoss/bin/python
else
    PY=../.venv-llm5/bin/python
    export HF_HOME="$PWD/.hf-cache"
fi
echo "model=$MODEL prompt=$PROMPT evidence=$EVIDENCE split=$SPLIT limit=${LIMIT:-all} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
cd scripts
$PY llm_threeclass.py --model "$MODEL" --prompt "$PROMPT" --evidence "$EVIDENCE" --split "$SPLIT" $LIMIT_ARG
