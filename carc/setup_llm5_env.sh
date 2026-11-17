#!/bin/bash
# One-time setup on CARC for models that need transformers 5 (Qwen3.8-27B, MedGemma 27B),
# run from the project root on a login node after logging in to Hugging Face (MedGemma is gated):
#   bash carc/setup_llm5_env.sh
# Separate from .venv and .venv-gptoss, so earlier runs keep their exact versions. Weights (about
# 100 GiB) go to the project directory rather than the home directory quota.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$PWD/.hf-cache" HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
uv venv --python 3.11 .venv-llm5
uv pip install --python .venv-llm5/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-llm5/bin/python "transformers==5.22.1" accelerate scikit-learn
for model in Qwen/Qwen3.8-27B google/medgemma-27b-text-it; do
    .venv-llm5/bin/python -c "from huggingface_hub import snapshot_download as d; print(d('$model'))"
done
du -sh "$HF_HOME"
