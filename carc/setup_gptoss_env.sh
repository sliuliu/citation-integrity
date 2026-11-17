#!/bin/bash
# One-time setup on CARC, run from the project root on a login node. Separate from
# .venv, whose pinned versions the MultiVerS runs depend on:
#   bash carc/setup_gptoss_env.sh
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 .venv-gptoss
uv pip install --python .venv-gptoss/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv-gptoss/bin/python "transformers==4.57.6" accelerate scikit-learn
# Only the transformers weights, not the original/ and metal/ copies. Compute nodes have no network.
.venv-gptoss/bin/python -c "from huggingface_hub import snapshot_download as d; \
print(d('openai/gpt-oss-20b', ignore_patterns=['original/*', 'metal/*']))"
