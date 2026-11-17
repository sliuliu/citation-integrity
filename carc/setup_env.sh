#!/bin/bash
# One-time setup on CARC, run from the project root on a login node:
#   bash carc/setup_env.sh
set -euo pipefail
command -v uv >/dev/null || pip install --user uv
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python .venv/bin/python "transformers==4.26.1" "pytorch-lightning==1.9.5" "numpy<1.25" \
    pandas scikit-learn tqdm "setuptools<70" sentencepiece "protobuf<4"
# Cache the Longformer config/tokenizer so compute nodes need no network.
.venv/bin/python -c "from transformers import AutoTokenizer, LongformerConfig; \
AutoTokenizer.from_pretrained('allenai/longformer-large-4096'); LongformerConfig.from_pretrained('allenai/longformer-large-4096')"
