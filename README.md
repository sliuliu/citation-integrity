# Citation Integrity: reproduction and extensions (CSCI544 Group 23)

Reproduction of Sarol et al. 2024, *Assessing citation integrity in biomedical publications*
(Bioinformatics, btae420), followed by experiments on improving NOT_ACCURATE detection.
Pipeline: citing sentence (citance) -> BM25 + MonoT5 evidence retrieval from the cited paper ->
three-way classification (ACCURATE / NOT_ACCURATE / IRRELEVANT).

- **[RESULTS.md](RESULTS.md)**: summary of the results.
- **[runs/README.md](runs/README.md)**: full experiment log with every number, job id and check.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | All code: retrieval, data building, MultiVerS / NLI training, LLM experiments, analysis |
| `carc/` | USC CARC setup scripts and Slurm jobs (A40 GPUs) |
| `runs/` | Predictions, metrics and per-epoch histories of every run, plus the run log |

Not in the repository (see `.gitignore`): model weights, virtual environments, derived data and
third-party code.

## Getting the inputs

| Input | Source |
|---|---|
| `upstream/Citation-Integrity` | https://github.com/ScienceNLP-Lab/Citation-Integrity (commit e9823957b): annotations, released checkpoint |
| `upstream/multivers` | https://github.com/dwadden/multivers (commit a6ce033): MultiVerS code, `healthver.ckpt` |
| `external/team-toolkit` | https://github.com/liuruida200-liu/CSCI544_Group23_Weeks1_2_Reproduction_Toolkit (commit 0ee9a36): PubMedBERT reranker |
| `checkpoints/author/` | The authors' released `bestModel*.ckpt` |

Derived data under `data/` is rebuilt by the scripts:

| Data | Script |
|---|---|
| `data/retrieval/` | `bm25_candidates.py` (Pyserini, Java 11), `monot5_rerank.py` |
| `data/topk/` | `build_topk_data.py`, `build_reranker_data.py` |
| `data/oracle/` | `build_oracle_data.py` |
| `data/agent/` | `llm_agent.py --prepare` |

## Environments

| Environment | Used for | Setup |
|---|---|---|
| `.venv` | MultiVerS, NLI, analysis (torch 2.0.1, transformers 4.26.1) | `carc/setup_env.sh` |
| `.venv-bm25` | BM25 retrieval (Pyserini 0.24.0, Java 11) | local |
| `.venv-gptoss` | gpt-oss-20b (transformers 4.57.6) | `carc/setup_gptoss_env.sh` |
| `.venv-llm5` | Qwen3.8-27B, MedGemma 27B (transformers 5.22.1) | `carc/setup_llm5_env.sh` |

Each script's docstring gives its exact command; each `carc/*.sbatch` header gives its `sbatch` line.
