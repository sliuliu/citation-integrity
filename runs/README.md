# Runs (fresh restart, 2026-10-26)

| Run | Checkpoint | Inputs | Correct | Micro-F1 | Macro-F1 | ACC / N_ACC / IRR F1 |
|---|---|---|---:|---:|---:|---|
| 01-released-ckpt-author-top5-test | author `bestModel-inference.ckpt` | author `scifact_10` test (top-5) | 338/606 | 0.5578 | 0.4438 | 0.6933 / 0.3123 / 0.3259 |

Paper Table 4 for comparison: top-5 model 0.58/0.50, top-20 model 0.59/0.52.
Run 01 reproduces an earlier exploratory run of ours (not included)
prediction-for-prediction (606/606 identical), here in FP32 instead of FP16.

## Gold-evidence oracle (paper Table 4), CARC A40, jobs 13478320 / 13479153

All runs start from MultiVerS `healthver.ckpt`, use the authors' recipe
(scripts/train_multivers.py) for 5 epochs, one seed (76). Test scores, 606 citations.
"author" = train on train+dev, final epoch reported (the released model is the
final epoch); "clean" = train on train, epoch chosen by dev macro-F1.
Label convention A = paper Section 2.5.3 (ETIQUETTE -> NOT_ACCURATE);
B = authors' reader on multivers-format (no annotated evidence -> IRRELEVANT).

| Run | Epoch | Micro | Macro | ACC | N_ACC | IRR |
|---|---:|---:|---:|---:|---:|---:|
| Paper: Oracle (gold evidence) | | 0.73 | 0.75 | 0.79 | 0.52 | 0.93 |
| oracle-citance-B-author | 5 | 0.794 | 0.725 | 0.843 | 0.349 | 0.983 |
| oracle-citance-B-clean | 4 (dev) | 0.762 | 0.714 | 0.812 | 0.358 | 0.971 |
| oracle-citance-A-author | 5 | 0.708 | 0.592 | 0.851 | 0.354 | 0.571 |
| oracle-citance-A-clean | 5 (dev) | 0.703 | 0.583 | 0.850 | 0.343 | 0.556 |
| Paper: Oracle (gold citation + gold evidence) | | 0.75 | 0.78 | 0.80 | 0.57 | 0.96 |
| oracle-goldctx-B-author | 5 | 0.785 | 0.732 | 0.833 | 0.386 | 0.975 |
| oracle-goldctx-A-clean | 5 (dev) | 0.719 | 0.607 | 0.857 | 0.455 | 0.508 |

On the 487 evidence-bearing test citations, A and B runs perform alike
(accuracy 0.70-0.76, N_ACC F1 0.27-0.39); the A/B gap comes from how the 119
citations without annotated evidence are labelled. Only convention B reaches the
paper's IRR F1. Remaining gap to the paper: N_ACC F1 (0.35-0.39 vs 0.52/0.57).
Per-epoch scores: `runs/oracle-*/history.json`; predictions per epoch in `epoch*/`.

### Three seeds (76, 1, 2), jobs 13478320 / 13479153 / 13480557 (seed 1) / 13480558 (seed 2)

`scripts/aggregate_seeds.py`. Mean (min-max) over the three seeds, test, same
epoch rule as above.

| Run | Seeds (epoch) | Micro | Macro | ACC | N_ACC | IRR |
|---|---|---:|---:|---:|---:|---:|
| Paper: Oracle (gold evidence) | | 0.73 | 0.75 | 0.79 | 0.52 | 0.93 |
| citance-B-author | 76 (ep5), 1 (ep5), 2 (ep5) | 0.787 (0.777-0.794) | 0.721 (0.718-0.725) | 0.836 (0.827-0.843) | 0.346 (0.339-0.351) | 0.979 (0.975-0.983) |
| citance-B-clean | 76 (ep4), 1 (ep3), 2 (ep2) | 0.753 (0.724-0.772) | 0.711 (0.707-0.714) | 0.801 (0.766-0.824) | 0.355 (0.322-0.385) | 0.977 (0.971-0.983) |
| citance-A-author | 76 (ep5), 1 (ep5), 2 (ep5) | 0.700 (0.695-0.708) | 0.584 (0.579-0.592) | 0.845 (0.841-0.851) | 0.367 (0.354-0.377) | 0.540 (0.520-0.571) |
| citance-A-clean | 76 (ep5), 1 (ep4), 2 (ep4) | 0.702 (0.696-0.706) | 0.578 (0.571-0.583) | 0.851 (0.844-0.858) | 0.345 (0.307-0.384) | 0.540 (0.516-0.556) |
| Paper: Oracle (gold citation + gold evidence) | | 0.75 | 0.78 | 0.80 | 0.57 | 0.96 |
| goldctx-B-author | 76 (ep5), 1 (ep5), 2 (ep5) | 0.776 (0.771-0.785) | 0.728 (0.723-0.732) | 0.823 (0.818-0.833) | 0.388 (0.376-0.400) | 0.973 (0.967-0.975) |
| goldctx-A-clean | 76 (ep5), 1 (ep4), 2 (ep3) | 0.730 (0.719-0.736) | 0.612 (0.607-0.620) | 0.861 (0.857-0.865) | 0.463 (0.455-0.474) | 0.513 (0.495-0.536) |

Seed variance is small (N_ACC range <= 0.08, B-author runs <= 0.025), so the
N_ACC gap to the paper (0.35-0.39 vs 0.52/0.57 under convention B) is not seed
noise. Under B no run reaches 0.43 N_ACC F1 at any epoch of any seed. The
error is NOT_ACCURATE <-> ACCURATE confusion: B-author test confusion rows
(gold N_ACC / IRR / ACC; cols pred) are e.g. [[36,3,61],[0,119,1],[69,1,316]]
(citance, seed 1); N_ACC precision and recall are both ~0.32-0.45. IRR is
solved by the empty-evidence shortcut. Paper's micro/IRR are matched or exceeded.

## Retrieval reproduction (BM25 + MonoT5), 2026-10-27

`scripts/bm25_candidates.py` (Pyserini 0.24.0 / Anserini SimpleSearcher, Java 11,
`.venv-bm25`) -> `data/retrieval/bm25-citance.jsonl`; `scripts/monot5_rerank.py`
(pygaggle 08339dd MonoT5 reimplemented, castorini/monot5-base-med-msmarco, FP32,
batch 8) -> `data/retrieval/monot5-citance.jsonl`. Query = authors' scifact_10
claim text; index = all sentences of the cited article (annotations.zip
references_sentence/). `scripts/compare_author_topk.py` against the authors'
scifact_10 documents (their own top-5): same set 3060/3060, same order 3058/3060,
top-1 3060/3060. `scripts/build_topk_data.py` -> `data/topk/top{5,20}` (claims and
labels copied from scifact_10; top5 corpus equals the authors' for 3058/3060 docs).

| Run | Checkpoint | Inputs | Correct | Micro-F1 | Macro-F1 | ACC / N_ACC / IRR F1 |
|---|---|---|---:|---:|---:|---|
| 02-released-ckpt-our-top20-test | author `bestModel-inference.ckpt` | our top-20, citance | 355/606 | 0.5858 | 0.5164 | 0.6947 / 0.4315 / 0.4231 |
| Paper Table 4, top-20 | | | | 0.59 | 0.52 | 0.69 / 0.43 / 0.42 |

Run 02 matches all five published top-20 numbers at two decimals. The released
checkpoint was trained on top-5 inputs (scifact_10), so the paper's top-20 row is
consistent with that top-5-trained model evaluated on top-20 evidence.
Confusion (rows gold N_ACC/IRR/ACC): [[85,12,73],[21,22,7],[118,20,248]].

### Random-evidence controls (released checkpoint, test)

`build_topk_data.py --random-seed 0`: k sentences drawn uniformly from the cited
article (redrawn if the input would exceed 4096 tokens; 1 test citation for k=20).

| Run | Inputs | Correct | Micro-F1 | Macro-F1 | ACC / N_ACC / IRR F1 |
|---|---|---:|---:|---:|---|
| 01 (above) | authors' top-5 (= ours) | 338/606 | 0.558 | 0.444 | 0.693 / 0.312 / 0.326 |
| 03-released-ckpt-random5-test | random 5 | 199/606 | 0.328 | 0.306 | 0.451 / 0.261 / 0.206 |
| 02 (above) | our top-20 | 355/606 | 0.586 | 0.516 | 0.695 / 0.431 / 0.423 |
| 04-released-ckpt-random20-test | random 20 | 266/606 | 0.439 | 0.410 | 0.522 / 0.376 / 0.332 |

Retrieved vs random: -23.0 micro / -13.8 macro (k=5), -14.7 / -10.6 (k=20).

## Retrained on retrieved evidence (top-5 / top-20), CARC jobs 13511512 (seed 76) / 13511513 (1) / 13511514 (2)

From healthver.ckpt, authors' recipe, labels A (scifact_10), data `data/topk/top{5,20}`;
each model evaluated on the same k it was trained on. `aggregate_seeds.py --glob "topk*"`.
Mean (min-max) over three seeds, test.

| Run | Micro | Macro | ACC | N_ACC | IRR |
|---|---:|---:|---:|---:|---:|
| Paper top-20 | 0.59 | 0.52 | 0.69 | 0.43 | 0.42 |
| Released ckpt (top-5 trained) on top-20 (run 02) | 0.586 | 0.516 | 0.695 | 0.431 | 0.423 |
| topk20-author (train+dev, ep5) | 0.563 (0.556-0.576) | 0.505 (0.501-0.510) | 0.665 | 0.432 | 0.418 |
| topk20-clean (train, dev-selected) | 0.562 (0.530-0.579) | 0.498 (0.476-0.514) | 0.667 | 0.428 | 0.399 |
| Paper top-5 | 0.58 | 0.50 | - | - | - |
| Released ckpt on top-5 (run 01) | 0.558 | 0.444 | 0.693 | 0.312 | 0.326 |
| topk5-author (train+dev, ep5) | 0.547 (0.518-0.571) | 0.463 (0.455-0.476) | 0.665 | 0.384 | 0.339 |
| topk5-clean (train, dev-selected) | 0.556 (0.531-0.583) | 0.460 (0.440-0.473) | 0.685 | 0.367 | 0.329 |

The top-5 retrain lands near the released checkpoint (micro 0.547 vs 0.558, macro
0.463 vs 0.444), so the training recipe is close. Training on top-20 does not beat
the top-5-trained released checkpoint evaluated on top-20 (macro 0.505, all seeds
<= 0.510, vs 0.516). Peak GPU memory 19 GB (top-5) / 23 GB (top-20), no gradient
checkpointing.

Evidence ladder, same recipe and protocol (train+dev, final epoch), labels A:
top-5 0.463 -> top-20 0.505 -> gold (oracle-citance-A-author) 0.584 macro;
N_ACC 0.384 -> 0.432 -> 0.367; ACC 0.665 -> 0.665 -> 0.845; IRR 0.339 -> 0.418 -> 0.540.

## PubMedBERT reranker (team toolkit) as an evidence source, 2026-10-28

Reranker retrained on CARC with the midterm recipe (`carc/rerank_pubmedbert.sbatch`, job
13517059; external/team-toolkit at 0ee9a36): R@20 dev 0.640 / test 0.635 / train 5-fold
CV 0.563 (report: 0.643 / 0.638 / 0.560). `data/topk/pubmedbert20` from
`build_reranker_data.py` (train = out-of-fold CV rankings; 1 train claim lost its 2
lowest-ranked sentences to fit 4096 tokens). MultiVerS jobs 13517493 / 13517494 (s1) /
13517495 (s2); the first attempt (13517060-2) failed on that claim.

Gold-sentence coverage of the test documents (1,214 gold sentences, 487 citations):
top-5 0.275, MonoT5 top-20 0.513, PubMedBERT top-20 0.634 (all gold present 0.166 / 0.363 / 0.497).

| Evidence | Released ckpt (frozen) | Retrained, train+dev ep5 (3 seeds) | Retrained, clean (3 seeds) |
|---|---|---|---|
| MonoT5 top-20 | 0.586 / 0.516 (run 02) | 0.563 / 0.505, N_ACC 0.432 | 0.562 / 0.498, N_ACC 0.428 |
| PubMedBERT top-20 | 0.548 / 0.481 (run 05) | 0.549 / 0.477 (0.458-0.497), N_ACC 0.367 | 0.568 / 0.486 (0.465-0.497), N_ACC 0.354 |

micro / macro F1, test. Paired by seed, accuracy differences MonoT5 vs PubMedBERT:
author 349 vs 324 (McNemar p=0.041), 337 vs 334, 338 vs 340; clean 321 vs 342 (p=0.11),
351 vs 346, 349 vs 345; other p > 0.7. Better retrieval (+12 points gold coverage) does
not improve classification, frozen or retrained; macro and N_ACC are slightly lower.

## Would a heavier NOT_ACCURATE loss weight help? (`scripts/reweight_sweep.py`)

The recipe already weights NOT_ACCURATE x2.13 and IRRELEVANT x9.3. Scaling p(NOT_ACCURATE)
by w (~ an extra loss weight), w tuned on dev for the clean runs, applied to test:
retrieved evidence (top-20 MonoT5, PubMedBERT) test macro changes -0.021..+0.002; even the
test-optimal w gains <= +0.014. ROC AUC of p(N_ACC) for N_ACC vs ACC is only 0.56-0.63
(gold evidence 0.73-0.76), so reweighting moves the threshold, not the separability.

NOT_ACCURATE recall by annotated type (test, final epoch, mean over 3 seeds; counts
ETIQUETTE 73, OVERSIMPLIFY 47, NOT_SUBSTANTIATE 29, CONTRADICT 12, MISQUOTE 9):

| Model | ETIQUETTE | OVERSIMPLIFY | NOT_SUBST | CONTRADICT | MISQUOTE |
|---|---:|---:|---:|---:|---:|
| topk5-author | 0.48 | 0.35 | 0.51 | 0.25 | 0.33 |
| topk20-author | 0.60 | 0.46 | 0.47 | 0.44 | 0.56 |
| pubmedbert20-author | 0.47 | 0.39 | 0.32 | 0.31 | 0.41 |
| oracle-citance-A-author (gold) | 0.38 | 0.17 | 0.24 | 0.44 | 0.41 |
| released ckpt, top-20 | 0.58 | 0.40 | 0.52 | 0.25 | 0.67 |

## Zero-shot NLI diagnostic (`scripts/nli_diagnostic.py`, cache `runs/nli-diagnostic/`)

MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli, premise = evidence sentence,
hypothesis = citance without citation markers; per-citation aggregate chosen on dev.
Test ROC AUC, gold NOT_ACCURATE vs ACCURATE, same citations for both models:

| Evidence | NLI (zero-shot) | MultiVerS |
|---|---:|---:|
| MonoT5 top-20 (556) | 0.616 | released 0.616; retrained author 0.608-0.616 |
| PubMedBERT top-20 (556) | 0.605 | released 0.607; retrained author 0.584-0.631 |
| gold, citations with evidence (486) | 0.649 | oracle-citance-A-author 0.654-0.685 |

(The 0.73-0.76 gold AUC in the reweighting section includes empty-evidence citations;
under labels A an empty document is always NOT_ACCURATE when the gold is N_ACC or ACC,
so that number contains a shortcut. 0.654-0.685 above excludes them.)

MonoT5 top-20: NLI and released MultiVerS scores correlate weakly (Spearman 0.12-0.16);
equal-weight rank average gives AUC 0.647-0.652 with each of 4 MultiVerS models.
At equal budget (203 flagged each), only 85 overlap. NLI is better on NOT_SUBSTANTIATE
(0.62 vs 0.52) and CONTRADICT (0.42 vs 0.25) and when retrieval missed the evidence
(0.43 vs 0.34); MultiVerS is better on ETIQUETTE (0.58 vs 0.47), OVERSIMPLIFY (0.40 vs
0.32), MISQUOTE. With gold evidence, OVERSIMPLIFY vs ACCURATE AUC is 0.59 (NLI) / 0.58
(MultiVerS): close to chance for both.

## MultiVerS + NLI + citance features (`scripts/nli_combine.py`), MonoT5 top-20, test

Stage 1: logistic regression on train using NLI aggregates (9) and citance features (4:
multi_cit marker, other_cit / cit counts, log length). Stage 2: logistic regression on dev
using MultiVerS log-probs of a clean run (+ stage 1). Control: MultiVerS log-probs alone
refit on dev. Means over the three clean seeds:

| Model | Micro | Macro | ACC | N_ACC | IRR | AUC N_ACC vs ACC |
|---|---:|---:|---:|---:|---:|---:|
| MultiVerS as trained | 0.562 | 0.498 | 0.667 | 0.429 | 0.400 | 0.564-0.625 |
| control: MultiVerS refit on dev | 0.540 | 0.475 | 0.658 | 0.384 | 0.384 | 0.482-0.576 |
| MultiVerS + stage 1 | 0.556 | 0.500 | 0.650 | 0.470 | 0.382 | 0.637-0.650 |
| stage 1 alone | 0.462 | 0.400 | 0.584 | 0.435 | 0.180 | 0.648 |

Stage-1 ablation, test AUC: NLI features only 0.610, citance features only 0.620, both 0.648.
Per seed N_ACC F1 0.414->0.462, 0.438->0.481, 0.434->0.468; macro 0.476->0.487,
0.504->0.511, 0.514->0.503.

## Fine-tuned NLI classifier (`scripts/train_nli.py`), CARC jobs 13588234 (seed 76) / 13588235 (1) / 13588236 (2)

MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli with its NLI head reused
(entailment/contradiction/neutral -> ACC/N_ACC/IRR), MonoT5 top-20 truncated to 512 tokens
from the lowest-ranked end, labels A, authors' class weights, 5 epochs, lr 1e-5, batch 16.
Peak GPU memory 13.8 GB. Test, mean (min-max) over three seeds:

| Run | Micro | Macro | ACC | N_ACC | IRR | AUC N_ACC vs ACC |
|---|---:|---:|---:|---:|---:|---:|
| Paper top-20 | 0.59 | 0.52 | 0.69 | 0.43 | 0.42 | |
| Released ckpt, top-20 | 0.586 | 0.516 | 0.695 | 0.431 | 0.423 | 0.616 |
| MultiVerS retrained, train+dev ep5 | 0.563 | 0.505 | 0.665 | 0.432 | 0.418 | 0.608-0.616 |
| NLI fine-tuned, train+dev ep5 | 0.636 (0.627-0.642) | 0.524 (0.519-0.533) | 0.754 | 0.446 | 0.371 | 0.666-0.683 |
| MultiVerS retrained, clean | 0.562 | 0.498 | 0.667 | 0.428 | 0.399 | 0.564-0.625 |
| NLI fine-tuned, clean | 0.619 (0.591-0.639) | 0.494 (0.474-0.513) | 0.741 | 0.401 | 0.341 | 0.639-0.648 |

Paired by seed, correct of 606, MultiVerS vs NLI: author 349/380 (p=0.042), 337/388 (p=0.0015),
338/389 (p=0.00087); clean 321/380 (p=0.00021), 351/358 (p=0.68), 349/387 (p=0.015).
AUC vs ACC by type, NLI vs MultiVerS: ETIQUETTE 0.70-0.77 vs 0.59-0.70, NOT_SUBST 0.63-0.72 vs
0.55-0.63, CONTRADICT 0.68-0.76 vs 0.51-0.56, MISQUOTE 0.65-0.73 vs 0.59-0.68, OVERSIMPLIFY
0.50-0.53 vs 0.51-0.56 (chance for both). Rank average of the two adds at most +0.025 AUC.

## Trivial baseline for N_ACC F1

Predicting NOT_ACCURATE for all 606 test citations gives N_ACC F1 0.438 (precision 0.281, recall
1.0), macro 0.146. So N_ACC F1 alone cannot show skill near that value, including the paper's 0.43
and our 0.43-0.47; read it with precision, recall, macro-F1 and AUC.

## LLM on OVERSIMPLIFY (`scripts/llm_local.py`, CARC, two A40), gpt-oss-20b, no training

47 OVERSIMPLIFY + 47 sampled ACCURATE test citations, each judged with two evidence sources; greedy
decoding, fixed prompt, 188/188 parsed. AUC of overstatement_score, same citations for all models:

| Evidence | gpt-oss-20b AUC (approx 95% CI) | Flagged OVERSIMPLIFY / ACCURATE | Other models, AUC per seed |
|---|---:|---:|---|
| MonoT5 top-20 | 0.541 (0.42-0.66) | 0.43 / 0.34 | MultiVerS 0.531, 0.523, 0.577; NLI fine-tuned 0.558, 0.590, 0.580 |
| Gold | 0.730 (0.63-0.83) | 0.79 / 0.38 | MultiVerS trained on gold 0.576, 0.526, 0.533 |

Surface features of the gold inputs do not explain it (AUC: sentence count 0.547, evidence length
0.510, citance length 0.468, multi-citation 0.479, other-citation count 0.468). Annotators chose gold
evidence knowing the label, so content-level selection bias is not ruled out. CIs are Hanley-McNeil.

Paired, bootstrap over the 20 cited articles (`scripts/llm_compare.py`): LLM gold - LLM top-20 +0.189
[-0.029, +0.373]; LLM gold - MultiVerS gold +0.185 [+0.045, +0.301]; LLM top-20 - MultiVerS top-20 -0.009
[-0.176, +0.160]; LLM top-20 - fine-tuned NLI -0.041 [-0.189, +0.075]. Top-20 AUC 0.607 when all gold is
present (12 + 21 citations), 0.503 otherwise (35 + 26). Explanations show some gold-input flags rest on
the short evidence not mentioning a detail, including false alarms on ACCURATE citations.

## Checks against the paper's Supplementary Material (2026-10-30)

Run 02 (released checkpoint + our top-20) reproduces Table S3's fine-grained row distribution for all six
error types at two decimals (e.g. OVERSIMPLIFY 0.53/0.40/0.06, ETIQUETTE 0.38/0.58/0.04) and Table S2's
precision and recall (ACC 0.76/0.64, N_ACC 0.38/0.50, IRR 0.41/0.44). The supplement states lr 5e-05, but
the optimizer state in bestModel.ckpt has initial_lr 1e-05, as in train_target.py, so our recipe matches
the released checkpoint. All four GPT demonstrations are train citations (ids 101, 2076, 747, 943).

PubMedBERT top-20 arm (from the CARC log, 94/94 judged): AUC 0.577, flagged OVERSIMPLIFY 0.43, ACCURATE 0.38
(MonoT5 top-20: 0.541, 0.43 / 0.34; gold: 0.730, 0.79 / 0.38). Paired interval pending download of results.jsonl.

Gold hidden among 20 sentences (`goldfill`, from the CARC log, 94/94 judged): AUC 0.590, flagged OVERSIMPLIFY 0.45,
ACCURATE 0.34. With every gold sentence present, AUC stays near retrieved evidence and far below gold alone (0.730,
0.79 / 0.38). OVERSIMPLIFY flagging falls from 0.79 to 0.45 while ACCURATE stays near 0.34-0.38, so short evidence
does not simply raise all flags. Distractors and annotator selection of gold evidence remain unseparated.
Paired intervals pending download.

## Three-way LLM test, gpt-oss-20b, MonoT5 top-20, dev (`scripts/llm_threeclass.py`)

See the printed table of `--analyze`: improved prompt micro 0.67 / macro 0.61, paper's prompt 0.63 / 0.55, on dev. Prompt
for test: improved. Citations without a parsed answer count as wrong. Test not yet run.

PubMedBERT top-3 arm of the OVERSIMPLIFY test (94/94): AUC 0.582; PubMedBERT top-3 - top-20 +0.005 [-0.097, +0.110]; gold - top-3 +0.148
[-0.041, +0.326]. Fewer retrieved sentences do not reproduce the gold-only result. The three-way test files stopped at 440, 443 and 555 of 606 and,
being in file order, miss most no-evidence citations, so partial test scores are not valid; resubmit to finish.

### Three-way LLM test results (606/606 each, gpt-oss-20b, greedy, unparsed counted wrong)

| Run | Micro | Macro | ACC F1 | N_ACC P / R / F1 | IRR P / R / F1 | AUC N_ACC vs ACC |
|---|---:|---:|---:|---:|---:|---:|
| Paper GPT-4 (Table S2, top-5) | 0.65 | 0.45 | 0.80 | 0.66 / 0.05 / 0.09 | 0.35 / 0.74 / 0.48 | |
| Paper best MultiVerS = released ckpt, top-20 | 0.586 | 0.516 | 0.695 | 0.38 / 0.50 / 0.431 | 0.41 / 0.44 / 0.423 | 0.616 |
| paper prompt, top-5 (GPT-4's setting) | 0.625 | 0.548 | 0.76 | 0.58 / 0.41 / 0.48 | 0.26 / 0.86 / 0.40 | |
| improved prompt, MonoT5 top-20 | 0.637 | 0.574 | 0.72 | 0.43 / 0.74 / 0.545 | 0.56 / 0.38 / 0.45 | 0.725 |
| improved prompt, PubMedBERT top-20 | 0.642 | 0.599 | 0.73 | 0.44 / 0.72 / 0.542 | 0.56 / 0.50 / 0.53 | 0.729 |

Paired, 20 articles resampled: improved top-20 - released ckpt: macro +0.057 [-0.015, +0.132], N_ACC F1 +0.114 [+0.028, +0.206],
micro +0.051 [-0.014, +0.116]. PubMedBERT - MonoT5 (improved): macro +0.025 [-0.021, +0.064], N_ACC -0.003 [-0.031, +0.027].
improved top-20 - paper prompt top-5: macro +0.026 [-0.038, +0.088], N_ACC +0.064 [-0.001, +0.134].
Improved, AUC vs ACC by type (MonoT5 / PubMedBERT): ETIQUETTE 0.79 / 0.77, NOT_SUBST 0.83 / 0.78, CONTRADICT 0.84 / 0.89,
MISQUOTE 0.69 / 0.83, OVERSIMPLIFY 0.54 / 0.57. ACCURATE flagged NOT_ACCURATE 35%. Prompt chosen on dev; written after reading
the paper's aggregate GPT-4 test scores, without seeing any test prediction.

### More LLMs, dev (improved prompt, MonoT5 top-20, 316 each; dev 191 ACC / 101 N_ACC / 24 IRR)

Qwen/Qwen3.8-27B (thinking, reasoning_effort medium, seeded sampling T=1.0 top_p 0.95 top_k 20, 8192 tokens; 316/316 parsed, about
122 s each on two A40) and google/medgemma-27b-text-it (greedy, no thinking; 315/316 parsed, the other a repetition loop; max_new_tokens
lowered 2048 -> 512 after the speed check, which changes no answer since parsed answers average about 110 tokens).

| Model | Micro | Macro | N_ACC P / R / F1 | IRR P / R / F1 | AUC N_ACC vs ACC | Macro, irr>=15 |
|---|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b | 0.668 | 0.609 | 0.51 / 0.73 / 0.60 | 0.59 / 0.42 / 0.49 | 0.736 | 0.635 |
| Qwen3.8-27B | 0.690 | 0.491 | 0.51 / 0.74 / 0.60 | 0.33 / 0.04 / 0.07 | 0.796 | 0.602 |
| MedGemma 27B | 0.608 | 0.485 | 0.45 / 0.68 / 0.54 | 0.29 / 0.17 / 0.21 | 0.688 | 0.483 |

Qwen calls 23 of 24 gold IRRELEVANT citations NOT_ACCURATE (22 as NOT_SUBSTANTIATE) but scores them irrelevant_score 10-15 against
3-5 for the rest (AUC IRR vs rest 0.854; gpt-oss 0.879; MedGemma 0.563, scores only 0 or 100). Secondary rule fixed on dev before any
Qwen/MedGemma test run: IRRELEVANT when irrelevant_score >= 15 (IRR_THRESHOLD; 11-15 equal on dev for both gpt-oss and Qwen). Only 24
dev positives and Qwen's scores sit on 5-point steps, so the rule is fragile. `--analyze` prints it as a second line per file. Applied
to gpt-oss test (already run, rule chosen without looking at test): MonoT5 macro 0.574 -> 0.596, PubMedBERT 0.599 -> 0.597.
The improved prompt was written while reading gpt-oss dev outputs, which favours gpt-oss in this comparison.

### More LLMs, test (improved prompt, MonoT5 top-20, `scripts/llm_threeclass_compare.py`)

MedGemma 27B, 606/606 parsed: micro 0.540, macro 0.466, N_ACC 0.36 / 0.71 / 0.47, IRR 0.50 / 0.22 / 0.31, AUC 0.625; with irr>=15
macro 0.459. Paired, 20 articles resampled: MedGemma - released ckpt macro -0.050 [-0.119, +0.021], N_ACC F1 +0.041 [-0.041, +0.126];
gpt-oss - MedGemma macro +0.108 [+0.048, +0.163], N_ACC +0.073 [+0.022, +0.129]. With the dev-fixed rule, gpt-oss irr>=15 - released
ckpt macro +0.080 [+0.008, +0.157] (without the rule +0.057 [-0.015, +0.132]). Qwen test still running (178/606 at 7.5 h, about 150 s each).

Qwen3.8-27B test, 606/606 parsed (mean 1290 output tokens, max 4309): micro 0.635, macro 0.514, N_ACC 0.42 / 0.72 / 0.53,
IRR 1.00 / 0.16 / 0.28, AUC 0.755; with irr>=15 micro 0.644, macro 0.596, IRR 0.60 / 0.48 / 0.53. Gold IRRELEVANT -> predicted
[ACC, N_ACC, IRR]: gpt-oss [1, 30, 19], Qwen [1, 41, 8], MedGemma [2, 37, 11]. Paired, 20 articles resampled:
Qwen - released ckpt macro -0.002 [-0.079, +0.070], N_ACC F1 +0.096 [-0.002, +0.191]; Qwen irr>=15 - released ckpt macro +0.080
[+0.011, +0.146]; gpt-oss - Qwen macro +0.060 [-0.001, +0.118], N_ACC +0.018 [-0.034, +0.071]; with irr>=15 both 0.596, difference
+0.000 [-0.060, +0.062]; MedGemma - Qwen micro -0.096 [-0.149, -0.044].
AUC N_ACC vs ACC (556 ACC/N_ACC citations, score margin; released ckpt by p(N_ACC) - p(ACC)): Qwen - released +0.123 [+0.065, +0.186],
gpt-oss - released +0.093 [+0.035, +0.156], MedGemma - released -0.007 [-0.071, +0.058], Qwen - gpt-oss +0.030 [-0.017, +0.084].

## Evidence-seeking agent, gpt-oss-20b (`scripts/llm_agent.py`, after MedExAgent, arXiv 2605.07058)

Not fine-tuned (few-shot prompt): before judging, the model calls search(query) (BM25 k1 0.9 b 0.4 over every sentence of the cited
paper, top 5) and read(index) (index +-2), at most 8 calls, then answers with the improved prompt's JSON. Starts: `none`
(citing text + paper title) and `top20` (also the MonoT5 top-20, numbered by position in the paper). Plan: LIMIT=20 speed
check of both, dev for both, test once for the start chosen on dev. Data: data/agent (--prepare), bundle ci-carc-agent.tgz.

Dev, start `none` (316/316 parsed, about 81 s each; chunked prefill PREFILL_CHUNK=1024 added after the top20 speed check ran
out of memory; the earlier 20-citation speed checks were deleted and rerun): micro 0.655, macro 0.608, N_ACC 0.50 / 0.67 / 0.58,
IRR 0.50 / 0.54 / 0.52, AUC 0.708 (MonoT5 top-20, same prompt: 0.668 / 0.609, N_ACC 0.60, AUC 0.736). Paired over the 10 dev
articles: macro -0.001 [-0.089, +0.075], N_ACC F1 -0.023 [-0.062, +0.021]. Calls: mean 5.1, median 4 (1144 search, 459 read);
65 citations kept calling after the 8-call limit. Distinct sentences seen 14.6 on average. Gold coverage on the 255 dev citations
with evidence: agent 0.508 (all gold seen 0.36) vs MonoT5 top-20 0.550 (0.35); agent better on 58, worse on 69. The agent's own
search finds no more gold than MonoT5 and the labels tie. Search is a simplified BM25 (no stemming, no stopword list), unlike Pyserini.
Retrieval checks on the same 255 dev citations, no GPU: replaying the agent's 792 answered search queries through Pyserini
(Porter stemming, stopwords, the main pipeline's index) gives search-only gold coverage 0.348 vs 0.367 with our BM25 (same
top-5 per query 0.63; better on 20, worse on 28), so a better BM25 would not have found more gold. At the agent's own number of
sentences per citation (search + read), MonoT5 with the citance covers 0.499 vs the agent's 0.508; MonoT5 top-5 covers 0.322.
The agent's queries are about as good as MonoT5, not better (scripts/agent_requery.py).

## Select, then judge, gpt-oss-20b (`scripts/llm_select.py`)

Step 1: gpt-oss sees the citing text and the MonoT5 top-20 and returns the numbers of at most K sentences needed to check each
claim (none allowed). Step 2: the unchanged improved prompt with only those sentences (numbers and MonoT5 order kept; "(none)" if
none; all 20 if the selection is malformed). Dry run checked that rebuilding with all 20 gives the main run's request exactly.
Planned: K=5 and K=10 on dev, test once for the better one. Bundle ci-carc-select.tgz.

Dev results (316 each, all parsed; paired over the 10 dev articles against MonoT5 top-20 with the same prompt):

| Run | Micro | Macro | N_ACC F1 | IRR P / R / F1 | AUC | Macro irr>=15 | Macro diff vs top-20 |
|---|---:|---:|---:|---:|---:|---:|---|
| MonoT5 top-20 (main) | 0.668 | 0.609 | 0.60 | 0.59 / 0.42 / 0.49 | 0.736 | 0.635 | |
| agent, start none | 0.655 | 0.608 | 0.58 | 0.50 / 0.54 / 0.52 | 0.708 | 0.595 | -0.001 [-0.089, +0.075] |
| agent, start top20 | 0.649 | 0.596 | 0.56 | 0.62 / 0.42 / 0.50 | 0.694 | 0.631 | -0.013 [-0.084, +0.042] |
| select K=5 | 0.639 | 0.596 | 0.57 | 0.40 / 0.71 / 0.51 | 0.743 | 0.595 | -0.013 [-0.138, +0.081] |
| select K=10 | 0.665 | 0.635 | 0.60 | 0.47 / 0.75 / 0.58 | 0.742 | 0.622 | +0.026 [-0.095, +0.138] |

Agent top20: 3.6 calls on average, none on 62 citations. Selection: K=5 keeps 2.8 sentences on average, K=10 3.6; empty for 64 / 60
citations, including 21 of 24 gold IRRELEVANT (and 20 ACC, 19-23 N_ACC), which is where the IRRELEVANT recall gain comes from.
On citations with gold in the top-20, the selection keeps 0.55 (K=5) / 0.59 (K=10) of that gold, and 0.30 / 0.28 of the kept
sentences are gold, against 0.06 in the full top-20. One K=5 selection failed by naming sentence 20 of 0-19 (fell back to all 20).
select10 vs top-20 with irr>=15: macro -0.000 [-0.093, +0.090]. N_ACC F1 unchanged by every variant.
