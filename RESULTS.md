# Results (as of 2026-11-17)

All scores are on the test split unless marked dev: 606 citations (386 ACCURATE / 170 NOT_ACCURATE /
50 IRRELEVANT) to 20 cited papers. "Micro" is accuracy; "macro" is macro-F1 over the three classes.
AUC is the ROC AUC of NOT_ACCURATE vs ACCURATE, which shows how well a model separates the two
regardless of its threshold. 95% intervals in brackets come from a bootstrap that resamples whole cited
papers, because citations of one paper share their evidence. Details and every check are in
[runs/README.md](runs/README.md).

## Summary

1. **The paper's main result is reproduced exactly.** The authors' released checkpoint with our top-20
   retrieval matches every published top-20 number, including the per-error-type breakdown in the
   supplement.
2. **Better retrieval does not help the classifier.** A stronger reranker (PubMedBERT) finds 12 points
   more of the gold evidence, but MultiVerS scores do not improve, frozen or retrained.
3. **Zero-shot LLMs beat the paper's best model on NOT_ACCURATE.** gpt-oss-20b with our prompt raises
   NOT_ACCURATE F1 from .431 to .545, with an interval that excludes zero. The paper reported GPT-4 at
   .09.
4. **OVERSIMPLIFY stays unsolved.** Every method is near chance unless it is given only the
   annotator-chosen gold sentences.
5. **Letting the LLM gather or filter its own evidence ties the fixed top-20 on dev.** The one test run
   still pending is select-then-judge with K=10.

## 1. Reproduction

| | Micro | Macro | ACC F1 | N_ACC F1 | IRR F1 |
|---|---:|---:|---:|---:|---:|
| Paper, top-20 | 0.59 | 0.52 | 0.69 | 0.43 | 0.42 |
| **Ours: released checkpoint + our BM25/MonoT5 top-20** | **0.586** | **0.516** | **0.695** | **0.431** | **0.423** |
| Ours: retrained with the authors' recipe on top-20 (3 seeds) | 0.563 | 0.505 | 0.665 | 0.432 | 0.418 |
| Paper, top-5 | 0.58 | 0.50 | | | |
| Ours: released checkpoint, top-5 | 0.558 | 0.444 | 0.693 | 0.312 | 0.326 |

- Our retrieval returns the same top-5 as the authors' data for 3,058 of 3,060 citations.
- The released checkpoint was trained on top-5 evidence. The paper's top-20 row is that model evaluated
  on top-20 evidence.
- The supplement says the learning rate was 5e-5, but the checkpoint's optimizer state says 1e-5. We
  used 1e-5.

**Gold-evidence oracle.** The paper reports micro .73 / macro .75 and IRR F1 .93. We reach .79 / .72
and IRR .98 only when citations without annotated evidence are labelled IRRELEVANT. That is the
convention in the authors' code, but not the one in the paper's text (labels A vs B in the run log).
NOT_ACCURATE F1 stays at .35 against the paper's .52, across three seeds. This gap is unexplained.

## 2. Evidence controls

| Evidence (released checkpoint unless noted) | Gold coverage | Micro | Macro |
|---|---:|---:|---:|
| Random 20 sentences | | 0.439 | 0.410 |
| MonoT5 top-20 | 0.513 | 0.586 | 0.516 |
| PubMedBERT top-20 (team reranker) | 0.634 | 0.548 | 0.481 |
| PubMedBERT top-20, retrained (3 seeds) | 0.634 | 0.549 | 0.477 |

Retrieval matters compared with random sentences. Beyond that, more gold coverage does not raise the
scores.

## 3. Improving NOT_ACCURATE

Reference point: labelling every citation NOT_ACCURATE already gives N_ACC F1 .438 (macro .146).
N_ACC F1 should therefore be read together with macro-F1 and AUC.

| Method (MonoT5 top-20) | Micro | Macro | N_ACC F1 | AUC |
|---|---:|---:|---:|---:|
| Paper GPT-4 (paper's prompt, top-5) | 0.65 | 0.45 | 0.09 | |
| Released MultiVerS checkpoint (paper's best) | 0.586 | 0.516 | 0.431 | 0.616 |
| Extra NOT_ACCURATE loss weight | | +0.014 at best | | unchanged |
| Fine-tuned DeBERTa NLI (3 seeds) | 0.636 | 0.524 | 0.446 | 0.67-0.68 |
| gpt-oss-20b, paper's prompt, top-5 | 0.625 | 0.548 | 0.48 | |
| **gpt-oss-20b, our prompt** | **0.637** | **0.574** | **0.545** | **0.725** |
| gpt-oss-20b, our prompt, PubMedBERT top-20 | 0.642 | 0.599 | 0.542 | 0.729 |
| Qwen3.8-27B (thinking), our prompt | 0.635 | 0.514 | 0.528 | 0.755 |
| MedGemma 27B (medical), our prompt | 0.540 | 0.466 | 0.472 | 0.625 |

The LLMs are zero-shot and use no training data beyond eight train examples in the prompt. Our prompt
spells out the paper's annotation guidelines and gives one example per error type.

Paired against the released checkpoint:

| | Macro | N_ACC F1 | AUC |
|---|---|---|---|
| gpt-oss-20b | +0.057 [-0.015, +0.132] | **+0.114 [+0.028, +0.206]** | **+0.093 [+0.035, +0.156]** |
| Qwen3.8-27B | -0.002 [-0.079, +0.070] | +0.096 [-0.002, +0.191] | **+0.123 [+0.065, +0.186]** |
| MedGemma 27B | -0.050 [-0.119, +0.021] | +0.041 [-0.041, +0.126] | -0.007 [-0.071, +0.058] |

- **Qwen rarely answers IRRELEVANT.** It labels 41 of the 50 IRRELEVANT citations NOT_ACCURATE, but it
  still gives them a higher irrelevant score.
- **A threshold fixed on dev recovers them.** The rule is: answer IRRELEVANT when irrelevant_score
  ≥ 15. With it, macro reaches .596 for both gpt-oss and Qwen, +.080 over the checkpoint, with
  intervals [+.008, +.157] and [+.011, +.146]. This is a secondary analysis.
- **The medical model does worst.** MedGemma is significantly behind gpt-oss on macro (+.108
  [+.048, +.163] for gpt-oss).

## 4. OVERSIMPLIFY and the gold-evidence puzzle

This test uses the 47 OVERSIMPLIFY citations and 47 sampled ACCURATE ones. The table gives the AUC of
gpt-oss-20b's overstatement score.

| Evidence given to the LLM | AUC |
|---|---:|
| MonoT5 top-20 | 0.541 |
| PubMedBERT top-20 | 0.577 |
| PubMedBERT top-3 | 0.582 |
| All gold sentences hidden among 20 | 0.590 |
| **Gold sentences only** | **0.730** |
| (MultiVerS trained and tested on gold) | 0.53-0.58 |

Complete gold coverage is not enough: the gold sentences only help when nothing else is shown. Either
the distractors hurt, or the annotators' choice of evidence (made while knowing the label) carries
information. These two explanations cannot yet be separated.

## 5. Letting the LLM choose its evidence (dev only, gpt-oss-20b)

| Method | Micro | Macro | N_ACC F1 | IRR F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Fixed MonoT5 top-20 | 0.668 | 0.609 | 0.60 | 0.49 | 0.736 |
| Agent with search/read tools over the whole paper (after MedExAgent) | 0.655 | 0.608 | 0.58 | 0.52 | 0.708 |
| Same agent, starting from the top-20 | 0.649 | 0.596 | 0.56 | 0.50 | 0.694 |
| Select ≤5 of the top-20, then judge | 0.639 | 0.596 | 0.57 | 0.51 | 0.743 |
| **Select ≤10 of the top-20, then judge** | 0.665 | **0.635** | 0.60 | **0.58** | 0.742 |

Dev has only 10 papers, so every difference from the fixed top-20 has an interval that includes zero.

- **The agent finds no more gold than MonoT5.** Its gold coverage is .51, against .50 for MonoT5 at the
  same number of sentences. Replaying its queries through Pyserini BM25 does not help either.
- **Selection makes the evidence denser but less complete.** The share of gold among the kept sentences
  rises from 6% to about 30%, but about 40% of the gold that was in the top-20 is dropped.
- **K=10 gains only on IRRELEVANT.** An empty selection works as an IRRELEVANT signal: 21 of the 24
  IRRELEVANT citations get one. NOT_ACCURATE does not change.
- **Pending:** the test run for K=10.

## Limitations and open questions

1. **Label convention for the oracle.** Citations without annotated evidence can be labelled IRRELEVANT
   (as in the authors' code) or not (as in the paper's text). The oracle row is reproduced only under
   the first convention.
2. **Caveats on the LLM result:**
   - The prompt was written while looking at gpt-oss dev outputs.
   - Qwen samples, and we ran it with one seed only.
   - The IRRELEVANT threshold is a post-hoc rule, even though it was fixed on dev before the test run.
3. **The gold-evidence gap.** It remains the main open lever for OVERSIMPLIFY.
