"""Would a heavier NOT_ACCURATE loss weight help? Estimate it from saved probabilities.

Scaling a class's loss weight by w shifts its logit by about log(w), so we scale
p(NOT_ACCURATE) by w at prediction time instead of retraining. For runs trained
on train only ("clean"), w is chosen on dev (by macro-F1) and applied to test.
Also reports the test-optimal w (an upper bound, not a valid result) and the
ROC AUC of p(NOT_ACCURATE) for gold NOT_ACCURATE vs gold ACCURATE, which no
reweighting can change.
"""
import argparse
import glob
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from ci_data import LABELS, ROOT, evaluate, read_jsonl

WEIGHTS = np.exp(np.linspace(np.log(0.25), np.log(8), 41))


def predict(rows, w):
    out = []
    for r in rows:
        p = dict(r["probs"])
        p["NOT_ACCURATE"] *= w
        out.append(max(LABELS, key=p.get))
    return out


def scores(rows, w):
    m = evaluate([r["gold"] for r in rows], predict(rows, w))
    return m["macro_f1"], m["per_class"]["NOT_ACCURATE"]["f1"], m["micro_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", default=["topk20-clean*", "pubmedbert20-clean*", "oracle-citance-A-clean*"])
    args = ap.parse_args()

    for pattern in args.glob:
        for run in sorted(glob.glob(str(ROOT / "runs" / pattern / "summary.json"))):
            d = run[:-len("/summary.json")]
            ep = json.load(open(run))["best_epoch"]
            dev = read_jsonl(f"{d}/epoch{ep}/predictions_dev.jsonl")
            test = read_jsonl(f"{d}/epoch{ep}/predictions_test.jsonl")
            w_dev = max(WEIGHTS, key=lambda w: scores(dev, w)[0])
            w_test = max(WEIGHTS, key=lambda w: scores(test, w)[0])
            pair = [r for r in test if r["gold"] in ("NOT_ACCURATE", "ACCURATE")]
            auc = roc_auc_score([r["gold"] == "NOT_ACCURATE" for r in pair], [r["probs"]["NOT_ACCURATE"] for r in pair])
            base, tuned, best = scores(test, 1.0), scores(test, w_dev), scores(test, w_test)
            print(f"{d.split('/')[-1]:32s} ep{ep} AUC {auc:.3f} | test macro/N_ACC/micro "
                  f"w=1: {base[0]:.3f}/{base[1]:.3f}/{base[2]:.3f}  dev-tuned w={w_dev:.2f}: "
                  f"{tuned[0]:.3f}/{tuned[1]:.3f}/{tuned[2]:.3f}  test-best w={w_test:.2f}: "
                  f"{best[0]:.3f}/{best[1]:.3f}/{best[2]:.3f}")


if __name__ == "__main__":
    main()
