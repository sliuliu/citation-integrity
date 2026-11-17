"""Combine MultiVerS with zero-shot NLI scores and citance features.

Stage 1 (train split): multinomial logistic regression on features that do not
come from MultiVerS (NLI aggregates over the evidence sentences, citation-marker
counts, citance length), so the classifier's own overfit train predictions are
never used.
Stage 2 (dev split): multinomial logistic regression on MultiVerS log-probs plus
the stage-1 log-probs, using a MultiVerS run trained on train only ("clean"), for
which dev is held out. A stage-2 model on MultiVerS log-probs alone is the
control: it shows how much comes from refitting the decision rule on dev rather
than from the NLI features. Test is scored once.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from ci_data import LABELS, ROOT, evaluate, read_jsonl

AUTHOR = ROOT / "data/author_scifact_10"


def citance_features(claim):
    return [claim.count("<|multi_cit|>") > 0, claim.count("<|other_cit|>"), claim.count("<|cit|>"),
            np.log1p(len(claim.split()))]


def nli_features(probs):
    P = np.array(probs) if probs else np.full((1, 3), 1 / 3)
    ent, neu, con = P[:, 0], P[:, 1], P[:, 2]
    return [ent.max(), ent.mean(), con.max(), con.mean(), neu.mean(), ent[0], con[0],
            (ent > 0.5).mean(), con.max() - ent.max()]


def features(split, nli_dir, data):
    claims = {c["id"]: c["claim"] for c in read_jsonl(AUTHOR / f"claims_{split}.jsonl")}
    rows = read_jsonl(Path(nli_dir) / f"{data.replace('/', '_')}_{split}.jsonl")
    X = np.array([nli_features(r["probs"]) + citance_features(claims[r["id"]]) for r in rows], dtype=float)
    return [r["id"] for r in rows], X, np.array([LABELS.index(r["gold"]) for r in rows])


def multivers_logprobs(path, ids):
    p = {r["id"]: [r["probs"][k] for k in LABELS] for r in read_jsonl(path)}
    return np.log(np.clip(np.array([p[i] for i in ids]), 1e-6, 1))


def report(name, y, prob):
    pred = [LABELS[k] for k in prob.argmax(1)]
    m = evaluate([LABELS[k] for k in y], pred)
    keep = np.isin(y, [LABELS.index("NOT_ACCURATE"), LABELS.index("ACCURATE")])
    auc = roc_auc_score(y[keep] == LABELS.index("NOT_ACCURATE"), prob[keep, LABELS.index("NOT_ACCURATE")])
    print(f"  {name:28s} micro {m['micro_f1']:.3f} macro {m['macro_f1']:.3f} "
          f"ACC {m['per_class']['ACCURATE']['f1']:.3f} N_ACC {m['per_class']['NOT_ACCURATE']['f1']:.3f} "
          f"IRR {m['per_class']['IRRELEVANT']['f1']:.3f} | AUC {auc:.3f}")
    return m


def lr():
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="topk/top20")
    ap.add_argument("--runs", nargs="+", default=["topk20-clean-13511512", "topk20-clean-s1-13511513",
                                                  "topk20-clean-s2-13511514"])
    ap.add_argument("--nli-dir", default=str(ROOT / "runs/nli-diagnostic"))
    args = ap.parse_args()

    splits = {s: features(s, args.nli_dir, args.data) for s in ("train", "dev", "test")}
    scaler = StandardScaler().fit(splits["train"][1])
    stage1 = lr().fit(scaler.transform(splits["train"][1]), splits["train"][2])
    s1 = {s: np.log(stage1.predict_proba(scaler.transform(X))) for s, (_, X, _) in splits.items()}
    print("Stage 1 alone (NLI + citance features, fit on train):")
    report("test", splits["test"][2], np.exp(s1["test"]))

    for run in args.runs:
        d = ROOT / "runs" / run
        ep = json.load(open(d / "summary.json"))["best_epoch"]
        mv = {s: multivers_logprobs(d / f"epoch{ep}/predictions_{s}.jsonl", splits[s][0]) for s in ("dev", "test")}
        y_dev, y_test = splits["dev"][2], splits["test"][2]
        print(f"{run} (epoch {ep}), test:")
        report("MultiVerS as trained", y_test, np.exp(mv["test"]))
        control = lr().fit(mv["dev"], y_dev)
        report("MultiVerS refit on dev", y_test, control.predict_proba(mv["test"]))
        both = lr().fit(np.hstack([mv["dev"], s1["dev"]]), y_dev)
        report("MultiVerS + stage 1, on dev", y_test, both.predict_proba(np.hstack([mv["test"], s1["test"]])))


if __name__ == "__main__":
    main()
