"""Paired comparisons for the three-way LLM test (results of llm_threeclass.py) on the test split.

Each LLM run is compared with the released MultiVerS checkpoint on our top-20 (run 02, the paper's best row) and
with the other LLM runs. 95% intervals come from a bootstrap over cited articles, because citations of one article
share its evidence. Only complete runs (606 citations) are used; an unparsed answer counts as wrong.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from llm_oversimplify import ROOT, read_jsonl
from llm_threeclass import LABELS, threshold_label

RELEASED = ROOT / "runs/02-released-ckpt-our-top20-test/predictions.jsonl"
METRICS = {
    "macro": lambda g, p: f1_score(g, p, labels=LABELS, average="macro"),
    "N_ACC F1": lambda g, p: f1_score(g, p, labels=["NOT_ACCURATE"], average="macro", zero_division=0),
    "micro": lambda g, p: float(np.mean(g == p)),
}


def boot(gold, a, b, groups, metric, n=5000, seed=0):
    "Observed metric(a) - metric(b), with a percentile interval resampling whole articles."
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    diffs = []
    for _ in range(n):
        idx = np.concatenate([members[k] for k in rng.integers(len(members), size=len(members))])
        diffs.append(metric(gold[idx], a[idx]) - metric(gold[idx], b[idx]))
    return metric(gold, a) - metric(gold, b), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["improved-topk_top20"],
                    help="result file names without split, e.g. improved-topk_top20 agent-none")
    args = ap.parse_args()

    claims = read_jsonl(ROOT / "data/topk/top20/claims_test.jsonl")
    ids = [c["id"] for c in claims]
    groups = np.array([c["cited_doc_ids"][0] // 1000 for c in claims])
    released = {r["id"]: r for r in read_jsonl(RELEASED)}
    gold = np.array([released[i]["gold"] for i in ids])
    systems = {"released ckpt": np.array([released[i]["pred"] for i in ids])}
    paths = sorted(p for run in args.runs for p in glob.glob(str(ROOT / f"runs/llm-threeclass-*/test-{run}.jsonl")))
    for path in paths:
        rows = {r["id"]: r for r in read_jsonl(path)}
        model = Path(path).parent.name[len("llm-threeclass-"):]
        if len(args.runs) > 1:
            model += "/" + Path(path).stem[len("test-"):]
        if len(rows) != len(ids):
            print(f"skipping {model}: {len(rows)}/{len(ids)} citations")
            continue
        assert all(rows[i]["gold"] == g for i, g in zip(ids, gold))
        systems[model] = np.array([rows[i].get("label", "UNJUDGED") for i in ids])
        systems[f"{model} irr>=15"] = np.array([threshold_label(rows[i]) for i in ids])
    print(f"{len(ids)} citations from {len(set(groups))} cited articles, runs {' '.join(args.runs)}\n")

    for name, pred in systems.items():
        print(f"{name:24s} " + "  ".join(f"{m} {f(gold, pred):.3f}" for m, f in METRICS.items()))

    print("\nPaired differences (95% interval, articles resampled):")
    names = list(systems)
    pairs = [(a, "released ckpt") for a in names[1:]]
    llms = [a for a in names[1:] if "irr" not in a]
    pairs += [(a, b) for i, a in enumerate(llms) for b in llms[i + 1:]]
    pairs += [(f"{a} irr>=15", f"{b} irr>=15") for i, a in enumerate(llms) for b in llms[i + 1:]]
    for a, b in pairs:
        cells = []
        for m, f in METRICS.items():
            d, lo, hi = boot(gold, systems[a], systems[b], groups, f)
            cells.append(f"{m} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"  {a} - {b}: " + "  ".join(cells))


if __name__ == "__main__":
    main()
