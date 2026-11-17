"""Paired comparisons for the LLM OVERSIMPLIFY test (results of llm_local.py).

AUC differences get 95% intervals from a bootstrap over cited articles, because
citations of one article share its evidence and are not independent. Also splits
the retrieved-evidence AUC by how much gold evidence retrieval actually delivered.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from llm_oversimplify import MV, ROOT, read_jsonl

OUT = ROOT / "runs/llm-oversimplify-gptoss20b"
RETRIEVED = {"top20": "topk/top20", "pubmedbert20": "topk/pubmedbert20"}
MODELS = {"MultiVerS top-20 (mean of seeds)": "topk20-author-*", "NLI fine-tuned top-20 (mean of seeds)": "nli-top20-author-*",
          "MultiVerS gold (mean of seeds)": "oracle-citance-A-author-*"}


def seed_mean_scores(pattern, ids):
    "Rank-averaged p(NOT_ACCURATE) of the final epoch across seeds."
    from scipy.stats import rankdata

    ranks = []
    for d in sorted(glob.glob(str(ROOT / "runs" / pattern))):
        if not Path(d, "summary.json").exists():
            continue
        epoch = json.load(open(Path(d, "history.json")))[-1]["epoch"]
        prob = {r["id"]: r["probs"]["NOT_ACCURATE"] for r in read_jsonl(Path(d, f"epoch{epoch}/predictions_test.jsonl"))}
        ranks.append(rankdata([prob[i] for i in ids]))
    return np.mean(ranks, axis=0)


def boot_diff(y, a, b, groups, n=5000, seed=0):
    "Observed AUC(a) - AUC(b), with a percentile interval resampling whole articles."
    rng = np.random.default_rng(seed)
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    keys = list(members)
    diffs = []
    for _ in range(n):
        idx = np.concatenate([members[keys[k]] for k in rng.integers(len(keys), size=len(keys))])
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return roc_auc_score(y, a) - roc_auc_score(y, b), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = read_jsonl(Path(args.out) / "results.jsonl")
    llm = {s: {r["id"]: r for r in rows if r["source"] == s and "overstatement_score" in r}
           for s in {r["source"] for r in rows}}
    ids = sorted(llm["gold"])
    y = np.array([llm["gold"][i]["gold"] == "OVERSIMPLIFY" for i in ids])
    mv = {c["id"]: c for c in read_jsonl(MV / "claims-test.jsonl")}
    corpus = {d["doc_id"]: d["abstract"] for d in read_jsonl(MV / "corpus.jsonl")}
    groups = np.array([min(mv[i]["cited_doc_ids"]) // 1000 for i in ids])
    print(f"{len(ids)} citations from {len(set(groups))} cited articles")

    score = {s: np.array([llm[s][i]["overstatement_score"] for i in ids], dtype=float) for s in llm}
    for s, v in score.items():
        print(f"LLM {s:13s} AUC {roc_auc_score(y, v):.3f}")

    print("\nPaired differences in AUC (95% interval, articles resampled):")
    pairs = [("gold", "top20"), ("pubmedbert20", "top20"), ("gold", "pubmedbert20"), ("goldfill", "top20"),
             ("goldfill", "pubmedbert20"), ("gold", "goldfill"), ("pubmedbert3", "pubmedbert20"), ("gold", "pubmedbert3")]
    pairs = [p for p in pairs if p[0] in score and p[1] in score]
    for a, b in pairs:
        d, lo, hi = boot_diff(y, score[a], score[b], groups)
        print(f"  LLM {a} - LLM {b}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    for name, pattern in MODELS.items():
        other = seed_mean_scores(pattern, ids)
        source = "gold" if "gold" in name else "top20"
        d, lo, hi = boot_diff(y, score[source], other, groups)
        print(f"  LLM {source} - {name} (AUC {roc_auc_score(y, other):.3f}): {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    print("\nRetrieved evidence, by share of the citation's gold sentences delivered:")
    for source, data in RETRIEVED.items():
        if source not in score:
            continue
        docs = {d["doc_id"]: set(d["abstract"]) for d in read_jsonl(ROOT / "data" / data / "corpus.jsonl")}
        author = {c["id"]: c for c in read_jsonl(ROOT / "data/author_scifact_10/claims_test.jsonl")}
        coverage = []
        for i in ids:
            gold = {corpus[int(d)][s] for d, es in mv[i]["evidence"].items() for e in es for s in e["sentences"]}
            coverage.append(len(gold & docs[author[i]["cited_doc_ids"][0]]) / len(gold))
        coverage = np.array(coverage)
        for label, keep in (("all gold present", coverage == 1), ("some or none", coverage < 1)):
            n_pos, n_neg = int(y[keep].sum()), int((~y[keep]).sum())
            auc = roc_auc_score(y[keep], score[source][keep]) if n_pos and n_neg else float("nan")
            print(f"  {source:13s} {label:17s} {n_pos:2d} OVERSIMPLIFY, {n_neg:2d} ACCURATE: AUC {auc:.3f}")


if __name__ == "__main__":
    main()
