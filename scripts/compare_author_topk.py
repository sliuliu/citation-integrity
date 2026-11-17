"""Compare reranked lists with the authors' scifact_10 evidence documents.

The authors' corpus documents are their retrieved sentences (top 5), in
rank order. Reports per split how often our top-k equals theirs exactly (same
order), as a set, and the overlap of the sentence sets.
"""
import argparse
import collections
import json

import bm25_candidates
from ci_data import ROOT, read_jsonl

AUTHOR = ROOT / "data/author_scifact_10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranked", required=True)
    args = ap.parse_args()

    _, sentences = bm25_candidates.load_articles()
    corpus = {d["doc_id"]: d["abstract"] for d in read_jsonl(AUTHOR / "corpus.jsonl")}
    theirs = {(split, c["id"]): corpus[c["cited_doc_ids"][0]]
              for split in bm25_candidates.SPLITS for c in read_jsonl(AUTHOR / f"claims_{split}.jsonl")}
    stats = collections.defaultdict(collections.Counter)
    for row in read_jsonl(args.reranked):
        ref = theirs[(row["split"], row["id"])]
        ours = [sentences[row["article"]][i] for i in row["ranked"][:len(ref)]]
        s = stats[row["split"]]
        s["n"] += 1
        s["same_order"] += ours == ref
        s["same_set"] += set(ours) == set(ref)
        s["top1"] += ours[:1] == ref[:1]
        s["overlap"] += len(set(ours) & set(ref))
        s["sents"] += len(set(ref))
    for split, s in stats.items():
        print(f"{split:5s} n={s['n']:4d} same order {s['same_order'] / s['n']:.3f} same set {s['same_set'] / s['n']:.3f} "
              f"top1 {s['top1'] / s['n']:.3f} sentence overlap {s['overlap'] / s['sents']:.3f}")


if __name__ == "__main__":
    main()
