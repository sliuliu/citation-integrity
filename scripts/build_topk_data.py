"""Build top-k retrieval datasets (paper Table 4, top-5 / top-20 rows).

Claims, labels and doc ids are copied unchanged from the authors' scifact_10
files; only each claim's document is replaced by our top-k reranked sentences
(monot5_rerank.py output), in rank order. With k=5 this should reproduce the
authors' own corpus.jsonl.
"""
import argparse
import json
import shutil
from pathlib import Path

import bm25_candidates
from ci_data import ROOT, read_jsonl

AUTHOR = ROOT / "data/author_scifact_10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reranked", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    _, sentences = bm25_candidates.load_articles()
    ranked = {(r["split"], r["id"]): r for r in read_jsonl(args.reranked)}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    docs = []
    for split in bm25_candidates.SPLITS:
        shutil.copy(AUTHOR / f"claims_{split}.jsonl", out / f"claims_{split}.jsonl")
        for c in read_jsonl(AUTHOR / f"claims_{split}.jsonl"):
            r = ranked[(split, c["id"])]
            docs.append({"doc_id": c["cited_doc_ids"][0], "title": "",
                         "abstract": [sentences[r["article"]][i] for i in r["ranked"][:args.k]]})
    with open(out / "corpus.jsonl", "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    lengths = [len(d["abstract"]) for d in docs]
    print(f"{len(docs)} docs, {sum(n < args.k for n in lengths)} shorter than {args.k} -> {out}")


if __name__ == "__main__":
    main()
