"""Build a top-k dataset from the team toolkit's reranker rankings.

external/team-toolkit/scripts/train_reranker.py writes one JSON line per claim,
{"id", "ranked": ["<multivers doc_id>:<sentence index>", ...]}, over the
multivers-format corpus. As in build_topk_data.py, claims, labels and doc ids
are copied unchanged from the authors' scifact_10 files and only each claim's
document is replaced, here by the reranker's top-k sentences in rank order.
"""
import argparse
import json
import shutil
from pathlib import Path

from ci_data import ROOT, get_tokenizer, read_jsonl

AUTHOR = ROOT / "data/author_scifact_10"
MV = ROOT / "upstream/Citation-Integrity/Data/multivers-format"
SPLITS = ["train", "dev", "test"]


def main():
    ap = argparse.ArgumentParser()
    for split in SPLITS:
        ap.add_argument(f"--{split}", nargs="+", required=True, help=f"rankings for {split} (fold files allowed)")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    corpus = {d["doc_id"]: d["abstract"] for d in read_jsonl(MV / "corpus.jsonl")}
    tokenizer = get_tokenizer()
    eos = tokenizer.eos_token
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    docs = []
    for split in SPLITS:
        ranked = {}
        for path in getattr(args, split):
            for r in read_jsonl(path):
                assert r["id"] not in ranked, (split, r["id"])
                ranked[r["id"]] = r["ranked"]
        shutil.copy(AUTHOR / f"claims_{split}.jsonl", out / f"claims_{split}.jsonl")
        claims = read_jsonl(AUTHOR / f"claims_{split}.jsonl")
        missing = [c["id"] for c in claims if c["id"] not in ranked]
        assert not missing, f"{split}: {len(missing)} claims without rankings, e.g. {missing[:5]}"
        dropped = 0
        for c in claims:
            keys = [k.split(":") for k in ranked[c["id"]][:args.k]]
            sents = [corpus[int(d)][int(s)] for d, s in keys]
            # Long table sentences can push an input past Longformer's 4096 tokens, which
            # the upstream tokenizer rejects; drop the lowest-ranked sentences until it fits.
            while len(tokenizer(eos.join([c["claim"], ""] + sents))["input_ids"]) > 4096:
                sents.pop()
                dropped += 1
            docs.append({"doc_id": c["cited_doc_ids"][0], "title": "", "abstract": sents})
        print(f"{split}: {len(claims)} claims, {dropped} lowest-ranked sentences dropped to fit 4096 tokens")
    with open(out / "corpus.jsonl", "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in docs)
    print(f"{len(docs)} docs, {sum(len(d['abstract']) < args.k for d in docs)} shorter than {args.k} -> {out}")


if __name__ == "__main__":
    main()
