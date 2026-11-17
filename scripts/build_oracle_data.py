"""Build gold-evidence oracle datasets (paper Table 4, "Oracle" rows).

Each claim gets one document containing the union of its annotated evidence
sentences from Data/multivers-format, ordered by reference chunk and sentence
position. Claims without annotated evidence get an empty document.

Variants:
  query   citance  claim text from the authors' scifact_10 files (Oracle, gold evidence)
          goldctx  claim text from multivers-format, the annotated citation
                   context (Oracle, gold citation + gold evidence)
  labels  A  three-way labels from scifact_10 (paper Section 2.5.3: ETIQUETTE
             -> NOT_ACCURATE, even when it has no evidence)
          B  the authors' reader applied to multivers-format: a claim without
             evidence is NEI, i.e. IRRELEVANT, including ETIQUETTE
"""
import argparse
import collections
import json
from pathlib import Path

from ci_data import ROOT, read_jsonl

MV = ROOT / "upstream/Citation-Integrity/Data/multivers-format"
AUTHOR = ROOT / "data/author_scifact_10"
SPLITS = ["train", "dev", "test"]
DOC_OFFSET = {"train": 1_000_000, "dev": 2_000_000, "test": 3_000_000}


def author_label(claim):
    labels = {e["label"] for es in claim["evidence"].values() for e in es}
    assert len(labels) <= 1
    return labels.pop() if labels else "IRRELEVANT"


def gold_sentences(mv_claim, corpus):
    order = {doc_id: i for i, doc_id in enumerate(mv_claim["cited_doc_ids"])}
    keys = set()
    for doc_id, rationales in mv_claim["evidence"].items():
        for r in rationales:
            for s in r["sentences"]:
                keys.add((order[int(doc_id)], int(doc_id), s))
    return [corpus[doc_id]["abstract"][s] for _, doc_id, s in sorted(keys)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT / "data/oracle"))
    args = ap.parse_args()

    corpus = {d["doc_id"]: d for d in read_jsonl(MV / "corpus.jsonl")}
    built = {}
    for split in SPLITS:
        mv = {c["id"]: c for c in read_jsonl(MV / f"claims-{split}.jsonl")}
        author = read_jsonl(AUTHOR / f"claims_{split}.jsonl")
        assert set(c["id"] for c in author) <= set(mv)
        built[split] = [(a, mv[a["id"]], gold_sentences(mv[a["id"]], corpus)) for a in author]

    summary = {}
    for query in ["citance", "goldctx"]:
        for conv in ["A", "B"]:
            out = Path(args.out_root) / f"{query}-{conv}"
            out.mkdir(parents=True, exist_ok=False)
            docs, stats = [], {}
            for split in SPLITS:
                rows, counts = [], collections.Counter()
                for a, m, sents in built[split]:
                    doc_id = DOC_OFFSET[split] + a["id"]
                    docs.append({"doc_id": doc_id, "title": "", "abstract": sents})
                    label = author_label(a)
                    if conv == "B" and not m["evidence"]:
                        label = "IRRELEVANT"
                    counts[label] += 1
                    counts["empty_evidence"] += not sents
                    evidence = {} if label == "IRRELEVANT" else {
                        str(doc_id): [{"sentences": list(range(len(sents))), "label": label}]}
                    rows.append({"id": a["id"], "claim": a["claim"] if query == "citance" else m["claim"],
                                 "evidence": evidence, "cited_doc_ids": [doc_id]})
                with open(out / f"claims_{split}.jsonl", "w") as f:
                    f.writelines(json.dumps(r) + "\n" for r in rows)
                stats[split] = dict(counts, n=len(rows))
            with open(out / "corpus.jsonl", "w") as f:
                f.writelines(json.dumps(d) + "\n" for d in docs)
            summary[out.name] = stats
            (out / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
