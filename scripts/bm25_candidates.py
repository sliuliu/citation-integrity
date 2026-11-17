"""BM25 candidates for each claim, following the authors' bm25t5.ipynb.

One Lucene index per cited reference article over all its sentences
(annotations.zip references_sentence/, in file order, ids sent_N), built with
the notebook's pyserini.index.lucene flags, then search(query, k=60) with the
default BM25(k1=0.9, b=0.4). Runs in .venv-bm25 (Pyserini 0.24.0, Java 11):

  JAVA_HOME=.runtime/jdk-11.0.32.1+1 .venv-bm25/bin/python scripts/bm25_candidates.py \
      --claims-dir data/author_scifact_10 --out data/retrieval/bm25.jsonl

The Anserini SimpleSearcher is called directly; pyserini.search.lucene wraps
the same call but imports faiss/torch at module level.
"""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "upstream/Citation-Integrity/Data/annotations.zip"
MV = ROOT / "upstream/Citation-Integrity/Data/multivers-format"
SPLITS = ["train", "dev", "test"]


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_articles():
    "par_id -> article key, and article key -> sentences in file order."
    par_to_article, sentences = {}, {}
    with zipfile.ZipFile(ANNOTATIONS) as z:
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if parts[0] != "annotations" or len(parts) != 4 or parts[2] != "references_sentence" \
                    or not parts[3].endswith(".json"):
                continue
            key = f"{parts[1]}/{parts[3][:-5]}"
            sentences[key] = []
            for line in z.read(name).decode().splitlines():
                if line.strip():
                    par = json.loads(line)
                    par_to_article[par["par_id"]] = key
                    sentences[key] += par["sentences"]
    return par_to_article, sentences


def build_index(sents, work):
    index = work / "index"
    if (index / "done").exists():
        return index
    docs = work / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "documents.json").write_text(json.dumps([{"id": f"sent_{i}", "contents": s} for i, s in enumerate(sents)]))
    subprocess.run([sys.executable, "-m", "pyserini.index.lucene", "--collection", "JsonCollection",
                    "--input", str(docs), "--index", str(index), "--generator", "DefaultLuceneDocumentGenerator",
                    "--threads", "1", "--storePositions", "--storeDocvectors", "--storeRaw"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (index / "done").write_text("")
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims-dir", required=True, help="claims_{split}.jsonl; their claim text is the query")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--work-dir", default=str(ROOT / ".work/bm25"))
    args = ap.parse_args()

    from pyserini.pyclass import autoclass
    searcher_cls = autoclass("io.anserini.search.SimpleSearcher")

    par_to_article, sentences = load_articles()
    claims = []
    for split in SPLITS:
        mv = {c["id"]: c for c in read_jsonl(MV / f"claims-{split}.jsonl")}
        for c in read_jsonl(Path(args.claims_dir) / f"claims_{split}.jsonl"):
            articles = {par_to_article[p] for p in mv[c["id"]]["cited_doc_ids"]}
            assert len(articles) == 1, c["id"]
            claims.append((split, c["id"], c["claim"], articles.pop()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    searchers = {}
    with open(out, "w") as f:
        for n, (split, cid, query, article) in enumerate(claims):
            if article not in searchers:
                index = build_index(sentences[article], Path(args.work_dir) / article)
                searchers[article] = searcher_cls(str(index))
            hits = searchers[article].search(query, args.k)
            f.write(json.dumps({"split": split, "id": cid, "article": article, "query": query,
                                "candidates": [int(h.docid[len("sent_"):]) for h in hits],
                                "bm25": [h.score for h in hits]}) + "\n")
            if (n + 1) % 500 == 0:
                print(f"{n + 1}/{len(claims)}", flush=True)
    print(f"{len(claims)} claims, {len(searchers)} articles -> {out}")


if __name__ == "__main__":
    main()
