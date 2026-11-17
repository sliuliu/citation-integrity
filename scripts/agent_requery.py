"Replay the agent's logged search queries through Pyserini BM25 (Porter stemming, stopwords) and compare gold coverage."
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from llm_oversimplify import MV, ROOT, read_jsonl
from llm_agent import BM25, SEARCH_K
from pyserini.pyclass import autoclass
Searcher = autoclass("io.anserini.search.SimpleSearcher")
mvc = {c["id"]: c for c in read_jsonl(MV / "claims-dev.jsonl")}
mvcorp = {d["doc_id"]: d["abstract"] for d in read_jsonl(MV / "corpus.jsonl")}
papers = {r["article"]: r["sentences"] for r in read_jsonl(ROOT / "data/agent/articles.jsonl")}
cites = {r["id"]: r["article"] for r in read_jsonl(ROOT / "data/agent/claims.jsonl") if r["split"] == "dev"}
rows = read_jsonl(ROOT / "runs/llm-threeclass-gptoss20b/dev-agent-none.jsonl")
searchers, ours, pys, both, nq, overlap = {}, [], [], [], 0, []
for r in rows:
    gold = {mvcorp[int(d)][s] for d, es in mvc[r["id"]]["evidence"].items() for e in es for s in e["sentences"]}
    if not gold:
        continue
    art = cites[r["id"]]; sents = papers[art]
    if art not in searchers:
        searchers[art] = Searcher(str(ROOT / ".work/bm25" / art / "index"))
    bm = BM25(sents)
    a, b = set(), set()
    for c in r["calls"][:8]:  # later calls got no results
        if c["tool"] != "search" or "query" not in c["args"]:
            continue
        q = str(c["args"]["query"]); nq += 1
        x = set(bm.top(q, SEARCH_K)); y = {int(h.docid[5:]) for h in searchers[art].search(q, SEARCH_K)}
        assert x == set(c["shown"]), (x, c["shown"])
        a |= x; b |= y; overlap.append(len(x & y) / SEARCH_K)
    cov = lambda s: len(gold & {sents[i] for i in s}) / len(gold)
    ours.append(cov(a)); pys.append(cov(b)); both.append((len(a), len(b)))
ours, pys = np.array(ours), np.array(pys)
print(f"{len(ours)} citations with gold evidence, {nq} search queries; same top-5 per query {np.mean(overlap):.2f}")
print(f"gold coverage from search calls only: our BM25 {ours.mean():.3f} | Pyserini {pys.mean():.3f}")
print(f"Pyserini better on {(pys > ours).sum()}, worse on {(pys < ours).sum()}; distinct sentences {np.mean([x for x, _ in both]):.1f} vs {np.mean([y for _, y in both]):.1f}")
