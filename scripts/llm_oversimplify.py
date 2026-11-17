"""Can an LLM tell OVERSIMPLIFY citations from ACCURATE ones?

Fine-tuned models stay near chance on this type (AUC 0.50-0.59), even with gold
evidence. Sample: every OVERSIMPLIFY test citation plus an equal seeded sample of
ACCURATE ones, each judged with five evidence sources (MonoT5 top-20, PubMedBERT top-20
and top-3, gold, gold hidden among 20 sentences). The
prompt is fixed in advance, holds no label information, and is not tuned on test.
Runs in .venv-llm (anthropic, scikit-learn) and reads its key from the environment
or a project .env file:

  python llm_oversimplify.py --dry-run     # sample, one prompt, cost estimate; no API calls
  python llm_oversimplify.py               # judge; completed requests are never repeated
  python llm_oversimplify.py --analyze     # AUC against the other models on the same citations
"""
import argparse
import concurrent.futures
import glob
import json
import os
import random
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "data/author_scifact_10"
MV = ROOT / "upstream/Citation-Integrity/Data/multivers-format"
OUT = ROOT / "runs/llm-oversimplify"
MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.0, 20.0  # dollars per million tokens

MARKERS = {"<|cit|>": "[THIS CITATION]", "<|multi_cit|>": "[THIS AND OTHER CITATIONS]", "<|other_cit|>": "[OTHER CITATION]"}
SYSTEM = """You check citations in biomedical papers. You are given a citing statement \
from one paper and sentences from the paper it cites. The marker [THIS CITATION] shows \
where the cited paper is referenced; [THIS AND OTHER CITATIONS] means several papers are \
cited together there; [OTHER CITATION] refers to a different paper.

Judge whether the statement represents the cited paper faithfully, or oversimplifies it. \
A statement oversimplifies when it conveys the source's topic and general direction but \
goes beyond what the source supports, for example by overgeneralizing a finding beyond its \
population, model system or conditions, by dropping important qualifications, or by stating \
a tentative or partial result as established. A faithful statement may paraphrase or \
summarize, as long as the source supports its strength and scope.

Give overstatement_score from 0 (fully faithful) to 100 (clearly oversimplified), the \
verdict that matches it, and a one-sentence explanation."""
SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "overstatement_score": {"type": "integer"},
        "verdict": {"type": "string", "enum": ["FAITHFUL", "OVERSIMPLIFIED"]},
    },
    "required": ["explanation", "overstatement_score", "verdict"],
    "additionalProperties": False,
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def citance(text):
    for marker, name in MARKERS.items():
        text = text.replace(marker, name)
    return re.sub(r"\s+", " ", text).strip()


def build_requests(seed):
    """The sampled citations, with no gold label in anything sent to the model."""
    author = read_jsonl(AUTHOR / "claims_test.jsonl")
    mv = {c["id"]: c for c in read_jsonl(MV / "claims-test.jsonl")}
    corpus = {d["doc_id"]: d["abstract"] for d in read_jsonl(MV / "corpus.jsonl")}
    top20 = {d["doc_id"]: d["abstract"] for d in read_jsonl(ROOT / "data/topk/top20/corpus.jsonl")}
    pubmedbert = {d["doc_id"]: d["abstract"] for d in read_jsonl(ROOT / "data/topk/pubmedbert20/corpus.jsonl")}

    def fine(c):
        return {e["label"] for es in mv[c["id"]]["evidence"].values() for e in es}

    def coarse(c):
        return {e["label"] for es in c["evidence"].values() for e in es}

    positive = [c for c in author if fine(c) == {"OVERSIMPLIFY"}]
    pool = sorted((c for c in author if coarse(c) == {"ACCURATE"}), key=lambda c: c["id"])
    negative = random.Random(seed).sample(pool, len(positive))

    requests = []
    for gold, claims in (("OVERSIMPLIFY", positive), ("ACCURATE", negative)):
        for c in claims:
            order = {d: i for i, d in enumerate(mv[c["id"]]["cited_doc_ids"])}
            keys = sorted((order[int(d)], int(d), s) for d, es in mv[c["id"]]["evidence"].items()
                          for e in es for s in e["sentences"])
            gold_sentences = list(dict.fromkeys(corpus[d][s] for _, d, s in keys))
            # All gold sentences hidden among the best other MonoT5 sentences, 20 in total and
            # shuffled, so neither the document length nor a sentence's position marks the gold.
            filler = [s for s in top20[c["cited_doc_ids"][0]] if s not in gold_sentences]
            goldfill = gold_sentences + filler[: max(20 - len(gold_sentences), 0)]
            random.Random(seed * 1_000_003 + c["id"]).shuffle(goldfill)
            evidence = {"top20": top20[c["cited_doc_ids"][0]], "gold": gold_sentences,
                        "pubmedbert20": pubmedbert[c["cited_doc_ids"][0]], "goldfill": goldfill,
                        "pubmedbert3": pubmedbert[c["cited_doc_ids"][0]][:3]}
            for source, sentences in evidence.items():
                numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
                requests.append({
                    "key": f"{source}-{c['id']}", "id": c["id"], "source": source, "gold": gold,
                    "prompt": f"Citing statement:\n{citance(c['claim'])}\n\nSentences from the cited paper:\n{numbered}",
                })
    return requests


def load_key():
    env = ROOT / ".env"
    if "ANTHROPIC_API_KEY" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")


def judge(client, req, effort):
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": req["prompt"]}],
    )
    row = {k: req[k] for k in ("key", "id", "source", "gold")}
    row.update(model=response.model, stop_reason=response.stop_reason,
               input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
    if response.stop_reason == "end_turn":
        row.update(json.loads(next(b.text for b in response.content if b.type == "text")))
    return row


def run(args):
    import anthropic

    load_key()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "results.jsonl"
    done = {r["key"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [r for r in build_requests(args.seed) if r["key"] not in done][: args.limit]
    print(f"{len(done)} already done, {len(todo)} to run", flush=True)
    client = anthropic.Anthropic()
    with open(path, "a") as f, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(judge, client, r, args.effort): r["key"] for r in todo}
        for n, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                row = future.result()
            except anthropic.APIStatusError as e:
                print(f"{futures[future]}: HTTP {e.status_code} {e.message}", flush=True)
                continue
            except anthropic.APIConnectionError as e:
                print(f"{futures[future]}: connection error {e}", flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    analyze()


def dry_run(args):
    requests = build_requests(args.seed)
    chars = [len(SYSTEM) + len(r["prompt"]) for r in requests]
    tokens_in = sum(chars) / 4
    print(f"{len(requests)} requests over {len({r['id'] for r in requests})} citations:",
          {g: sum(r["gold"] == g and r["source"] == "gold" for r in requests) for g in ("OVERSIMPLIFY", "ACCURATE")})
    print(f"about {tokens_in:,.0f} input tokens, median {statistics.median(chars) / 4:,.0f} per request")
    for out_each in (500, 1500, 4000):
        cost = (tokens_in * PRICE_IN + len(requests) * out_each * PRICE_OUT) / 1e6
        print(f"  if each answer with thinking uses {out_each} output tokens: about ${cost:.2f}")
    print("\n--- system ---\n" + SYSTEM + "\n\n--- first request (top-20) ---\n" + requests[0]["prompt"][:1200])


def model_auc(pattern, ids, gold):
    from sklearn.metrics import roc_auc_score

    out = []
    for d in sorted(glob.glob(str(ROOT / "runs" / pattern))):
        if not Path(d, "summary.json").exists():
            continue
        history = json.load(open(Path(d, "history.json")))
        epoch = history[-1]["epoch"] if "-author" in d else json.load(open(Path(d, "summary.json")))["best_epoch"]
        prob = {r["id"]: r["probs"]["NOT_ACCURATE"] for r in read_jsonl(Path(d, f"epoch{epoch}/predictions_test.jsonl"))}
        out.append(roc_auc_score(gold, [prob[i] for i in ids]))
    return out


def analyze(out=OUT):
    from sklearn.metrics import roc_auc_score

    rows = read_jsonl(Path(out) / "results.jsonl")
    for source in ("top20", "pubmedbert20", "pubmedbert3", "goldfill", "gold"):
        sub = [r for r in rows if r["source"] == source]
        ok = [r for r in sub if "overstatement_score" in r]
        if not ok:
            continue
        y = [r["gold"] == "OVERSIMPLIFY" for r in ok]
        pos = [r for r in ok if r["gold"] == "OVERSIMPLIFY"]
        neg = [r for r in ok if r["gold"] == "ACCURATE"]
        flagged = lambda rs: sum(r["verdict"] == "OVERSIMPLIFIED" for r in rs) / len(rs)
        print(f"{source}: {len(ok)}/{len(sub)} judged | AUC {roc_auc_score(y, [r['overstatement_score'] for r in ok]):.3f} "
              f"| flagged OVERSIMPLIFY {flagged(pos):.2f}, ACCURATE {flagged(neg):.2f}"
              f" | models {sorted({r['model'] for r in sub})}, stop {sorted({r['stop_reason'] for r in sub})}")
    ids = [r["id"] for r in rows if r["source"] == "top20" and "overstatement_score" in r]
    gold = [r["gold"] == "OVERSIMPLIFY" for r in rows if r["source"] == "top20" and "overstatement_score" in r]
    if ids:
        print("Other models, these citations (AUC per seed, final epoch):")
        for name, pattern in (("MultiVerS top-20", "topk20-author-*"), ("NLI fine-tuned top-20", "nli-top20-author-*"),
                              ("MultiVerS gold", "oracle-citance-A-author-*")):
            print(f"  {name:22s}", [round(a, 3) for a in model_auc(pattern, ids, gold)])
    used_in = sum(r["input_tokens"] for r in rows)
    used_out = sum(r["output_tokens"] for r in rows)
    print(f"tokens {used_in:,} in, {used_out:,} out", end="")
    print(f", about ${(used_in * PRICE_IN + used_out * PRICE_OUT) / 1e6:.2f}" if rows[0]["model"].startswith("claude") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N remaining requests")
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args)
    elif args.analyze:
        analyze()
    else:
        run(args)


if __name__ == "__main__":
    main()
