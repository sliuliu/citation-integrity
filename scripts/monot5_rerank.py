"""Rerank BM25 candidates with MonoT5, reimplementing pygaggle's MonoT5.rerank.

pygaggle (commit 08339dd, the version the notebook's API matches):
  - T5BatchTokenizer: slow tokenizer, "Query: {query} Document: {document} Relevant:",
    padding longest, truncation, max_length 512, batches of 8 in BM25 order
  - greedy_decode one step from decoder_start_token_id, FP32 (use_amp False)
  - score = log_softmax over the ["▁false", "▁true"] logits, taken at "▁true"
  - rerank = sorted(rescore(...), key=score, reverse=True), a stable sort
The notebook keeps the first 50 reranked sentences; we keep all of them.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL = "castorini/monot5-base-med-msmarco"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bm25", required=True, help="output of bm25_candidates.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted run of the same --bm25 file")
    args = ap.parse_args()

    import bm25_candidates
    _, sentences = bm25_candidates.load_articles()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(args.device).eval()
    vocab = tokenizer.get_vocab()
    false_id, true_id = vocab["▁false"], vocab["▁true"]

    rows = [json.loads(line) for line in open(args.bm25)]
    if args.resume and Path(args.out).exists():
        # Drop a partially written last line, keep complete ones.
        done = [line for line in open(args.out) if line.endswith("\n")]
        Path(args.out).write_text("".join(done))
        rows = rows[len(done):]
        print(f"resuming after {len(done)} claims", flush=True)
    t0 = time.time()
    with open(args.out, "a" if args.resume else "w") as f, torch.no_grad():
        for n, row in enumerate(rows):
            sents = sentences[row["article"]]
            scores = []
            for i in range(0, len(row["candidates"]), args.batch_size):
                batch = row["candidates"][i:i + args.batch_size]
                enc = tokenizer.batch_encode_plus(
                    [f"Query: {row['query']} Document: {sents[j]} Relevant:" for j in batch],
                    padding="longest", truncation=True, max_length=512, return_tensors="pt").to(args.device)
                start = torch.full((len(batch), 1), model.config.decoder_start_token_id, device=args.device)
                logits = model(**enc, decoder_input_ids=start).logits[:, -1, [false_id, true_id]]
                scores += torch.log_softmax(logits, dim=1)[:, 1].tolist()
            order = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
            f.write(json.dumps({**{k: row[k] for k in ("split", "id", "article", "query")},
                                "ranked": [row["candidates"][k] for k in order],
                                "monot5": [scores[k] for k in order]}) + "\n")
            if (n + 1) % 200 == 0:
                print(f"{n + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    print(f"{len(rows)} claims -> {args.out}")


if __name__ == "__main__":
    main()
