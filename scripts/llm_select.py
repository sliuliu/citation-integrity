"""Select, then judge: gpt-oss-20b first picks at most K of the MonoT5 top-20 sentences, then judges with only those.

Motivation (runs/README.md): on OVERSIMPLIFY the LLM scored AUC 0.730 with the gold sentences alone but 0.590 with the
same gold sentences among 20, so distractors hurt; better retrieval and the evidence-seeking agent raised neither
coverage nor scores. Here the model removes distractors itself. Step 1 sees the citing text and the 20 sentences and
returns the numbers of the sentences needed to check each claim (possibly none). Step 2 is the improved prompt of
llm_threeclass.py, unchanged, with only the chosen sentences, keeping their numbers and MonoT5 order. Greedy decoding,
reasoning effort medium, as in the main runs.

  python llm_select.py --dry-run --k 5 --split dev
  python llm_select.py --k 5 --split dev        # on CARC, carc/llm_select.sbatch
Output: runs/llm-threeclass-gptoss20b/<split>-select<K>-topk_top20.jsonl, read by llm_threeclass.py --analyze.
"""
import argparse
import json
import time

from llm_oversimplify import ROOT, read_jsonl
from llm_threeclass import FINAL, IMPROVED_INSTRUCTIONS, analyze, build, judge, out_dir

MODEL = "openai/gpt-oss-20b"
EVIDENCE = "topk/top20"
SELECT_TOKENS = 4096


def select_instructions(k):
    return f"""You prepare evidence for checking a citation in a biomedical article. You receive the citing text and numbered sentences retrieved from the cited paper. The marker <|cit|> shows where the cited paper is referenced, <|multi_cit|> means it is cited together with other papers at that point, and <|other_cit|> refers to a different paper.

Pick the sentences that are needed to judge whether the citation faithfully reflects the paper: for each claim in the citing text, the sentences that support it, contradict it, or qualify it (for example by population, region, conditions or numbers). Leave out sentences that are only on the same topic. Choose as few as needed, at most {k}. If no sentence is about what the citing text says, choose none.

Reply with only a JSON object: {{"sentences": [the sentence numbers]}}."""


def top20(split):
    "Citation id -> its 20 retrieved sentences, in the order and numbering of llm_threeclass.build."
    docs = {d["doc_id"]: d["abstract"] for d in read_jsonl(ROOT / "data" / EVIDENCE / "corpus.jsonl")}
    return {c["id"]: docs[c["cited_doc_ids"][0]] for c in read_jsonl(ROOT / "data" / EVIDENCE / f"claims_{split}.jsonl")}


def parse_selection(text, n, k):
    "Chosen sentence numbers in their retrieval order, or None if the answer is missing or malformed."
    final = FINAL.search(text)
    if not final:
        return None
    body = final.group(1)
    try:
        chosen = json.loads(body[body.find("{"): body.rfind("}") + 1])["sentences"]
        chosen = sorted({int(i) for i in chosen})
    except (ValueError, KeyError, TypeError):
        return None
    if any(not 0 <= i < n for i in chosen):
        return None
    return chosen[:k]


def select(model, tok, req, sentences, k, effort):
    import torch

    case = req["messages"][-1]["content"]
    messages = [{"role": "developer", "content": select_instructions(k)}, {"role": "user", "content": case}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, reasoning_effort=effort,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=SELECT_TOKENS, do_sample=False)
    new = out[0, ids["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=False)
    return parse_selection(text, len(sentences), k), len(new), text


def reduced(req, sentences, chosen):
    "The improved-prompt request with only the chosen sentences; all 20 when the selection failed."
    if chosen is None:
        return req
    head = req["messages"][-1]["content"].split("\n\n<evidence>\n", 1)[0]
    evidence = " ".join(f"[{i}] {sentences[i]}" for i in chosen) if chosen else "(none)"
    messages = [{"role": "developer", "content": IMPROVED_INSTRUCTIONS},
                {"role": "user", "content": f"{head}\n\n<evidence>\n{evidence}\n</evidence>"}]
    return dict(req, messages=messages)


def run(args):
    from llm_threeclass import load

    out = out_dir(MODEL)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}-select{args.k}-{EVIDENCE.replace('/', '_')}.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [r for r in build("improved", EVIDENCE, args.split) if r["id"] not in done][: args.limit]
    retrieved = top20(args.split)
    print(f"{MODEL} {path.name}: {len(done)} already done, {len(todo)} to run", flush=True)
    model, tok = load(MODEL)
    t0 = time.time()
    with open(path, "a") as f:
        for n, req in enumerate(todo, 1):
            sentences = retrieved[req["id"]]
            chosen, select_tokens, select_text = select(model, tok, req, sentences, args.k, args.effort)
            row = judge(model, tok, reduced(req, sentences, chosen), "improved", MODEL, args.effort, 4096, 76)
            row.update(selected=chosen, select_tokens=select_tokens)
            if chosen is None:  # kept to see why the selection failed
                row["select_raw"] = select_text[-1500:]
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 20 == 0 or n <= 3:
                print(f"{n}/{len(todo)} {(time.time() - t0) / n:.1f} s each, last: {row['stop_reason']}, "
                      f"selected {chosen}", flush=True)
    analyze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="at most this many sentences")
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=None, help="only the next N citations (speed checks)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        requests, retrieved = build("improved", EVIDENCE, args.split), top20(args.split)
        assert all(reduced(r, retrieved[r["id"]], list(range(len(retrieved[r["id"]]))))["messages"] == r["messages"]
                   for r in requests), "rebuilding all 20 sentences must give the main run's request"
        print(select_instructions(args.k), "\n---\n", reduced(requests[0], retrieved[requests[0]["id"]], [0, 3])["messages"][-1]["content"])
        print(parse_selection('<|channel|>final<|message|>{"sentences": [7, 2, 2, 30]}<|return|>', 20, 5),
              parse_selection('<|channel|>final<|message|>{"sentences": [7, 2, 2, 3, 9, 11]}<|return|>', 20, 5),
              parse_selection('<|channel|>final<|message|>{"sentences": []}<|return|>', 20, 5))
    else:
        run(args)


if __name__ == "__main__":
    main()
