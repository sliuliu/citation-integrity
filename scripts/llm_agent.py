"""Evidence-seeking agent for the three-way test, after MedExAgent (Gao et al. 2026, arXiv 2605.07058).

MedExAgent asks and examines before it diagnoses. Here gpt-oss-20b, zero-shot and untrained, searches the
cited paper with two tools before it judges the citation, instead of only receiving a fixed MonoT5 top-20:
  search(query)  the SEARCH_K sentences of the cited paper that best match the query (BM25 over all of them)
  read(index)    sentences index-2 .. index+2, to see the qualifications around a sentence
After at most MAX_CALLS calls it gives the same JSON answer as the improved prompt of llm_threeclass.py, so
the results go through the same analysis (--analyze there, llm_threeclass_compare.py). Two starts, chosen
between on dev before test is judged:
  none   only the citing text and the title of the cited paper
  top20  also the MonoT5 top-20 of the main runs, so the tools can only add evidence
Sentences are numbered by their position in the paper. Greedy decoding, reasoning effort medium.

  python llm_agent.py --prepare                       # locally, needs upstream annotations.zip
  python llm_agent.py --dry-run --start none --split dev
  python llm_agent.py --start none --split dev         # on CARC, carc/llm_agent.sbatch
Output: runs/llm-threeclass-gptoss20b/<split>-agent-<start>.jsonl; completed citations are never repeated.
"""
import argparse
import json
import math
import re
import time
from collections import Counter

from llm_oversimplify import ROOT, read_jsonl
from llm_threeclass import FINAL, IMPROVED_INSTRUCTIONS, MODELS, analyze, gold_label, out_dir, parse_answer

DATA = ROOT / "data/agent"
MODEL = "openai/gpt-oss-20b"
MAX_CALLS = 8
SEARCH_K = 5
TURN_TOKENS = 4096  # per model turn
TOTAL_TOKENS = 12288  # over all turns of one citation
# Each turn re-reads the whole conversation; gpt-oss runs eager attention, whose score matrix for a 5,000-token
# prefill does not fit beside the weights (an A40 ran out), so the prompt is prefilled in chunks.
PREFILL_CHUNK = 1024
TOKEN = re.compile(r"[a-z0-9]+")

TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": f"Search the cited paper. Returns the {SEARCH_K} sentences that best match the query, with their numbers.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "words to look for"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read",
        "description": "Read the cited paper around one sentence: sentences index-2 to index+2.",
        "parameters": {"type": "object", "properties": {"index": {"type": "integer", "description": "a sentence number"}},
                       "required": ["index"]}}},
]

INTRO = {
    "none": "You receive the citing text and the title of the cited paper.",
    "top20": "You receive the citing text, the title of the cited paper and excerpts retrieved from it, which may miss the relevant passage.",
}
TOOL_USE = (f"Before you judge, use the tools to find the passages of the cited paper that bear on each claim in the citing text: "
            f"search with your own queries, including specific terms, numbers and qualifications, and read around a sentence "
            f"to see its context. You have at most {MAX_CALLS} tool calls. Sentences are numbered by their position in the "
            f"paper, and the sentences you find count as excerpts.")


def instructions(start):
    "The improved prompt with its first paragraph rewritten for tool use."
    first, rest = IMPROVED_INSTRUCTIONS.split("\n\n", 1)
    markers = first[first.index("The marker"):]
    head = "You check whether a citation in a biomedical article faithfully reflects the paper it cites."
    return f"{head} {INTRO[start]} {TOOL_USE} {markers}\n\n{rest}"


def prepare():
    "Every sentence of each dev and test cited paper, and which paper each citation cites."
    from bm25_candidates import load_articles

    _, sentences = load_articles()
    rows = [r for r in read_jsonl(ROOT / "data/retrieval/bm25-citance.jsonl") if r["split"] in ("dev", "test")]
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "claims.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"split": r["split"], "id": r["id"], "article": r["article"]}) + "\n")
    with open(DATA / "articles.jsonl", "w") as f:
        for a in sorted({r["article"] for r in rows}):
            f.write(json.dumps({"article": a, "sentences": sentences[a]}) + "\n")
    print(f"{len(rows)} citations, {len({r['article'] for r in rows})} papers -> {DATA}")


class BM25:
    "Okapi BM25 over the sentences of one paper, with the Lucene defaults of the retrieval (k1 0.9, b 0.4)."

    def __init__(self, sentences, k1=0.9, b=0.4):
        self.docs = [Counter(TOKEN.findall(s.lower())) for s in sentences]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def top(self, query, k):
        terms = set(TOKEN.findall(query.lower()))
        scores = []
        for i, (d, n) in enumerate(zip(self.docs, self.lens)):
            s = sum(self.idf[t] * d[t] * (self.k1 + 1) / (d[t] + self.k1 * (1 - self.b + self.b * n / self.avg))
                    for t in terms if t in d)
            if s > 0:
                scores.append((s, i))
        return [i for _, i in sorted(scores, key=lambda x: (-x[0], x[1]))[:k]]


def build(start, split):
    "Requests in the file order of llm_threeclass.py. The gold label is kept aside, never in the messages."
    papers = {r["article"]: r["sentences"] for r in read_jsonl(DATA / "articles.jsonl")}
    cites = {r["id"]: r["article"] for r in read_jsonl(DATA / "claims.jsonl") if r["split"] == split}
    corpus = {d["doc_id"]: d["abstract"] for d in read_jsonl(ROOT / "data/topk/top20/corpus.jsonl")}
    requests = []
    for c in read_jsonl(ROOT / f"data/topk/top20/claims_{split}.jsonl"):
        sentences = papers[cites[c["id"]]]
        title = sentences[0].lstrip("# ").strip()
        case = f"# Test Case\n<title>\n{title}\n</title>\n\n<context>\n{c['claim']}\n</context>"
        if start == "top20":
            position = {s: i for i, s in enumerate(sentences)}
            excerpts = " ".join(f"[{position[s]}] {s}" for s in corpus[c["cited_doc_ids"][0]])
            case += f"\n\n<evidence>\n{excerpts}\n</evidence>"
        messages = [{"role": "developer", "content": instructions(start)}, {"role": "user", "content": case}]
        requests.append({"id": c["id"], "gold": gold_label(c), "article": cites[c["id"]], "messages": messages})
    return requests, papers


def tool_result(name, args, sentences, bm25):
    "The tool's reply to the model, and the sentence numbers it showed."
    try:
        if name == "search":
            shown = bm25.top(str(args["query"]), SEARCH_K)
        elif name == "read":
            i = int(args["index"])
            shown = list(range(max(0, i - 2), min(len(sentences), i + 3)))
        else:
            return f"Unknown tool {name}; use search or read.", []
    except (KeyError, TypeError, ValueError):
        return 'Invalid arguments; use {"query": "..."} for search and {"index": 12} for read.', []
    if not shown:
        return "No matching sentences.", []
    return "\n".join(f"[{i}] {sentences[i]}" for i in shown), shown


def tool_call(text):
    "(name, arguments) of a call that ends this turn, or None when the turn holds the final answer."
    if "<|call|>" not in text:
        return None
    head = text[: text.index("<|call|>")]
    names = re.findall(r"to=functions\.(\w+)", head)
    body = head.rsplit("<|message|>", 1)[-1]
    try:
        args = json.loads(body)
    except ValueError:
        args = {"raw": body}
    return (names[-1] if names else "unknown"), args if isinstance(args, dict) else {"raw": args}


def judge(model, tok, req, papers, effort):
    import torch

    sentences = papers[req["article"]]
    bm25 = BM25(sentences)
    messages = list(req["messages"])
    calls, generated, input_tokens, text = [], 0, 0, ""
    while generated < TOTAL_TOKENS:
        ids = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, reasoning_effort=effort,
                                      return_tensors="pt", return_dict=True).to(model.device)
        input_tokens = int(ids["input_ids"].shape[1])
        with torch.inference_mode():
            out = model.generate(**ids, max_new_tokens=min(TURN_TOKENS, TOTAL_TOKENS - generated), do_sample=False,
                                 prefill_chunk_size=PREFILL_CHUNK)
        new = out[0, input_tokens:]
        generated += len(new)
        text = tok.decode(new, skip_special_tokens=False)
        call = tool_call(text)
        if call is None:
            break
        name, args = call
        if len(calls) < MAX_CALLS:
            result, shown = tool_result(name, args, sentences, bm25)
        else:
            result, shown = "No tool calls left. Give your final answer now.", []
        calls.append({"tool": name, "args": args, "shown": shown})
        thinking = "".join(re.findall(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", text, re.S))
        messages += [{"role": "assistant", "thinking": thinking,
                      "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]},
                     {"role": "tool", "content": result}]
    final = FINAL.search(text)
    answer = parse_answer("improved", final.group(1) if final else None)
    stop = "end_turn" if answer else ("max_tokens" if generated >= TOTAL_TOKENS or len(new) >= TURN_TOKENS
                                      else "parse_error")
    row = {"id": req["id"], "gold": req["gold"], "stop_reason": stop, "input_tokens": input_tokens,
           "output_tokens": generated, "n_calls": len(calls), "calls": calls}
    row.update(answer or {"raw": text[-2000:]})
    return row


def run(args):
    from llm_threeclass import load

    out = out_dir(MODEL)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.split}-agent-{args.start}.jsonl"
    done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
    requests, papers = build(args.start, args.split)
    todo = [r for r in requests if r["id"] not in done][: args.limit]
    print(f"{MODEL} {path.name}: {len(done)} already done, {len(todo)} to run", flush=True)
    model, tok = load(MODEL)
    t0 = time.time()
    with open(path, "a") as f:
        for n, req in enumerate(todo, 1):
            row = judge(model, tok, req, papers, args.effort)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 20 == 0 or n <= 3:
                print(f"{n}/{len(todo)} {(time.time() - t0) / n:.1f} s each, last: {row['stop_reason']}, "
                      f"{row['n_calls']} calls, {row['output_tokens']} tokens", flush=True)
    analyze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", choices=list(INTRO))
    ap.add_argument("--split", choices=["dev", "test"])
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--limit", type=int, default=None, help="only the next N citations (speed checks)")
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assert MODELS[MODEL]["format"] == "harmony"
    if args.prepare:
        prepare()
    elif args.dry_run:
        requests, _ = build(args.start, args.split)
        print(f"{len(requests)} requests")
        print(requests[0]["messages"][0]["content"][:1200], "\n...\n")
        print(requests[0]["messages"][1]["content"][:1500])
    else:
        run(args)


if __name__ == "__main__":
    main()
