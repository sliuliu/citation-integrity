"""The OVERSIMPLIFY test of llm_oversimplify.py with an open-weight model on a local GPU.

Same citations, evidence, instructions and analysis; only the model changes. The
weights ship in MXFP4; they are expanded to bfloat16 at load, which avoids running
the Triton kernels from the Hub. The result is about 39 GiB, and expanding needs
temporary memory on top (one 44 GiB A40 ran out), so it is split over two GPUs.
Greedy decoding, so a rerun on the same software gives the same answers. Runs in
.venv-gptoss on CARC (carc/llm_gptoss.sbatch). Completed requests are never repeated.
"""
import argparse
import json
import re
import time
from pathlib import Path

from llm_oversimplify import ROOT, SYSTEM, analyze, build_requests, read_jsonl

JSON_INSTRUCTION = """

Reply with only a JSON object with exactly these keys: "explanation" (string), \
"overstatement_score" (integer from 0 to 100), "verdict" ("FAITHFUL" or "OVERSIMPLIFIED")."""
FINAL = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)", re.S)


def parse_answer(text):
    "The judgment from the final channel, or None if it is missing or malformed."
    final = FINAL.search(text)
    if not final:
        return None
    start, end = final.group(1).find("{"), final.group(1).rfind("}")
    try:
        answer = json.loads(final.group(1)[start:end + 1])
        score = int(answer["overstatement_score"])
    except (ValueError, KeyError, TypeError):
        return None
    if answer.get("verdict") not in ("FAITHFUL", "OVERSIMPLIFIED") or not 0 <= score <= 100:
        return None
    return {"explanation": str(answer.get("explanation", "")), "overstatement_score": score,
            "verdict": answer["verdict"]}


def judge(model, tok, req, name, effort, max_new_tokens):
    import torch

    messages = [{"role": "developer", "content": SYSTEM + JSON_INSTRUCTION}, {"role": "user", "content": req["prompt"]}]
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, reasoning_effort=effort,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    new = out[0, ids["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=False)
    answer = parse_answer(text)
    stop = "end_turn" if answer else ("max_tokens" if len(new) >= max_new_tokens else "parse_error")
    row = {k: req[k] for k in ("key", "id", "source", "gold")}
    row.update(model=name, stop_reason=stop, input_tokens=int(ids["input_ids"].shape[1]), output_tokens=len(new))
    row.update(answer or {"raw": text[-2000:]})
    return row


def load_model(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

    assert torch.cuda.device_count() >= 2, "expanding the MXFP4 weights needs two GPUs"
    tok = AutoTokenizer.from_pretrained(name)
    # The cap leaves each GPU room for the temporary tensors of the expansion and for generation.
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="auto",
                                                 max_memory={0: "30GiB", 1: "30GiB"},
                                                 quantization_config=Mxfp4Config(dequantize=True)).eval()
    print("loaded, GiB per GPU:", [round(torch.cuda.memory_allocated(i) / 2**30, 1) for i in (0, 1)], flush=True)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--out-dir", default=str(ROOT / "runs/llm-oversimplify-gptoss20b"))
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import torch

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.jsonl"
    done = {r["key"] for r in read_jsonl(path)} if path.exists() else set()
    todo = [r for r in build_requests(args.seed) if r["key"] not in done][: args.limit]
    print(f"{len(done)} already done, {len(todo)} to run", flush=True)

    model, tok = load_model(args.model)
    t0 = time.time()
    with open(path, "a") as f:
        for n, req in enumerate(todo, 1):
            row = judge(model, tok, req, args.model, args.effort, args.max_new_tokens)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 10 == 0 or n == 1:
                peak = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in (0, 1)]
                print(f"{n}/{len(todo)} {(time.time() - t0) / n:.1f} s each, last: {row['stop_reason']}, "
                      f"{row['output_tokens']} tokens, peak GiB per GPU {peak}", flush=True)
    analyze(out)


if __name__ == "__main__":
    main()
