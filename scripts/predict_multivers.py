"""Run a Citation Integrity MultiVerS checkpoint on a SciFact-format split.

Inputs are built with the authors' tokenization (see ci_data). The authors'
predict.py needs an unreleased ../train_configs.json and the Longformer
science checkpoint, neither of which is required for inference.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import ci_data
import ci_model


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def predict(model, loader, device, fp16=False):
    model.eval()
    rows = []
    for batch in loader:
        tok = {k: v.to(device) for k, v in batch["tokenized"].items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
            logits = model(tok)
        probs = torch.softmax(logits.float(), dim=1).cpu()
        for cid, lab, pr in zip(batch["id"].tolist(), batch["label"].tolist(), probs):
            rows.append({"id": cid, "gold": ci_data.LABELS[lab], "pred": ci_data.LABELS[int(pr.argmax())],
                         "probs": dict(zip(ci_data.LABELS, map(float, pr)))})
    return rows


def write_predictions(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True, help="dir with claims_{split}.jsonl and corpus.jsonl")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        sys.exit(f"refusing to overwrite existing {out}")
    out.mkdir(parents=True)
    torch.set_num_threads(2)

    tokenizer = ci_data.get_tokenizer()
    examples = ci_data.load_examples(args.data_dir, args.split)
    loader = DataLoader(ci_data.TokenizedDataset(examples, tokenizer), batch_size=args.batch_size,
                        collate_fn=ci_data.Collator(tokenizer))

    model, vocab = ci_model.load_model(args.checkpoint)
    assert vocab == len(tokenizer), (vocab, len(tokenizer))
    if args.precision == "fp16":
        model.half()
    model.to(args.device)

    t0 = time.time()
    rows = predict(model, loader, args.device)
    elapsed = time.time() - t0

    write_predictions(rows, out / "predictions.jsonl")
    metrics = ci_data.evaluate([r["gold"] for r in rows], [r["pred"] for r in rows])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run = {
        "argv": sys.argv,
        "checkpoint_sha256": sha256(args.checkpoint),
        "claims_sha256": sha256(Path(args.data_dir) / f"claims_{args.split}.jsonl"),
        "corpus_sha256": sha256(Path(args.data_dir) / "corpus.jsonl"),
        "n_predicted": len(rows),
        "seconds": round(elapsed, 1),
        "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if args.device == "cuda" else None,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
    }
    (out / "run.json").write_text(json.dumps(run, indent=2))
    print(json.dumps({k: metrics[k] for k in ("n", "correct", "micro_f1", "macro_f1")}),
          {k: round(v["f1"], 4) for k, v in metrics["per_class"].items()}, f"{elapsed:.0f}s")


if __name__ == "__main__":
    main()
