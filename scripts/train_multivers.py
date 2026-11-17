"""Fine-tune MultiVerS for citation accuracy, replicating the authors' setup.

Mirrors upstream train_target.py + multivers/train.py (PyTorch Lightning 1.2)
without Lightning:
  - AdamW (transformers defaults: eps 1e-6, no weight decay), lr 1e-5,
    linear schedule with 10% warmup over scheduler_total_epochs (5)
  - batch size 1, gradient accumulation 8 (loss divided by 8, optimizer also
    steps on the last batch of an epoch), fp16 autocast, no gradient clipping
  - label loss only (rationale_weight 0), seed 76, shuffled training data
  - label weights: data_train.py hard-codes no_reweight_labels = False, so
    each class gets max_count / count (max_label_weight is never applied)

Protocol choices are explicit flags so that the authors' protocol (train on
train+dev, validate on test) and a clean one (train on train, select on dev)
use identical code. Predictions for every eval split are saved after every
epoch, so any selection rule can be applied afterwards without checkpoints.
"""
import argparse
import collections
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

import ci_data
import ci_model
from predict_multivers import predict, sha256, write_predictions


def label_weights(examples, mode):
    counts = collections.Counter(e["label"] for e in examples)
    if mode == "none":
        return {k: 1.0 for k in counts}
    top = max(counts.values())
    return {k: top / v for k, v in counts.items()}


def evaluate_split(model, loader, device, out, tag):
    rows = predict(model, loader, device, fp16=True)
    write_predictions(rows, out / f"predictions_{tag}.jsonl")
    return ci_data.evaluate([r["gold"] for r in rows], [r["pred"] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--init", required=True, help="starting checkpoint (healthver.ckpt or a Citation Integrity one)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-splits", default="train", help="comma-separated; authors used train,dev")
    ap.add_argument("--eval-splits", default="dev,test")
    ap.add_argument("--select-on", default="dev", help="eval split whose macro-F1 picks the saved best checkpoint")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--accumulate", type=int, default=8)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--label-weighting", default="author", choices=["author", "none"])
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--eval-batch-size", type=int, default=2)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--save", default="best", choices=["best", "all", "none"])
    ap.add_argument("--freeze-encoder", action="store_true", help="smoke tests on small GPUs only")
    ap.add_argument("--limit", type=int, default=None, help="use the first N examples of each split (smoke tests)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists():
        sys.exit(f"refusing to overwrite existing {out}")
    out.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = ci_data.get_tokenizer()
    collate = ci_data.Collator(tokenizer)
    train_examples = []
    for split in args.train_splits.split(","):
        train_examples += ci_data.load_examples(args.data_dir, split)[: args.limit]
    weights = label_weights(train_examples, args.label_weighting)
    eval_loaders = {
        split: DataLoader(ci_data.TokenizedDataset(ci_data.load_examples(args.data_dir, split)[: args.limit], tokenizer),
                          batch_size=args.eval_batch_size, collate_fn=collate)
        for split in args.eval_splits.split(",")
    }
    assert args.select_on in eval_loaders
    train_loader = DataLoader(ci_data.TokenizedDataset(train_examples, tokenizer), batch_size=1, shuffle=True,
                              collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))

    model, init_vocab = ci_model.load_model(args.init, vocab_size=len(tokenizer),
                                            gradient_checkpointing=args.gradient_checkpointing)
    if args.freeze_encoder:
        model.encoder.requires_grad_(False)
    model.to(args.device)

    steps_per_epoch = math.ceil(len(train_examples) / args.accumulate)
    total_steps = args.epochs * steps_per_epoch
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                  eps=1e-6, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps * args.warmup, total_steps)
    scaler = torch.cuda.amp.GradScaler()
    weight_tensor = torch.tensor([weights.get(i, 0.0) for i in range(len(ci_data.LABELS))], device=args.device)

    config = {
        "argv": sys.argv, "args": vars(args),
        "init_sha256": sha256(args.init), "init_vocab": init_vocab, "vocab": len(tokenizer),
        "n_train": len(train_examples),
        "train_label_counts": {ci_data.LABELS[k]: v for k, v in
                               sorted(collections.Counter(e["label"] for e in train_examples).items())},
        "label_weights": {ci_data.LABELS[k]: v for k, v in sorted(weights.items())},
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "data_sha256": {p.name: sha256(p) for p in sorted(Path(args.data_dir).glob("*.jsonl"))},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ("n_train", "train_label_counts", "label_weights", "total_steps")}),
          flush=True)

    history, best = [], None

    def run_eval(epoch, train_loss):
        nonlocal best
        ep_dir = out / f"epoch{epoch}"
        ep_dir.mkdir()
        record = {"epoch": epoch, "train_loss": train_loss,
                  **{s: evaluate_split(model, dl, args.device, ep_dir, s) for s, dl in eval_loaders.items()}}
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch} loss {train_loss} " + " ".join(
            f"{s}: micro {record[s]['micro_f1']:.4f} macro {record[s]['macro_f1']:.4f}" for s in eval_loaders),
            flush=True)
        score = record[args.select_on]["macro_f1"]
        if args.save == "all" or (args.save == "best" and (best is None or score > best[1])):
            name = f"epoch{epoch}.ckpt" if args.save == "all" else "best.ckpt"
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "selected_on": args.select_on,
                        "score": score}, out / name)
        if best is None or score > best[1]:
            best = (epoch, score)

    run_eval(0, None)  # the starting checkpoint, before any update
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            tok = {k: v.to(args.device) for k, v in batch["tokenized"].items()}
            label = batch["label"].to(args.device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(tok)
            loss = (F.cross_entropy(logits.float(), label, reduction="none") * weight_tensor[label]).sum()
            scaler.scale(loss / args.accumulate).backward()
            total, n = total + loss.item(), n + 1
            if (i + 1) % args.accumulate == 0 or i + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % 200 == 0:
                print(f"  epoch {epoch} step {i + 1}/{len(train_loader)} loss {total / n:.4f} "
                      f"{(i + 1) / (time.time() - t0):.2f} it/s", flush=True)
        run_eval(epoch, total / n)

    summary = {"best_epoch_by": args.select_on, "best_epoch": best[0], "best_score": best[1],
               "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
