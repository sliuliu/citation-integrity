"""Fine-tune an NLI cross-encoder as the citation-accuracy classifier.

Same data, label weighting, protocols and outputs as train_multivers.py, so the
runs aggregate with aggregate_seeds.py. The input is the NLI pair
(premise = the claim's evidence sentences in rank order, hypothesis = the citance
without citation markers), truncated from the end of the premise to --max-length,
so the lowest-ranked sentences go first. The model's own NLI head is reused:
entailment -> ACCURATE, contradiction -> NOT_ACCURATE, neutral -> IRRELEVANT, so
epoch 0 is the zero-shot model under that mapping.
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import ci_data
from nli_diagnostic import clean
from predict_multivers import write_predictions
from train_multivers import label_weights

NLI_NAME = {"ACCURATE": "entail", "NOT_ACCURATE": "contradict", "IRRELEVANT": "neutral"}


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples, self.tok, self.max_length = examples, tokenizer, max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        premise, hypothesis = " ".join(ex["sentences"]), clean(ex["claim"])
        try:
            enc = self.tok(premise, hypothesis, truncation="only_first", max_length=self.max_length)
        except Exception:  # a citance alone longer than max_length (one train claim)
            enc = self.tok(premise, hypothesis, truncation="longest_first", max_length=self.max_length)
        return {"id": ex["id"], "label": ex["label"], **enc}


def collate(pad_id):
    def fn(batch):
        width = max(len(b["input_ids"]) for b in batch)
        out = {"id": torch.tensor([b["id"] for b in batch]), "label": torch.tensor([b["label"] for b in batch])}
        for k in batch[0]:
            if k in ("id", "label"):
                continue
            pad = pad_id if k == "input_ids" else 0
            out[k] = torch.tensor([b[k] + [pad] * (width - len(b[k])) for b in batch])
        return out
    return fn


def head_order(model):
    "Indices of the model's NLI logits in ci_data.LABELS order."
    names = {i: n.lower() for i, n in model.config.id2label.items()}
    return [next(i for i, n in names.items() if NLI_NAME[lab] in n) for lab in ci_data.LABELS]


def predict(model, loader, order, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k not in ("id", "label")}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits.float()[:, order]
            probs = torch.softmax(logits, dim=1).cpu()
            for cid, lab, pr in zip(batch["id"].tolist(), batch["label"].tolist(), probs):
                rows.append({"id": cid, "gold": ci_data.LABELS[lab], "pred": ci_data.LABELS[int(pr.argmax())],
                             "probs": dict(zip(ci_data.LABELS, map(float, pr)))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-splits", default="train")
    ap.add_argument("--eval-splits", default="dev,test")
    ap.add_argument("--select-on", default="dev")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16, help="effective batch size")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--label-weighting", default="author", choices=["author", "none"])
    ap.add_argument("--seed", type=int, default=76)
    ap.add_argument("--save", default="none", choices=["best", "none"])
    ap.add_argument("--limit", type=int, default=None, help="first N examples of each split (smoke tests)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    assert args.batch_size % args.micro_batch == 0

    out = Path(args.out_dir)
    if out.exists():
        sys.exit(f"refusing to overwrite existing {out}")
    out.mkdir(parents=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(args.device)
    order = head_order(model)
    order_t = torch.tensor(order, device=args.device)
    fn = collate(tok.pad_token_id)

    train_examples = []
    for split in args.train_splits.split(","):
        train_examples += ci_data.load_examples(args.data_dir, split)[: args.limit]
    weights = label_weights(train_examples, args.label_weighting)
    weight_tensor = torch.tensor([weights.get(i, 0.0) for i in range(len(ci_data.LABELS))], device=args.device)
    train_loader = DataLoader(PairDataset(train_examples, tok, args.max_length), batch_size=args.micro_batch,
                              shuffle=True, collate_fn=fn, generator=torch.Generator().manual_seed(args.seed))
    eval_loaders = {s: DataLoader(PairDataset(ci_data.load_examples(args.data_dir, s)[: args.limit], tok,
                                              args.max_length), batch_size=16, collate_fn=fn)
                    for s in args.eval_splits.split(",")}
    assert args.select_on in eval_loaders

    accumulate = args.batch_size // args.micro_batch
    steps_per_epoch = math.ceil(len(train_loader) / accumulate)
    total_steps = args.epochs * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup), total_steps)

    config = {"argv": sys.argv, "args": vars(args), "head_order": order, "n_train": len(train_examples),
              "train_label_counts": {ci_data.LABELS[k]: v for k, v in
                                     sorted(collections.Counter(e["label"] for e in train_examples).items())},
              "label_weights": {ci_data.LABELS[k]: v for k, v in sorted(weights.items())},
              "total_steps": total_steps, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(json.dumps({k: config[k] for k in ("n_train", "train_label_counts", "label_weights", "total_steps")}),
          flush=True)

    history, best = [], None

    def run_eval(epoch, train_loss):
        nonlocal best
        ep_dir = out / f"epoch{epoch}"
        ep_dir.mkdir()
        record = {"epoch": epoch, "train_loss": train_loss}
        for s, dl in eval_loaders.items():
            rows = predict(model, dl, order, args.device)
            write_predictions(rows, ep_dir / f"predictions_{s}.jsonl")
            record[s] = ci_data.evaluate([r["gold"] for r in rows], [r["pred"] for r in rows])
        history.append(record)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch} loss {train_loss} " + " ".join(
            f"{s}: micro {record[s]['micro_f1']:.4f} macro {record[s]['macro_f1']:.4f} "
            f"N_ACC {record[s]['per_class']['NOT_ACCURATE']['f1']:.4f}" for s in eval_loaders), flush=True)
        score = record[args.select_on]["macro_f1"]
        if best is None or score > best[1]:
            best = (epoch, score)
            if args.save == "best":
                model.save_pretrained(out / "best")
                tok.save_pretrained(out / "best")

    run_eval(0, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(train_loader):
            label = batch["label"].to(args.device)
            inputs = {k: v.to(args.device) for k, v in batch.items() if k not in ("id", "label")}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits.float().index_select(1, order_t)
            loss = (F.cross_entropy(logits, label, reduction="none") * weight_tensor[label]).sum()
            (loss / args.batch_size).backward()
            total, n = total + loss.item(), n + len(label)
            if (i + 1) % accumulate == 0 or i + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % 100 == 0:
                print(f"  epoch {epoch} batch {i + 1}/{len(train_loader)} loss {total / n:.4f} "
                      f"{(i + 1) / (time.time() - t0):.2f} it/s", flush=True)
        run_eval(epoch, total / n)

    summary = {"best_epoch_by": args.select_on, "best_epoch": best[0], "best_score": best[1],
               "peak_cuda_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
