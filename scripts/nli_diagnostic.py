"""Can an off-the-shelf NLI model separate NOT_ACCURATE from ACCURATE citations?

Zero-shot diagnostic, no training. Each evidence sentence is the premise and the
citance (citation markers removed) the hypothesis; a citation's score aggregates
its sentences (max entailment, max contradiction, max entailment - max
contradiction). The aggregate is chosen on dev, then test ROC AUC for gold
NOT_ACCURATE vs ACCURATE is compared with MultiVerS's p(NOT_ACCURATE) on the
same citations. Sentence probabilities are cached in --cache.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ci_data import LABELS, ROOT, load_examples, read_jsonl

MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MARKER = re.compile(r"\[?\s*<\|(?:cit|multi_cit|other_cit)\|>\s*\]?")


def clean(claim):
    return re.sub(r"\s+", " ", MARKER.sub(" ", claim)).strip()


def sentence_probs(examples, tok, model, device, batch_size):
    "Per citation, a list of [entailment, neutral, contradiction] per evidence sentence."
    pairs = [(i, s, clean(ex["claim"])) for i, ex in enumerate(examples) for s in ex["sentences"]]
    out = [[] for _ in examples]
    with torch.no_grad():
        for b in range(0, len(pairs), batch_size):
            chunk = pairs[b:b + batch_size]
            try:
                enc = tok([s for _, s, _ in chunk], [h for _, _, h in chunk], truncation="only_first",
                          max_length=512, padding=True, return_tensors="pt").to(device)
            except Exception:
                # A citance alone over 512 tokens (one train claim) cannot keep its full text.
                enc = tok([s for _, s, _ in chunk], [h for _, _, h in chunk], truncation="longest_first",
                          max_length=512, padding=True, return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                probs = torch.softmax(model(**enc).logits.float(), dim=-1).tolist()
            for (i, _, _), p in zip(chunk, probs):
                out[i].append(p)
    return out


AGGREGATES = {
    "1 - max entailment": lambda P: 1 - P[:, 0].max(),
    "max contradiction": lambda P: P[:, 2].max(),
    "max contradiction - max entailment": lambda P: P[:, 2].max() - P[:, 0].max(),
}


def auc(gold, score, keep):
    idx = [i for i in range(len(gold)) if gold[i] in ("NOT_ACCURATE", "ACCURATE") and keep[i]]
    return roc_auc_score([gold[i] == "NOT_ACCURATE" for i in idx], [score[i] for i in idx]), len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["topk/top20", "topk/pubmedbert20", "oracle/citance-A"])
    ap.add_argument("--cache", default=str(ROOT / "runs/nli-diagnostic"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-only", nargs="*", default=[], help="also cache these splits (e.g. train), no analysis")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    tok = model = None
    for data in args.data:
        scores = {}
        for split in [*args.cache_only, "dev", "test"]:
            examples = load_examples(ROOT / "data" / data, split)
            path = cache / f"{data.replace('/', '_')}_{split}.jsonl"
            if not path.exists():
                if model is None:
                    tok = AutoTokenizer.from_pretrained(MODEL)
                    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(args.device).eval()
                probs = sentence_probs(examples, tok, model, args.device, args.batch_size)
                with open(path, "w") as f:
                    f.writelines(json.dumps({"id": ex["id"], "gold": LABELS[ex["label"]], "probs": p}) + "\n"
                                 for ex, p in zip(examples, probs))
            scores[split] = read_jsonl(path)
        keep = {s: [bool(r["probs"]) for r in scores[s]] for s in ("dev", "test")}
        gold = {s: [r["gold"] for r in scores[s]] for s in ("dev", "test")}
        dev_auc = {}
        for name, f in AGGREGATES.items():
            sc = [f(np.array(r["probs"])) if r["probs"] else 0.0 for r in scores["dev"]]
            dev_auc[name] = auc(gold["dev"], sc, keep["dev"])[0]
        best = max(dev_auc, key=dev_auc.get)
        test_sc = [AGGREGATES[best](np.array(r["probs"])) if r["probs"] else 0.0 for r in scores["test"]]
        a, n = auc(gold["test"], test_sc, keep["test"])
        print(f"{data}: dev AUC " + ", ".join(f"{k} {v:.3f}" for k, v in dev_auc.items()))
        print(f"{data}: chosen '{best}' -> test AUC {a:.3f} on {n} citations with evidence")
        (cache / f"{data.replace('/', '_')}_test_ids.json").write_text(
            json.dumps([r["id"] for r, k in zip(scores["test"], keep["test"]) if k]))


if __name__ == "__main__":
    main()
