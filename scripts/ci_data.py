"""Shared data loading for Citation Integrity MultiVerS runs.

Each claim cites exactly one document. Labels follow the authors' reader
(data_train.SciFactReader): the label of the claim's evidence entry, or NEI
(= IRRELEVANT) when the claim has no evidence entry. Tokenization reuses the
authors' SciFactDataset._tokenize_longformer, so inputs are byte-identical to
theirs. Unlike their _tensorize, documents with zero sentences are allowed;
they occur in the gold-evidence oracle for citations without annotated evidence.
"""
import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/Citation-Integrity/CitationAccuracyClassificationCode/multivers"
sys.path.insert(0, str(UPSTREAM))

import data_train  # noqa: E402

# Index order of the label head (data_train.SciFactDataset.label_lookup).
LABELS = ["NOT_ACCURATE", "IRRELEVANT", "ACCURATE"]
LABEL_ID = {"NOT_ACCURATE": 0, "NOT ENOUGH INFO": 1, "IRRELEVANT": 1, "ACCURATE": 2}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_tokenizer():
    return data_train.get_tokenizer(argparse.Namespace(encoder_name="longformer-large-science"))


def load_examples(data_dir, split):
    data_dir = Path(data_dir)
    corpus = {d["doc_id"]: d for d in read_jsonl(data_dir / "corpus.jsonl")}
    examples = []
    for claim in read_jsonl(data_dir / f"claims_{split}.jsonl"):
        assert len(claim["cited_doc_ids"]) == 1, claim["id"]
        doc = corpus[claim["cited_doc_ids"][0]]
        labels = {e["label"] for es in claim["evidence"].values() for e in es}
        assert len(labels) <= 1, claim["id"]
        examples.append({
            "id": claim["id"],
            "claim": claim["claim"],
            "sentences": doc["abstract"],
            "title": doc["title"],
            "label": LABEL_ID[labels.pop() if labels else "NOT ENOUGH INFO"],
        })
    assert len({e["id"] for e in examples}) == len(examples)
    return examples


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer
        self._upstream = data_train.SciFactDataset([], tokenizer, "SciFact", 1.0)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        if ex["sentences"]:
            with contextlib.redirect_stdout(io.StringIO()):
                tokenized, _, _ = self._upstream._tokenize_longformer(ex["claim"], ex["sentences"], ex["title"])
        else:
            # Same text the upstream code builds ("claim</s>title</s>"); its
            # sentence-count assertion cannot hold for an empty document.
            eos = self.tokenizer.eos_token
            tokenized = self.tokenizer(ex["claim"] + eos + (ex["title"] + eos if ex["title"] is not None else ""))
            tokenized["global_attention_mask"] = self._upstream._get_global_attention_mask(tokenized)
        assert tokenized is not None, f"claim {ex['id']} exceeds 4096 tokens"
        return {"id": ex["id"], "label": ex["label"], "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "global_attention_mask": tokenized["global_attention_mask"]}


class Collator:
    def __init__(self, tokenizer):
        self.pad = {"input_ids": tokenizer.pad_token_id, "attention_mask": 0, "global_attention_mask": 0}

    def __call__(self, batch):
        width = max(len(b["input_ids"]) for b in batch)
        tokenized = {k: torch.tensor([b[k] + [v] * (width - len(b[k])) for b in batch])
                     for k, v in self.pad.items()}
        return {"id": torch.tensor([b["id"] for b in batch]),
                "label": torch.tensor([b["label"] for b in batch]),
                "tokenized": tokenized}


def evaluate(gold, pred):
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=LABELS, zero_division=0)
    return {
        "n": len(gold),
        "correct": sum(g == q for g, q in zip(gold, pred)),
        "micro_f1": f1_score(gold, pred, labels=LABELS, average="micro"),
        "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro"),
        "per_class": {
            lab: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
            for i, lab in enumerate(LABELS)
        },
        "confusion_matrix": {
            "labels": LABELS,
            "rows_gold_cols_pred": confusion_matrix(gold, pred, labels=LABELS).tolist(),
        },
    }
