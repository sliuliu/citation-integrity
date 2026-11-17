"""MultiVerS label classifier rebuilt from checkpoint state dicts.

Parameter names match upstream model.MultiVerSModel, so its checkpoints load
directly. Only the label path is run (pooler output -> label_classifier); the
paper trains with rationale loss weight 0, so the rationale head never
affects labels.
"""
import torch
from torch import nn
from transformers import LongformerConfig, LongformerModel

import ci_data  # noqa: F401  (puts the upstream multivers dir on sys.path)
from allennlp_feedforward import FeedForward

IGNORED_KEYS = {"encoder.embeddings.position_ids"}


class MultiVerSClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels=3, gradient_checkpointing=False):
        super().__init__()
        config = LongformerConfig.from_pretrained("allenai/longformer-large-4096")
        self.encoder = LongformerModel(config)
        self.encoder.resize_token_embeddings(vocab_size)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hidden = config.hidden_size
        acts = [nn.GELU(), nn.Identity()]
        drops = [config.hidden_dropout_prob, 0]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.label_classifier = FeedForward(hidden, 2, [hidden, num_labels], acts, drops)
        self.rationale_classifier = FeedForward(hidden * 2, 2, [hidden, 1], acts, drops)

    def forward(self, tokenized):
        pooled = self.dropout(self.encoder(**tokenized).pooler_output)
        return self.label_classifier(pooled)


def read_state_dict(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt.get("state_dict", ckpt)


def load_model(path, vocab_size=None, **kwargs):
    """Load a checkpoint. If vocab_size exceeds the checkpoint's, extra embedding
    rows are initialised as in upstream train.py (torch.randn)."""
    state = read_state_dict(path)
    ckpt_vocab = state["encoder.embeddings.word_embeddings.weight"].shape[0]
    model = MultiVerSClassifier(ckpt_vocab, **kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and set(unexpected) <= IGNORED_KEYS, (missing, unexpected)
    if vocab_size is not None and vocab_size != ckpt_vocab:
        assert vocab_size > ckpt_vocab, (vocab_size, ckpt_vocab)
        model.encoder.resize_token_embeddings(vocab_size)
        emb = model.encoder.embeddings.word_embeddings.weight
        with torch.no_grad():
            emb[ckpt_vocab:] = torch.randn(vocab_size - ckpt_vocab, emb.shape[1])
    return model, ckpt_vocab
