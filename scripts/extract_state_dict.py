"""Extract the model state_dict from a PyTorch Lightning checkpoint.

Reads only the storages the state_dict uses (optimizer state is skipped), so
a 5 GB training checkpoint fits in little RAM, and tolerates pickled objects
from Lightning versions that are not installed.
"""
import argparse
import pickle
import zipfile

import torch


class Lazy:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, size, stride


class Missing:
    "Stand-in for classes/functions of other library versions."

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __setstate__(self, state):
        pass


def rebuild_tensor(storage, offset, size, stride, *args):
    return Lazy(storage, offset, size, stride)


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return rebuild_tensor
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError, ImportError):
            return Missing

    def persistent_load(self, saved_id):
        _, storage_type, key, _, numel = saved_id
        dtype = storage_type.dtype if hasattr(storage_type, "dtype") else storage_type(0).dtype
        return (key, dtype, numel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    zf = zipfile.ZipFile(args.src)
    prefix = next(n for n in zf.namelist() if n.endswith("data.pkl"))[: -len("data.pkl")]
    with zf.open(prefix + "data.pkl") as f:
        ckpt = Unpickler(f).load()

    state = {}
    for name, lazy in ckpt["state_dict"].items():
        key, dtype, numel = lazy.storage
        raw = bytearray(zf.read(f"{prefix}data/{key}"))
        flat = torch.frombuffer(raw, dtype=dtype)
        assert flat.numel() == numel
        state[name] = torch.as_strided(flat, lazy.size, lazy.stride, lazy.offset).clone()
    torch.save({"state_dict": state, "source": args.src, "epoch": ckpt.get("epoch"),
                "global_step": ckpt.get("global_step")}, args.dst)
    emb = state["encoder.embeddings.word_embeddings.weight"]
    print(f"{len(state)} tensors, vocab {emb.shape[0]}, epoch {ckpt.get('epoch')}, step {ckpt.get('global_step')}")


if __name__ == "__main__":
    main()
