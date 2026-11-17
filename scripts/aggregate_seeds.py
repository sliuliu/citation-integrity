"""Aggregate runs over seeds (test scores of the reported epoch).

"author" runs report the final epoch (as the released model); "clean" runs
report the epoch chosen by dev macro-F1 (summary.json best_epoch).
"""
import argparse
import collections
import json
import re
import statistics
from pathlib import Path

from ci_data import ROOT

COLS = [("micro", lambda m: m["micro_f1"]), ("macro", lambda m: m["macro_f1"]),
        ("ACC", lambda m: m["per_class"]["ACCURATE"]["f1"]),
        ("N_ACC", lambda m: m["per_class"]["NOT_ACCURATE"]["f1"]),
        ("IRR", lambda m: m["per_class"]["IRRELEVANT"]["f1"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    ap.add_argument("--glob", default="oracle-*", help="run directory pattern")
    args = ap.parse_args()

    groups = collections.defaultdict(list)
    for run in sorted(Path(args.runs).glob(f"{args.glob}/summary.json")):
        name = re.fullmatch(r"(.+?)(?:-s(\d+))?-\d+", run.parent.name)
        exp, seed = name.group(1), int(name.group(2) or 76)
        history = json.loads((run.parent / "history.json").read_text())
        summary = json.loads(run.read_text())
        epoch = history[-1]["epoch"] if exp.endswith("-author") else summary["best_epoch"]
        test = history[epoch]["test"]
        n_acc_by_epoch = [round(h["test"]["per_class"]["NOT_ACCURATE"]["f1"], 3) for h in history[1:]]
        groups[exp].append((seed, epoch, [f(test) for _, f in COLS], n_acc_by_epoch))

    print("| Run | Seeds (epoch) | " + " | ".join(c for c, _ in COLS) + " |")
    print("|---|---|" + "---:|" * len(COLS))
    for exp, rows in groups.items():
        rows.sort(key=lambda r: (r[0] != 76, r[0]))
        cells = []
        for i in range(len(COLS)):
            vals = [r[2][i] for r in rows]
            cells.append(f"{statistics.mean(vals):.3f} ({min(vals):.3f}-{max(vals):.3f})")
        seeds = ", ".join(f"{s} (ep{e})" for s, e, _, _ in rows)
        print(f"| {exp} | {seeds} | " + " | ".join(cells) + " |")

    print("\nTest N_ACC F1 by epoch 1-5:")
    for exp, rows in groups.items():
        for seed, _, _, by_epoch in rows:
            print(f"  {exp:25s} seed {seed:2d}: {by_epoch}")


if __name__ == "__main__":
    main()
