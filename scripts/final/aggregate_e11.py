#!/usr/bin/env python3
"""E11: summarize no-tools runs at 32k, split by prompt length.

Inputs: experiments/final/E11_notools_longctx/fit_rows.csv (written in the job)
and the per-run summary.json files.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/final/E11_notools_longctx"


def bucket(rows, lo, hi):
    xs = [r for r in rows if lo <= int(r["prompt_tokens"]) < hi]
    return len(xs), (sum(int(r["correct"]) for r in xs) / len(xs)) if xs else ""


def main():
    rows = list(csv.DictReader(open(OUT / "fit_rows.csv")))
    out = []
    for run in sorted({r["run"] for r in rows}):
        harness = Path(run).parts[-2]
        split = Path(run).parts[-1]
        xs = [r for r in rows if r["run"] == run]
        summ = json.loads((ROOT / run / "summary.json").read_text())
        o = summ["overall"]
        n8, a8 = bucket(xs, 0, 8192)
        n32, a32 = bucket(xs, 8192, 32768)
        ngt, agt = bucket(xs, 32768, 10**9)
        out.append({
            "harness": harness, "split": split, "n": len(xs),
            "accuracy": o["accuracy"], "parsed": o["parsed"], "feasible": o["feasible"],
            "tokens": (summ.get("tokens") or {}).get("total"),
            "max_prompt_tokens": max(int(r["prompt_tokens"]) for r in xs),
            "n_le8k": n8, "acc_le8k": a8,
            "n_8k_to_32k": n32, "acc_8k_to_32k": a32,
            "n_gt32k": ngt, "acc_gt32k": agt,
        })
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['harness']:22s} {r['split']:12s} n={r['n']:3d} acc={r['accuracy']:.3f} "
              f"<=8k n={r['n_le8k']:3d} acc={r['acc_le8k']} | 8-32k n={r['n_8k_to_32k']:3d} "
              f"| >32k n={r['n_gt32k']:3d} max={r['max_prompt_tokens']}")


if __name__ == "__main__":
    main()
