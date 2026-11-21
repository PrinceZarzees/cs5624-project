#!/usr/bin/env python3
"""E6: aggregate the expanded real-benchmark runs.

Reads experiments/final/E6_real_bench/<split>/<harness>/summary.json and
scores, writes summary.csv and per_task.csv with Wilson CIs. Also reports the
accuracy on the subset of ids that existed in the original 20/task data, so the
old and new numbers remain comparable.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphharness.data_loader import load_jsonl  # noqa: E402
from graphharness.stats import wilson_ci  # noqa: E402

OUT = ROOT / "experiments/final/E6_real_bench"
SPLITS = {"ga_easy": "GraphArena easy", "ga_hard": "GraphArena hard", "graphqa_test": "GraphQA test"}
HARNESSES = ["baseline_a", "baseline_b", "baseline_b_v2"]
ORIG = {
    "ga_easy": ROOT / "data/grapharena_eval/ga_easy.jsonl",
    "ga_hard": ROOT / "data/grapharena_eval/ga_hard.jsonl",
    "graphqa_test": ROOT / "data/graphqa_eval/graphqa_test.jsonl",
}


def main():
    overall, per_task = [], []
    for split, label in SPLITS.items():
        orig_ids = {r["id"] for r in load_jsonl(ORIG[split])} if ORIG[split].exists() else set()
        for h in HARNESSES:
            d = OUT / split / h
            if not (d / "summary.json").exists():
                continue
            summ = json.loads((d / "summary.json").read_text())
            o = summ["overall"]
            lo, hi = wilson_ci(round(o["accuracy"] * summ["n"]), summ["n"])
            # subset comparable with the frozen 20/task run
            sub = []
            sp = d / "scores.jsonl"
            if sp.exists() and orig_ids:
                sub = [r for r in map(json.loads, open(sp)) if r and r["id"] in orig_ids]
            overall.append({
                "benchmark": label, "split": split, "harness": h, "n": summ["n"],
                "accuracy": o["accuracy"], "parsed": o["parsed"], "feasible": o["feasible"],
                "wilson_lo": lo, "wilson_hi": hi,
                "errors": summ.get("errors"),
                "tokens": (summ.get("tokens") or {}).get("total"),
                "n_original": len(sub),
                "accuracy_original_ids": (sum(bool(x.get("correct")) for x in sub) / len(sub)) if sub else "",
            })
            for t in summ.get("by_task", []):
                lo_t, hi_t = wilson_ci(round(t["accuracy"] * t["n"]), t["n"])
                per_task.append({
                    "benchmark": label, "split": split, "harness": h,
                    "task": t["task"], "n": t["n"], "accuracy": t["accuracy"],
                    "parsed": t["parsed"], "feasible": t["feasible"],
                    "wilson_lo": lo_t, "wilson_hi": hi_t,
                })
    if overall:
        with open(OUT / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(overall[0]))
            w.writeheader()
            w.writerows(overall)
    if per_task:
        with open(OUT / "per_task.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_task[0]))
            w.writeheader()
            w.writerows(per_task)
    print(f"wrote {len(overall)} overall rows and {len(per_task)} per-task rows")
    for r in overall:
        print(f"{r['benchmark']:16s} {r['harness']:14s} n={r['n']:4d} "
              f"acc={r['accuracy']:.3f} old-ids n={r['n_original']} "
              f"acc={r['accuracy_original_ids']}")


if __name__ == "__main__":
    main()
