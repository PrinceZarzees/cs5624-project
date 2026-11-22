#!/usr/bin/env python3
"""Summarize a directory of HDA given-condition runs."""
import argparse, csv, json
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="experiments/hda/all_given")
ap.add_argument("--out", default="report/tables/hda/tab_hda_given_all.csv")
a = ap.parse_args()
rows = []
for d in sorted(Path(a.runs).iterdir()):
    p = d / "final.json"
    if not p.exists():
        continue
    x = json.loads(p.read_text())
    tests = x.get("test") or {}
    rows.append({
        "task": x.get("task", d.name), "best_it": x.get("best_it"),
        "val": x.get("best_val_acc"),
        "test_small": (tests.get("test_small") or {}).get("accuracy"),
        "test_medium": (tests.get("test_medium") or {}).get("accuracy"),
        "test_large": (tests.get("test_large") or {}).get("accuracy"),
        "test_xl": (tests.get("test_xl") or {}).get("accuracy"),
        "test_stress": (tests.get("test_stress") or {}).get("accuracy"),
        "prompt_tokens": (x.get("tokens") or {}).get("prompt"),
        "completion_tokens": (x.get("tokens") or {}).get("completion"),
    })
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["task"])
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} runs -> {a.out}")
for r in rows:
    print(r["task"], "val", r["val"], "small", r["test_small"], "large", r["test_large"])
