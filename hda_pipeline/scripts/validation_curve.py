#!/usr/bin/env python3
"""Extract the per-iteration validation curve from an HDA run log."""
import argparse, csv, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run", help="run directory or log.jsonl")
ap.add_argument("--out", default=None)
a = ap.parse_args()
log = Path(a.run)
if log.is_dir():
    log = log / "log.jsonl"
out = Path(a.out) if a.out else log.parent / "validation_curve.csv"
rows = []
for line in log.read_text().splitlines():
    if not line.strip():
        continue
    e = json.loads(line)
    if e.get("kind") == "iteration":
        rows.append({"iteration": e.get("it"), "val_accuracy": e.get("val_acc"),
                     "best_accuracy": e.get("best_acc"),
                     "verdicts": json.dumps(e.get("verdicts") or {}),
                     "statement": (e.get("statement") or "")[:200]})
if rows:
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} iterations -> {out}")
    for r in rows:
        print(f"iter {r['iteration']:2d}  val {r['val_accuracy']}  best {r['best_accuracy']}")
else:
    print("no iteration events yet")
