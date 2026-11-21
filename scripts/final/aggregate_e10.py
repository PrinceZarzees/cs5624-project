#!/usr/bin/env python3
"""Aggregate the E10 language x library cells."""
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/final/E10_lang_lib"
CELLS = {"py_lib": ("Python", "NetworkX"), "py_nolib": ("Python", "none"),
         "c_nolib": ("C", "none"), "c_lib": ("C", "graph_algos")}

def main():
    rows = []
    for cell, (lang, lib) in CELLS.items():
        for split in ("test_small", "test_large"):
            p = OUT/"cells"/cell/split/"summary.json"
            if not p.exists(): continue
            d = json.loads(p.read_text()); o = d["overall"]
            rows.append({"language": lang, "library": lib, "cell": cell, "split": split,
                         "n": d["n"], "accuracy": o["accuracy"], "parsed": o["parsed"],
                         "feasible": o["feasible"], "errors": d.get("errors"),
                         "tokens": (d.get("tokens") or {}).get("total")})
    if rows:
        with open(OUT/"summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {OUT/'summary.csv'} ({len(rows)} rows)")
        for r in rows: print(r)

if __name__ == "__main__":
    main()
