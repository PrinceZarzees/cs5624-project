#!/usr/bin/env python3
"""Summarize HDA run directories (final.json) for the report."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "report/tables/hda"
OUT.mkdir(parents=True, exist_ok=True)


def esc(x):
    s = str(x)
    for ch, rep in [("%", "\\%"), ("&", "\\&"), ("_", "\\_")]:
        s = s.replace(ch, rep)
    return s


def tex(name, header, rows):
    lines = [f"\\begin{{tabular}}{{{'l' + 'r' * (len(header) - 1)}}}", "\\toprule",
             " & ".join(esc(h) for h in header) + r" \\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(esc(x) for x in r) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="tab_hda_pilot")
    a = ap.parse_args()
    rows = []
    for base in a.runs:
        for p in sorted(Path(base).rglob("final.json")):
            d = json.loads(p.read_text())
            cfg = d
            val = d.get("best_val_acc")
            val = 0.0 if val is None or val < 0 else val
            tests = d.get("test") or {}
            ts = tests.get("test_small") or {}
            acc = ts.get("accuracy", "")
            rows.append([cfg.get("task", p.parent.parent.name), cfg.get("condition", "given"),
                         cfg.get("k", ""), len(d.get("val_curve") or []), f"{val:.1%}",
                         f"{float(acc):.1%}" if acc != "" else "--",
                         d.get("statement_judge") or ""])
    if not rows:
        print("no final.json found")
        return
    header = ["task", "condition", "k", "iters", "best val", "test_small", "judge"]
    with open(OUT / f"{a.out}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    tex(a.out, header, rows)
    print(f"wrote {len(rows)} rows -> {OUT / a.out}.tex")


if __name__ == "__main__":
    main()
