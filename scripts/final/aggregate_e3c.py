#!/usr/bin/env python3
"""E3c: val vs test_large accuracy for each accepted checkpoint."""
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/final/E3_probe"

def acc(p):
    d = json.loads(p.read_text()); return d["overall"]["accuracy"], d["overall"]["parsed"]

def main():
    rows = []
    for run in ("skillopt_probe", "rrsi_probe"):
        base = OUT / run
        if not base.exists(): continue
        accepted = {}
        hp = base / "history.jsonl"
        if hp.exists():
            for line in hp.read_text().splitlines():
                if line.strip():
                    r = json.loads(line); accepted[str(r.get("round"))] = r.get("accepted")
        for ck in ["start"] + sorted(p.stem for p in (base/"checkpoints").glob("round_*.json")):
            v = base / ck / "val/summary.json"; l = base / ck / "test_large/summary.json"
            if not (v.exists() and l.exists()): continue
            va, _ = acc(v); la, _ = acc(l)
            rnd = ck.replace("round_", "").lstrip("0") or "0"
            rows.append({"run": run, "checkpoint": ck, "round": rnd,
                         "accepted_at_round": accepted.get(rnd, ""),
                         "val_accuracy": va, "large_accuracy": la,
                         "delta_gen": va - la})
    if rows:
        with open(OUT/"summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        for r in rows:
            print(f"{r['run']:16s} {r['checkpoint']:9s} val={r['val_accuracy']:.3f} "
                  f"large={r['large_accuracy']:.3f} gap={r['delta_gen']:+.3f}")
    else:
        print("no E3c eval results yet")

if __name__ == "__main__":
    main()
