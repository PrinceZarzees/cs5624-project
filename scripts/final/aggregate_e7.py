#!/usr/bin/env python3
"""Aggregate E7 router/decomposition runs."""
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/final/E7_router"

def summ(path):
    p = Path(path) / "summary.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if "overall" in d:
        return d
    rows = d.get("rows", [])
    n = sum(r.get("n", 0) for r in rows)
    k = sum(r.get("n", 0) * r.get("accuracy", 0) for r in rows)
    return {"n": n, "overall": {"accuracy": k / n if n else 0},
            "by_task": rows, "mode": d.get("mode")}

def main():
    rows = []
    for mode, path in [("global_v2", OUT/"disc/global"), ("rule_router", OUT/"disc/rule"),
                       ("learned_router", OUT/"disc/learned"),
                       ("decomposed_rule", OUT/"disc/decomp_rule"),
                       ("decomposed_learned", OUT/"disc/decomp_learned")]:
        d = summ(path)
        if not d: continue
        if "overall" in d:
            o = d["overall"]
            rows.append({"set": "disconnected120", "mode": mode, "task": "ALL",
                         "n": d.get("n"), "accuracy": o["accuracy"],
                         "parsed": o.get("parsed"), "feasible": o.get("feasible")})
        for t in d.get("by_task", []):
            rows.append({"set": "disconnected120", "mode": mode, "task": t.get("task", "ALL"),
                         "n": t.get("n"), "accuracy": t.get("accuracy"),
                         "parsed": t.get("parsed"), "feasible": t.get("feasible")})
    for mode in ("rule", "learned"):
        d = summ(OUT/f"test_medium/{mode}")
        if not d: continue
        for t in d.get("by_task", []):
            rows.append({"set": "test_medium", "mode": f"{mode}_router", "task": t.get("task", "ALL"),
                         "n": t.get("n"), "accuracy": t.get("accuracy"),
                         "parsed": t.get("parsed"), "feasible": t.get("feasible")})
    if rows:
        with open(OUT/"summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {OUT/'summary.csv'} ({len(rows)} rows)")
    meta = OUT/"router.meta.json"
    if meta.exists():
        print(meta.read_text())

if __name__ == "__main__":
    main()
