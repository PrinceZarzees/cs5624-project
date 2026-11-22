#!/usr/bin/env python3
"""Aggregate run summaries into a long CSV for tables/plots."""
import argparse, csv, glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graphharness.stats import aggregate_repeats

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="summary.json paths or globs")
    ap.add_argument("--out", required=True)
    args=ap.parse_args(argv)
    paths=[]
    for p in args.runs:
        paths += glob.glob(p) or [p]
    rows=[]
    for path in sorted(set(paths)):
        d=json.loads(Path(path).read_text())
        overall=d.get("overall") or {}
        rows.append({"path":path,"harness":d.get("harness"),"model":d.get("model"),
                     "split":d.get("split"),"n":d.get("n"),
                     "accuracy":overall.get("accuracy"),"parsed":overall.get("parsed"),
                     "feasible":overall.get("feasible"),"errors":d.get("errors"),
                     "tokens":(d.get("tokens") or {}).get("total"),
                     "mean_duration_s":d.get("mean_agent_duration_s")})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ["path"]); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}")

if __name__=="__main__": main()
