#!/usr/bin/env python3
"""E5: score every pooled harness with a static rubric judge."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphharness.agent import Harness  # noqa: E402
from graphharness.cli_common import add_model_arguments, build_model  # noqa: E402
from graphharness.rubric_proxy import judge  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=str(ROOT / "experiments/final/E5_proxy/pool.csv"))
    ap.add_argument("--out", required=True)
    add_model_arguments(ap)
    args = ap.parse_args(argv)
    model = build_model(args)
    base = Path(args.pool).parent
    rows = list(csv.DictReader(open(args.pool)))
    out = []
    for r in rows:
        h = Harness.load(base / r["path"])
        rub = judge(model, h)
        scores = rub.get("scores") if isinstance(rub, dict) else None
        proxy = rub.get("overall") if isinstance(rub, dict) else None
        if proxy is None and scores:
            proxy = 20.0 * sum(float(v) for v in scores.values()) / len(scores)
        out.append({"id": r["id"], "proxy": proxy, "scores": scores,
                    "raw": rub if not scores else None})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    n_ok = sum(1 for x in out if x["proxy"] is not None)
    print(f"judged {len(out)} candidates, {n_ok} usable -> {args.out}")


if __name__ == "__main__":
    main()
