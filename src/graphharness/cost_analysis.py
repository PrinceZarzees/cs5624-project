"""Optimization-cost accounting from optimizer history/summary files.

    python -m graphharness.cost_analysis \
        --run "from_A=experiments/qwen3_4b_opt_a/optimizer" \
        --run "from_Bv2=experiments/qwen3_4b/optimizer" \
        --out experiments/optimization_costs.md

Reports validation trajectory, cumulative tokens, wall time, and optional
per-million-token cost estimates. The token counts come from the real
optimizer runs; no model calls are made here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(path: str | Path) -> dict:
    path = Path(path)
    history = [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines() if line.strip()]
    summary = json.loads((path / "optimization_summary.json").read_text())
    return {"path": str(path), "history": history, "summary": summary}


def summarize(run: dict, input_cost_per_m: float, output_cost_per_m: float) -> dict:
    hist = run["history"]
    summary = run["summary"]
    # The optimizer logs one total token count per round; treat it as total.
    cumulative = 0
    rows = []
    for row in hist:
        cumulative += int(row.get("tokens") or 0)
        rows.append({
            "round": row.get("round"),
            "role": row.get("role", ""),
            "train_accuracy": row.get("train_accuracy"),
            "val_accuracy": row.get("val_accuracy"),
            "best_val_accuracy": row.get("best_val_accuracy"),
            "accepted": row.get("accepted"),
            "tokens": int(row.get("tokens") or 0),
            "cumulative_tokens": cumulative,
        })
    total = int(summary.get("total_tokens") or cumulative)
    # Conservative split: assume 80% prompt, 20% completion if no split is stored.
    prompt = 0.8 * total
    completion = 0.2 * total
    cost = (prompt / 1e6) * input_cost_per_m + (completion / 1e6) * output_cost_per_m
    return {
        "run": run["path"],
        "rounds": len(hist),
        "initial_val": summary.get("initial_val_accuracy"),
        "best_val": summary.get("best_val_accuracy"),
        "duration_s": summary.get("duration_s"),
        "total_tokens": total,
        "estimated_cost_usd": cost,
        "rows": rows,
    }


def format_markdown(payload: list[dict]) -> str:
    lines = ["# Optimization cost accounting", ""]
    lines.append("| run | rounds | initial val | best val | wall time | tokens | est. cost |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for p in payload:
        lines.append(
            f"| {p['run']} | {p['rounds']} | {p['initial_val']:.1%} | "
            f"{p['best_val']:.1%} | {p['duration_s']:.0f}s | "
            f"{p['total_tokens']:,} | ${p['estimated_cost_usd']:.2f} |"
        )
    for p in payload:
        lines.append("")
        lines.append(f"## {p['run']}")
        lines.append("| round | role | train | val | best val | accepted | round tokens | cumulative |")
        lines.append("|---:|---|---:|---:|---:|---|---:|---:|")
        for r in p["rows"]:
            def pct(x):
                return "" if x is None else f"{float(x):.1%}"
            lines.append(
                f"| {r['round']} | {r['role']} | {pct(r['train_accuracy'])} | "
                f"{pct(r['val_accuracy'])} | {pct(r['best_val_accuracy'])} | "
                f"{r['accepted']} | {r['tokens']:,} | {r['cumulative_tokens']:,} |"
            )
    lines.append("")
    lines.append("Cost estimate uses configurable per-million input/output rates; "
                 "the optimizer history stores total tokens, so the split is an assumption.")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--input-cost-per-m", type=float, default=0.50)
    ap.add_argument("--output-cost-per-m", type=float, default=1.50)
    ap.add_argument("--out", default="experiments/optimization_costs.md")
    args = ap.parse_args(argv)
    payload = []
    for value in args.run:
        label, sep, path = value.partition("=")
        if not sep:
            label, path = path, path
        p = summarize(load_run(path), args.input_cost_per_m, args.output_cost_per_m)
        p["run"] = label
        payload.append(p)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(format_markdown(payload) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
