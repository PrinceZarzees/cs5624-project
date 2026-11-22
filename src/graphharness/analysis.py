"""Summaries and scale curves from per-problem score files."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .data_loader import load_jsonl

# Left edge, right edge, label.  The boundaries mirror the project's splits.
SIZE_BINS = [
    (0, 50, "4-50"),
    (51, 99, "51-99"),
    (100, 300, "100-300"),
    (301, 499, "301-499"),
    (500, 1000, "500-1000"),
    (1001, 10**9, "1000+"),
]


def size_bucket(n_nodes: int) -> str:
    for lo, hi, label in SIZE_BINS:
        if lo <= n_nodes <= hi:
            return label
    return "unknown"


def _rate(items: list[dict], key: str) -> float:
    return sum(bool(s.get(key)) for s in items) / len(items) if items else 0.0


def accuracy_by_size(scores: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        groups[size_bucket(int(s.get("n_nodes") or 0))].append(s)
    order = {label: i for i, (_, _, label) in enumerate(SIZE_BINS)}
    rows = []
    for label, items in sorted(groups.items(), key=lambda kv: order.get(kv[0], 99)):
        rows.append({
            "size_bin": label,
            "n": len(items),
            "mean_nodes": sum(s.get("n_nodes") or 0 for s in items) / len(items),
            "parsed": _rate(items, "parsed"),
            "feasible": _rate(items, "feasible"),
            "accuracy": _rate(items, "correct"),
        })
    return rows


def accuracy_by_task(scores: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        groups[str(s.get("task"))].append(s)
    return [
        {
            "task": task,
            "n": len(items),
            "mean_nodes": sum(s.get("n_nodes") or 0 for s in items) / len(items),
            "parsed": _rate(items, "parsed"),
            "feasible": _rate(items, "feasible"),
            "accuracy": _rate(items, "correct"),
        }
        for task, items in sorted(groups.items())
    ]


def failure_notes(scores: list[dict], limit: int = 20) -> list[dict]:
    failures = [s for s in scores if not s.get("correct")]
    return [
        {
            "id": s.get("id"),
            "task": s.get("task"),
            "n_nodes": s.get("n_nodes"),
            "parsed": s.get("parsed"),
            "feasible": s.get("feasible"),
            "note": s.get("note"),
        }
        for s in failures[:limit]
    ]


def format_size_table(rows: list[dict]) -> str:
    head = f"{'size bin':<10} {'n':>5} {'nodes':>8} {'parsed':>8} {'feasible':>9} {'accuracy':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['size_bin']:<10} {r['n']:>5} {r['mean_nodes']:>8.0f} "
            f"{r['parsed']:>8.1%} {r['feasible']:>9.1%} {r['accuracy']:>9.1%}"
        )
    return "\n".join(lines)


def format_task_table(rows: list[dict]) -> str:
    head = f"{'task':<22} {'n':>5} {'nodes':>8} {'parsed':>8} {'feasible':>9} {'accuracy':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{str(r['task']):<22} {r['n']:>5} {r['mean_nodes']:>8.0f} "
            f"{r['parsed']:>8.1%} {r['feasible']:>9.1%} {r['accuracy']:>9.1%}"
        )
    return "\n".join(lines)


def load_scores(paths: list[str | Path]) -> list[dict]:
    out = []
    for path in paths:
        out.extend(load_jsonl(path))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scores", nargs="+", help="per-problem score JSONL file(s)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of tables")
    args = ap.parse_args(argv)
    scores = load_scores(args.scores)
    payload = {
        "overall": {
            "n": len(scores),
            "parsed": _rate(scores, "parsed"),
            "feasible": _rate(scores, "feasible"),
            "accuracy": _rate(scores, "correct"),
        },
        "by_size": accuracy_by_size(scores),
        "by_task": accuracy_by_task(scores),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_size_table(payload["by_size"]))
        print()
        print(format_task_table(payload["by_task"]))


if __name__ == "__main__":
    main()
