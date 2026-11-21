"""Accuracy stratified by graph characteristics.

This is the analysis for the review asking where optimized harnesses generalize:
density, topology/family, task, and scale.

    python -m graphharness.characteristics \
        --run "A=experiments/qwen3_4b_scaling_full/scaling/baseline_a/*/scores.jsonl" \
        --run "Bv2=experiments/qwen3_4b_scaling_full/scaling/baseline_b_v2/*/scores.jsonl" \
        --out experiments/analysis_by_characteristics.md
"""

from __future__ import annotations

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

from .data_loader import DEFAULT_DATA_DIR, load_jsonl, load_split, to_graph

SPLITS = ("test_small", "test_medium", "test_large", "test_stress")
DENSITY_BINS = [
    (0.0, 0.01, "very sparse (<.01)"),
    (0.01, 0.03, "sparse (.01-.03)"),
    (0.03, 0.10, "medium (.03-.10)"),
    (0.10, 0.30, "dense (.10-.30)"),
    (0.30, 1.01, "very dense (>.30)"),
]


def density_bin(density: float) -> str:
    for lo, hi, label in DENSITY_BINS:
        if lo <= density < hi:
            return label
    return DENSITY_BINS[-1][2]


def _records_by_id(data_dir: str | Path) -> dict[str, dict]:
    out = {}
    for split in SPLITS:
        try:
            for record in load_split(split, data_dir):
                out[record["id"]] = record
        except FileNotFoundError:
            continue
    return out


def _rate(items: list[dict], key: str = "correct") -> float:
    return sum(bool(x.get(key)) for x in items) / len(items) if items else 0.0


def _augment(score: dict, record: dict) -> dict:
    graph = to_graph(record)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
    degrees = [d for _, d in graph.degree()]
    return {
        **score,
        "density": density,
        "density_bin": density_bin(density),
        "family": record.get("meta", {}).get("family", "unknown"),
        "mean_degree": (sum(degrees) / n) if n else 0.0,
        "n_nodes": n,
    }


def _table(groups: dict[str, list[dict]], key_name: str) -> list[dict]:
    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        rows.append({
            key_name: key,
            "n": len(items),
            "mean_nodes": sum(x["n_nodes"] for x in items) / len(items),
            "accuracy": _rate(items),
            "feasible": _rate(items, "feasible"),
            "parsed": _rate(items, "parsed"),
        })
    return rows


def analyze(runs: list[tuple[str, list[str]]], data_dir: str | Path = DEFAULT_DATA_DIR) -> dict:
    records = _records_by_id(data_dir)
    payload = {}
    for label, paths in runs:
        scores = []
        for pattern in paths:
            for path in sorted(glob.glob(pattern) or [pattern]):
                for s in load_jsonl(path):
                    if s["id"] in records:
                        scores.append(_augment(s, records[s["id"]]))
        by_density: dict[str, list[dict]] = defaultdict(list)
        by_family: dict[str, list[dict]] = defaultdict(list)
        by_task: dict[str, list[dict]] = defaultdict(list)
        for s in scores:
            by_density[s["density_bin"]].append(s)
            by_family[s["family"]].append(s)
            by_task[s["task"]].append(s)
        payload[label] = {
            "overall": {"n": len(scores), "accuracy": _rate(scores)},
            "by_density": _table(by_density, "density_bin"),
            "by_family": _table(by_family, "family"),
            "by_task": _table(by_task, "task"),
        }
    return payload


def format_markdown(payload: dict) -> str:
    lines = ["# Accuracy by graph characteristics", ""]
    for label, data in payload.items():
        lines.append(f"## {label}  (n={data['overall']['n']}, "
                     f"accuracy={data['overall']['accuracy']:.1%})")
        for section, title, key in [
            ("by_density", "Density", "density_bin"),
            ("by_family", "Topology / family", "family"),
            ("by_task", "Task", "task"),
        ]:
            lines.append(f"\n### {title}\n")
            lines.append(f"| {key} | n | mean nodes | accuracy | feasible | parsed |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for row in data[section]:
                lines.append(
                    f"| {row[key]} | {row['n']} | {row['mean_nodes']:.0f} | "
                    f"{row['accuracy']:.1%} | {row['feasible']:.1%} | {row['parsed']:.1%} |"
                )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH",
                    help="label and comma-separated score path(s)/globs; repeatable")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out", default="experiments/analysis_by_characteristics.md")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)
    runs = []
    for value in args.run:
        label, sep, path = value.partition("=")
        if not sep:
            label, path = Path(value).parent.name, value
        runs.append((label, [p.strip() for p in path.split(",") if p.strip()]))
    payload = analyze(runs, args.data_dir)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(format_markdown(payload) + "\n", encoding="utf-8")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "section", "value", "n", "mean_nodes", "accuracy", "feasible", "parsed"])
            for label, data in payload.items():
                for section, key in [("by_density", "density_bin"), ("by_family", "family"), ("by_task", "task")]:
                    for row in data[section]:
                        w.writerow([label, section, row[key], row["n"], row["mean_nodes"],
                                    row["accuracy"], row["feasible"], row["parsed"]])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
