"""Plot scaling curves from per-problem score files.

Example::

    python -m graphharness.plot_results \
        --run "Baseline A=experiments/scaling/baseline_a/*/scores.jsonl" \
        --run "Baseline B=experiments/scaling/baseline_b/*/scores.jsonl" \
        --out experiments/plots

Shell globs are expanded by the shell, so quote them or let the program expand
patterns itself (both are supported).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from .analysis import accuracy_by_size, accuracy_by_task, load_scores


def _expand(paths: list[str]) -> list[str]:
    out = []
    for path in paths:
        matches = sorted(glob.glob(path))
        out.extend(matches or [path])
    return out


def parse_runs(values: list[str]) -> list[tuple[str, list[str]]]:
    runs = []
    for value in values:
        label, sep, path = value.partition("=")
        if not sep:
            label, path = Path(value).parent.name, value
        paths = _expand([p.strip() for p in path.split(",") if p.strip()])
        runs.append((label.strip(), paths))
    return runs


def plot(runs: list[tuple[str, list[str]]], out_dir: str | Path,
         title: str = "Graph reasoning accuracy vs graph size") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    all_scores = {label: load_scores(paths) for label, paths in runs}

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for label, scores in all_scores.items():
        rows = accuracy_by_size(scores)
        if not rows:
            continue
        xs = [r["mean_nodes"] for r in rows]
        ys = [r["accuracy"] for r in rows]
        ax.plot(xs, ys, marker="o", label=label)
        for x, y, r in zip(xs, ys, rows):
            ax.annotate(f"n={r['n']}", (x, y), textcoords="offset points",
                        xytext=(4, -12), fontsize=7, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("Mean nodes per problem (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = out_dir / f"accuracy_vs_size.{ext}"
        fig.savefig(path, dpi=200)
        written.append(path)
    plt.close(fig)

    tasks = sorted({s.get("task") for scores in all_scores.values() for s in scores})
    if tasks:
        fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(tasks) * len(all_scores)), 4.2))
        width = 0.8 / max(1, len(all_scores))
        for i, (label, scores) in enumerate(all_scores.items()):
            rows = {r["task"]: r["accuracy"] for r in accuracy_by_task(scores)}
            xs = [j + i * width for j in range(len(tasks))]
            ax.bar(xs, [rows.get(t, 0.0) for t in tasks], width, label=label)
        ax.set_xticks([j + width * (len(all_scores) - 1) / 2 for j in range(len(tasks))])
        ax.set_xticklabels(tasks, rotation=20, ha="right")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend()
        ax.set_title("Accuracy by task")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = out_dir / f"accuracy_by_task.{ext}"
            fig.savefig(path, dpi=200)
            written.append(path)
        plt.close(fig)
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=SCORES",
                    help="label and comma-separated score JSONL path(s); repeatable")
    ap.add_argument("--out", default="experiments/plots")
    ap.add_argument("--title", default="Graph reasoning accuracy vs graph size")
    args = ap.parse_args(argv)
    runs = parse_runs(args.run)
    written = plot(runs, args.out, args.title)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
