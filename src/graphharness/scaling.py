"""Run one or more harnesses across all graph scales and compute scaling curves.

Example::

    python -m graphharness.scaling \
        --harnesses baseline_a baseline_b experiments/optimizer/best_harness.json \
        --splits test_small test_medium test_large test_stress \
        --model Qwen/Qwen2.5-72B-Instruct --base-url http://localhost:8000/v1 \
        --out experiments/scaling --limit-per-task 5 --workers 8

Each harness/split pair is written under ``<out>/<harness>/<split>/`` using the
normal resumable runner, so this command is safe to rerun after an interruption.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import format_size_table
from .cli_common import add_model_arguments, build_model, harness_from_arg
from .data_loader import DEFAULT_DATA_DIR
from .models import Model
from .runner import _harness_hash, run_experiment

DEFAULT_SPLITS = ["test_small", "test_medium", "test_large", "test_stress"]


def _run_rows(summary: dict) -> dict:
    overall = summary.get("overall") or {}
    return {
        "harness": summary["harness"],
        "split": summary["split"],
        "n": summary["n"],
        "accuracy": overall.get("accuracy", 0.0),
        "parsed": overall.get("parsed", 0.0),
        "feasible": overall.get("feasible", 0.0),
        "mean_nodes": overall.get("mean_nodes", 0.0),
        "tokens": summary.get("tokens", {}).get("total", 0),
        "mean_tokens": summary.get("tokens", {}).get("mean_total_per_problem", 0.0),
        "mean_duration_s": summary.get("mean_agent_duration_s", 0.0),
        "errors": summary.get("errors", 0),
        "by_size": summary.get("by_size", []),
    }


def generalize_gaps(runs: list[dict], small: str = "test_small",
                    large: str = "test_large") -> list[dict]:
    by_harness: dict[str, dict[str, dict]] = {}
    for row in runs:
        by_harness.setdefault(row["harness"], {})[row["split"]] = row
    gaps = []
    for harness, splits in sorted(by_harness.items()):
        if small in splits and large in splits:
            gaps.append({
                "harness": harness,
                "small_split": small,
                "large_split": large,
                "small_accuracy": splits[small]["accuracy"],
                "large_accuracy": splits[large]["accuracy"],
                "delta_gen": splits[small]["accuracy"] - splits[large]["accuracy"],
            })
    return gaps


def format_scaling_table(runs: list[dict]) -> str:
    head = (f"{'harness':<24} {'split':<13} {'n':>5} {'nodes':>7} "
            f"{'accuracy':>9} {'tokens':>10} {'sec/problem':>11}")
    lines = [head, "-" * len(head)]
    for row in runs:
        lines.append(
            f"{row['harness']:<24} {row['split']:<13} {row['n']:>5} "
            f"{row['mean_nodes']:>7.0f} {row['accuracy']:>9.1%} "
            f"{row['mean_tokens']:>10.0f} {row['mean_duration_s']:>11.2f}"
        )
    return "\n".join(lines)


def _dedupe_harnesses(harnesses) -> tuple[list, list[str]]:
    """Drop duplicate harness configs (the best checkpoint is often a baseline)."""
    kept = []
    skipped = []
    seen: dict[str, str] = {}
    for harness in harnesses:
        digest = _harness_hash(harness)
        if harness.name in seen:
            note = "same config" if seen[harness.name] == digest else "different config"
            skipped.append(f"{harness.name} ({note})")
            continue
        seen[harness.name] = digest
        kept.append(harness)
    return kept, skipped


def _write_summary(out_dir: Path, splits: list[str], harnesses, model_name: str,
                   runs: list[dict], quiet: bool = False) -> dict:
    payload = {
        "splits": splits,
        "harnesses": [h.to_dict() for h in harnesses],
        "model": model_name,
        "runs": runs,
        "generalization_gaps": generalize_gaps(runs),
    }
    (out_dir / "scaling_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [format_scaling_table(runs)]
    if payload["generalization_gaps"]:
        lines.append("")
        lines.append("Generalization gap (test_small - test_large):")
        for gap in payload["generalization_gaps"]:
            lines.append(
                f"- {gap['harness']}: {gap['small_accuracy']:.1%} - "
                f"{gap['large_accuracy']:.1%} = {gap['delta_gen']:+.1%}"
            )
    last = runs[-1] if runs else None
    if last and last["by_size"]:
        lines.append("")
        lines.append(f"Accuracy by size for {last['harness']} / {last['split']}:")
        lines.append(format_size_table(last["by_size"]))
    text = "\n".join(lines) + "\n"
    (out_dir / "scaling_summary.txt").write_text(text, encoding="utf-8")
    if not quiet:
        print("\n" + text)
    return payload


def summarize_existing(
    harness_specs: list[str],
    splits: list[str],
    out_dir: str | Path,
    quiet: bool = False,
) -> dict:
    """Rebuild the combined summary from existing per-run ``summary.json`` files."""
    out_dir = Path(out_dir)
    harnesses = [harness_from_arg(spec) for spec in harness_specs]
    harnesses, skipped = _dedupe_harnesses(harnesses)
    if skipped and not quiet:
        print(f"skipping duplicate harness config(s): {', '.join(skipped)}")
    runs = []
    model_name = "unknown"
    for harness in harnesses:
        for split in splits:
            path = out_dir / harness.name / split / "summary.json"
            if not path.exists():
                if not quiet:
                    print(f"missing summary, skipping: {path}")
                continue
            summary = json.loads(path.read_text(encoding="utf-8"))
            model_name = summary.get("model", model_name)
            runs.append(_run_rows(summary))
    return _write_summary(out_dir, splits, harnesses, model_name, runs, quiet)


def run_scaling(
    harness_specs: list[str],
    splits: list[str],
    model: Model,
    out_dir: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    tasks: list[str] | None = None,
    limit_per_task: int | None = None,
    limit: int | None = None,
    workers: int = 1,
    max_steps: int | None = None,
    resume: bool = True,
    quiet: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    harnesses = [harness_from_arg(spec) for spec in harness_specs]
    harnesses, skipped = _dedupe_harnesses(harnesses)
    if skipped and not quiet:
        print(f"skipping duplicate harness config(s): {', '.join(skipped)}")
    for harness in harnesses:
        for split in splits:
            run_dir = out_dir / harness.name / split
            if not quiet:
                print(f"\n=== {harness.name} / {split} ===", flush=True)
            summary = run_experiment(
                harness, model, split=split, out_dir=run_dir, data_dir=data_dir,
                tasks=tasks, limit=limit, limit_per_task=limit_per_task,
                workers=workers, max_steps=max_steps, resume=resume, quiet=quiet,
            )
            runs.append(_run_rows(summary))

    model_name = getattr(model, "name", type(model).__name__)
    return _write_summary(out_dir, splits, harnesses, model_name, runs, quiet)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harnesses", nargs="+", required=True,
                    help="built-in names or harness JSON checkpoints")
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--limit", type=int, help="max problems per harness/split")
    ap.add_argument("--limit-per-task", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--out", default="experiments/scaling")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--summarize-only", action="store_true",
                    help="rebuild scaling_summary.* from existing summary.json files")
    ap.add_argument("--quiet", action="store_true")
    add_model_arguments(ap)
    args = ap.parse_args(argv)
    if args.summarize_only:
        summarize_existing(args.harnesses, args.splits, args.out, quiet=args.quiet)
        return
    model = build_model(args)
    run_scaling(
        args.harnesses, args.splits, model, args.out,
        data_dir=args.data_dir, tasks=args.tasks, limit_per_task=args.limit_per_task,
        limit=args.limit, workers=args.workers, max_steps=args.max_steps,
        resume=not args.no_resume, quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
