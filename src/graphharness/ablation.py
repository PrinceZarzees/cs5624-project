"""Rule-by-rule ablation of an optimized harness.

For each rule in the harness, remove exactly that rule, rerun the same split,
and report the accuracy drop.  A large positive ``delta`` means the rule helped;
a negative delta means the rule hurt on this scale.

Example::

    python -m graphharness.ablation \
        --harness experiments/optimizer/best_harness.json \
        --split test_large --model ... --out experiments/ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cli_common import add_model_arguments, build_model, harness_from_arg
from .data_loader import DEFAULT_DATA_DIR
from .models import Model
from .runner import run_experiment


def _row(summary: dict, variant: str, rule: str = "", rule_index: int | None = None) -> dict:
    return {
        "variant": variant,
        "rule_index": rule_index,
        "rule": rule,
        "accuracy": (summary.get("overall") or {}).get("accuracy", 0.0),
        "parsed": (summary.get("overall") or {}).get("parsed", 0.0),
        "feasible": (summary.get("overall") or {}).get("feasible", 0.0),
        "tokens": (summary.get("tokens") or {}).get("total", 0),
        "mean_tokens": (summary.get("tokens") or {}).get("mean_total_per_problem", 0.0),
        "mean_duration_s": summary.get("mean_agent_duration_s", 0.0),
        "by_task": summary.get("by_task", []),
    }


def run_ablation(
    harness,
    model: Model,
    split: str,
    out_dir: str | Path,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    tasks: list[str] | None = None,
    limit: int | None = None,
    limit_per_task: int | None = None,
    workers: int = 1,
    max_steps: int | None = None,
    rule_indices: list[int] | None = None,
    resume: bool = True,
    quiet: bool = False,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = rule_indices if rule_indices is not None else list(range(len(harness.rules)))

    common = dict(
        split=split, data_dir=data_dir, tasks=tasks, limit=limit,
        limit_per_task=limit_per_task, workers=workers, max_steps=max_steps,
        resume=resume, quiet=quiet,
    )
    if not quiet:
        print("=== full harness ===", flush=True)
    full = run_experiment(harness, model, out_dir=out_dir / "full", **common)
    rows = [_row(full, "full")]
    for index in indices:
        variant = harness.without_rule(index)
        if not quiet:
            print(f"=== remove rule {index}: {harness.rules[index][:80]} ===", flush=True)
        summary = run_experiment(
            variant, model, out_dir=out_dir / f"no_rule_{index:02d}", **common
        )
        rows.append(_row(summary, variant.name, harness.rules[index], index))

    base = rows[0]["accuracy"]
    for row in rows[1:]:
        row["delta"] = base - row["accuracy"]

    payload = {
        "harness": harness.to_dict(),
        "model": getattr(model, "name", type(model).__name__),
        "split": split,
        "full_accuracy": base,
        "rows": rows,
    }
    (out_dir / "ablation_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        f"harness: {harness.name}    split: {split}    full accuracy: {base:.1%}",
        "",
        f"{'variant':<28} {'accuracy':>9} {'delta':>8}  rule",
        "-" * 100,
    ]
    for row in rows:
        delta = "" if "delta" not in row else f"{row['delta']:+.1%}"
        lines.append(
            f"{row['variant']:<28} {row['accuracy']:>9.1%} {delta:>8}  "
            f"{(row['rule'][:60] + '...') if len(row['rule']) > 60 else row['rule']}"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "ablation_summary.txt").write_text(text, encoding="utf-8")
    if not quiet:
        print(text)
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", required=True,
                    help="harness JSON checkpoint or built-in name")
    ap.add_argument("--split", default="test_large")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--limit-per-task", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--rules", nargs="*", type=int,
                    help="rule indices to ablate (default: all)")
    ap.add_argument("--out", default="experiments/ablation")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    add_model_arguments(ap)
    args = ap.parse_args(argv)

    harness = harness_from_arg(args.harness)
    model = build_model(args)
    run_ablation(
        harness, model, args.split, args.out, data_dir=args.data_dir,
        tasks=args.tasks, limit=args.limit, limit_per_task=args.limit_per_task,
        workers=args.workers, max_steps=args.max_steps, rule_indices=args.rules,
        resume=not args.no_resume, quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
