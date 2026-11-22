"""Run a harness over a split and save predictions, scores and trajectories.

Example::

    python -m graphharness.runner \
        --harness baseline_b --split test_medium \
        --model Qwen/Qwen2.5-72B-Instruct --base-url http://localhost:8000/v1 \
        --out experiments/baseline_b/test_medium --workers 8

The runner is resumable: existing rows in ``predictions.jsonl`` are reused and
only missing problem ids are sent to the model.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .agent import AgentResult, Harness, run_agent
from .analysis import accuracy_by_size, accuracy_by_task, failure_notes, format_size_table
from .cli_common import add_model_arguments, build_model, harness_from_arg
from .data_loader import DEFAULT_DATA_DIR, load_jsonl, load_split, save_jsonl
from .evaluate_answer import format_table, score_response, summarize
from .models import Model


@dataclass
class ProblemResult:
    id: str
    task: str
    split: str | None
    n_nodes: int
    response: str | None
    score: dict
    agent: AgentResult | None = None
    error: str = ""
    record: dict | None = field(default=None, repr=False)


def evaluate_records(
    harness: Harness,
    model: Model,
    records: list[dict],
    max_steps: int | None = None,
    progress=None,
) -> list[ProblemResult]:
    """Run every record, one at a time. Callers handle parallelism."""
    out: list[ProblemResult] = []
    for i, record in enumerate(records, 1):
        result: ProblemResult
        try:
            agent = run_agent(harness, model, record, max_steps=max_steps)
            result = ProblemResult(
                id=record["id"],
                task=record["task"],
                split=record.get("split"),
                n_nodes=len(record["graph"]["nodes"]),
                response=agent.response or None,
                score=score_response(record, agent.response or None),
                agent=agent,
                record=record,
            )
        except Exception as exc:
            result = ProblemResult(
                id=record["id"],
                task=record["task"],
                split=record.get("split"),
                n_nodes=len(record["graph"]["nodes"]),
                response=None,
                score=score_response(record, None),
                error=f"{type(exc).__name__}: {exc}",
                record=record,
            )
        out.append(result)
        if progress:
            progress(i, len(records), result)
    return out


def _limit_records(records: list[dict], limit: int | None,
                   limit_per_task: int | None) -> list[dict]:
    if limit_per_task is not None:
        counts: dict[str, int] = {}
        kept = []
        for record in records:
            task = record["task"]
            if counts.get(task, 0) < limit_per_task:
                kept.append(record)
                counts[task] = counts.get(task, 0) + 1
        records = kept
    if limit is not None:
        records = records[:limit]
    return records


def _safe_name(problem_id: str) -> str:
    return problem_id.replace("/", "__").replace(" ", "_")


def _harness_hash(harness: Harness) -> str:
    blob = json.dumps(harness.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _summary_payload(
    harness: Harness,
    model: Model,
    split: str,
    results: list[ProblemResult],
    started_at: str,
    finished_at: str,
    duration_s: float,
) -> dict:
    scores = [r.score for r in results]
    agents = [r.agent for r in results if r.agent is not None]
    rows = summarize(scores)
    overall = next((r for r in rows if r["task"] == "ALL"), None)
    return {
        "harness": harness.name,
        "harness_config": harness.to_dict(),
        "model": getattr(model, "name", type(model).__name__),
        "split": split,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "n": len(scores),
        "overall": overall or {},
        "by_task": accuracy_by_task(scores),
        "by_size": accuracy_by_size(scores),
        "tokens": {
            "prompt": sum(a.prompt_tokens for a in agents),
            "completion": sum(a.completion_tokens for a in agents),
            "total": sum(a.total_tokens for a in agents),
            "mean_total_per_problem": (
                sum(a.total_tokens for a in agents) / len(agents) if agents else 0.0
            ),
        },
        "tool_calls": sum(a.tool_calls for a in agents),
        "mean_agent_duration_s": (
            sum(a.duration_s for a in agents) / len(agents) if agents else 0.0
        ),
        "errors": sum(1 for r in results if r.error),
        "failures": failure_notes(scores, limit=25),
        "rows": rows,
    }


def _write_summary(out_dir: Path, payload: dict) -> None:
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"harness: {payload['harness']}    model: {payload['model']}    split: {payload['split']}",
        f"problems: {payload['n']}    errors: {payload['errors']}    "
        f"duration: {payload['duration_s']:.1f}s    tokens: {payload['tokens']['total']}",
        "",
        format_table(payload["rows"]),
        "",
        "Accuracy by graph size:",
        format_size_table(payload["by_size"]),
    ]
    if payload["failures"]:
        lines.append("")
        lines.append("First failures:")
        for f in payload["failures"][:10]:
            lines.append(f"- {f['id']} ({f['task']}, n={f['n_nodes']}): {f['note']}")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(
    harness: Harness,
    model: Model,
    split: str = "test_small",
    out_dir: str | Path = "experiments/run",
    data_dir: str | Path | None = None,
    tasks: list[str] | None = None,
    ids: list[str] | None = None,
    limit: int | None = None,
    limit_per_task: int | None = None,
    workers: int = 1,
    max_steps: int | None = None,
    resume: bool = True,
    save_trajectories: bool = True,
    quiet: bool = False,
) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        gh_version = package_version("graphharness")
    except PackageNotFoundError:
        gh_version = "unknown"
    run_meta = {
        "harness": harness.name,
        "harness_hash": _harness_hash(harness),
        "harness_config": harness.to_dict(),
        "model": getattr(model, "name", type(model).__name__),
        "split": split,
        "tasks": tasks,
        "graphharness_version": gh_version,
    }
    meta_path = out_dir / "run_config.json"
    if resume and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("harness_hash", "model", "split"):
            if old.get(key) != run_meta[key]:
                raise SystemExit(
                    f"{out_dir} was produced with a different {key} "
                    f"({old.get(key)!r} != {run_meta[key]!r}); use a new --out "
                    "directory or pass --no-resume"
                )
    meta_path.write_text(json.dumps(run_meta, indent=2) + "\n", encoding="utf-8")
    records = load_split(split, data_dir or DEFAULT_DATA_DIR, tasks=tasks)
    if ids:
        wanted = set(ids)
        records = [r for r in records if r["id"] in wanted]
        missing = wanted - {r["id"] for r in records}
        if missing:
            raise SystemExit(f"problem id(s) not found in {split}: {sorted(missing)[:5]}")
    records = _limit_records(records, limit, limit_per_task)
    if not records:
        raise SystemExit(f"no records in split {split!r} (did you build the data?)")

    pred_path = out_dir / "predictions.jsonl"
    if not resume and pred_path.exists():
        pred_path.unlink()
    existing: dict[str, str | None] = {}
    if resume and pred_path.exists():
        for row in load_jsonl(pred_path):
            existing[row["id"]] = row.get("response")
        if existing and not quiet:
            print(f"resuming: {len(existing)} cached predictions in {pred_path}")

    todo = [r for r in records if r["id"] not in existing]
    results_by_id: dict[str, ProblemResult] = {}
    for record in records:
        if record["id"] in existing:
            response = existing[record["id"]]
            results_by_id[record["id"]] = ProblemResult(
                id=record["id"],
                task=record["task"],
                split=record.get("split"),
                n_nodes=len(record["graph"]["nodes"]),
                response=response,
                score=score_response(record, response),
                record=record,
            )

    traj_dir = out_dir / "trajectories"
    if save_trajectories:
        traj_dir.mkdir(parents=True, exist_ok=True)

    def handle(result: ProblemResult) -> None:
        results_by_id[result.id] = result
        if result.response is not None:
            with open(pred_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": result.id, "response": result.response},
                                   separators=(",", ":")) + "\n")
        if save_trajectories and result.agent is not None:
            (traj_dir / f"{_safe_name(result.id)}.json").write_text(
                json.dumps(result.agent.to_dict(), indent=2) + "\n", encoding="utf-8"
            )

    completed = len(results_by_id)
    total = len(records)

    def progress(i: int, n: int, result: ProblemResult) -> None:
        nonlocal completed
        completed += 1
        if quiet:
            return
        status = "ok" if result.score.get("correct") else "miss"
        print(f"[{completed:>4}/{total}] {status:<4} {result.id} "
              f"({result.score.get('note', '') or 'correct'})", flush=True)

    if todo:
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(evaluate_records, harness, model, [r], max_steps, None)
                    for r in todo
                ]
                for future in futures:
                    for result in future.result():
                        handle(result)
                        completed += 1
                        if not quiet:
                            status = "ok" if result.score.get("correct") else "miss"
                            print(f"[{completed:>4}/{total}] {status:<4} {result.id} "
                                  f"({result.score.get('note', '') or 'correct'})", flush=True)
        else:
            for result in evaluate_records(harness, model, todo, max_steps=max_steps,
                                           progress=progress):
                handle(result)
    elif not quiet:
        print("all predictions were already cached")

    results = [results_by_id[r["id"]] for r in records]
    scores = [r.score for r in results]
    save_jsonl(scores, out_dir / "scores.jsonl")
    finished_at = datetime.now(timezone.utc).isoformat()
    payload = _summary_payload(
        harness, model, split, results, started_at, finished_at,
        duration_s=time.perf_counter() - started,
    )
    _write_summary(out_dir, payload)
    if not quiet:
        print()
        print((out_dir / "summary.txt").read_text())
        print(f"wrote {out_dir / 'predictions.jsonl'}")
        print(f"wrote {out_dir / 'scores.jsonl'}")
        print(f"wrote {out_dir / 'summary.json'}")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harness", default="baseline_b",
                    help="baseline name or path to a harness JSON checkpoint")
    ap.add_argument("--split", default="test_small")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out", default=None,
                    help="output directory (default: experiments/<harness>/<split>)")
    ap.add_argument("--tasks", nargs="*", help="only these task names")
    ap.add_argument("--ids", nargs="*", help="only these problem ids")
    ap.add_argument("--limit", type=int, help="max problems total")
    ap.add_argument("--limit-per-task", type=int, help="max problems per task")
    ap.add_argument("--workers", type=int, default=1, help="parallel model calls")
    ap.add_argument("--max-steps", type=int, help="override harness max_steps")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-trajectories", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--list-harnesses", action="store_true")
    add_model_arguments(ap)
    args = ap.parse_args(argv)

    if args.list_harnesses:
        from .agent import HARNESSES
        for harness in HARNESSES.values():
            print(f"{harness.name:<12} {len(harness.rules)} rules  {harness.notes}")
        return

    harness = harness_from_arg(args.harness)
    model = build_model(args)
    out = args.out or str(Path("experiments") / harness.name / args.split)
    try:
        run_experiment(
            harness, model, split=args.split, out_dir=out, data_dir=args.data_dir,
            tasks=args.tasks, ids=args.ids, limit=args.limit,
            limit_per_task=args.limit_per_task,
            workers=args.workers, max_steps=args.max_steps,
            resume=not args.no_resume,
            save_trajectories=not args.no_trajectories,
            quiet=args.quiet,
        )
    except KeyboardInterrupt:
        print("\ninterrupted; cached predictions can be resumed with the same command",
              file=sys.stderr)
        raise SystemExit(130) from None
    except Exception:
        traceback.print_exc()
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
