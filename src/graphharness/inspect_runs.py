"""Inspect a completed run: tool usage, failure reasons and trajectory excerpts.

    python -m graphharness.inspect_runs experiments/qwen3_4b_pilot/baseline_b
    python -m graphharness.inspect_runs <run_dir> --json --failures 0

The run directory must contain ``scores.jsonl``; trajectories are optional.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .data_loader import load_jsonl


def safe_name(problem_id: str) -> str:
    return problem_id.replace("/", "__").replace(" ", "_")


def load_trajectory(run_dir: Path, problem_id: str) -> dict | None:
    path = run_dir / "trajectories" / f"{safe_name(problem_id)}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def trajectory_stats(trajectory: dict | None) -> dict:
    if not trajectory:
        return {"steps": 0, "tool_calls": 0, "timeouts": 0, "errors": 0,
                "last_code": "", "last_observation": ""}
    steps = trajectory.get("steps") or []
    tool_steps = [s for s in steps if s.get("action") == "code"]
    last = tool_steps[-1] if tool_steps else {}
    observation = ""
    if last:
        if last.get("stdout"):
            observation += "stdout: " + str(last["stdout"])[:800]
        if last.get("stderr"):
            observation += ("\n" if observation else "") + "stderr: " + str(last["stderr"])[:800]
    return {
        "steps": len(steps),
        "tool_calls": int(trajectory.get("tool_calls") or len(tool_steps)),
        "timeouts": sum(bool(s.get("timed_out")) for s in tool_steps),
        "errors": sum(bool(s.get("ok") is False and not s.get("timed_out")) for s in tool_steps),
        "last_code": str(last.get("code") or "")[:1200],
        "last_observation": observation[:1600],
    }


def summarize_run(run_dir: str | Path, max_failures: int = 10) -> dict:
    run_dir = Path(run_dir)
    scores = load_jsonl(run_dir / "scores.jsonl")
    groups: dict[str, list[dict]] = defaultdict(list)
    for score in scores:
        groups[score["task"]].append(score)
    tasks = {}
    for task, items in sorted(groups.items()):
        stats = [trajectory_stats(load_trajectory(run_dir, s["id"])) for s in items]
        n = len(items)
        tasks[task] = {
            "n": n,
            "accuracy": sum(bool(s["correct"]) for s in items) / n,
            "feasible": sum(bool(s["feasible"]) for s in items) / n,
            "parsed": sum(bool(s["parsed"]) for s in items) / n,
            "mean_steps": sum(x["steps"] for x in stats) / n,
            "mean_tool_calls": sum(x["tool_calls"] for x in stats) / n,
            "timeouts": sum(x["timeouts"] for x in stats),
            "code_errors": sum(x["errors"] for x in stats),
            "failure_notes": dict(Counter(
                s.get("note") or "no note" for s in items if not s.get("correct")
            ).most_common(8)),
        }

    overall = {
        "n": len(scores),
        "accuracy": sum(bool(s["correct"]) for s in scores) / len(scores) if scores else 0.0,
        "feasible": sum(bool(s["feasible"]) for s in scores) / len(scores) if scores else 0.0,
        "parsed": sum(bool(s["parsed"]) for s in scores) / len(scores) if scores else 0.0,
        "tasks": tasks,
    }
    failures = []
    for score in scores:
        if score.get("correct"):
            continue
        traj = load_trajectory(run_dir, score["id"])
        failures.append({
            "id": score["id"],
            "task": score["task"],
            "n_nodes": score.get("n_nodes"),
            "note": score.get("note"),
            "answer_nodes": score.get("answer_nodes"),
            **trajectory_stats(traj),
        })
        if len(failures) >= max_failures:
            break
    if run_dir.joinpath("summary.json").exists():
        summary = json.loads((run_dir / "summary.json").read_text())
        run_info = {
            "harness": summary.get("harness"),
            "model": summary.get("model"),
            "split": summary.get("split"),
        }
    else:
        run_info = {}
    return {"run_dir": str(run_dir), **run_info, "overall": overall,
            "failures": failures}


def format_report(payload: dict) -> str:
    overall = payload["overall"]
    lines = [
        f"run: {payload['run_dir']}",
        f"harness: {payload.get('harness')}    model: {payload.get('model')}    "
        f"split: {payload.get('split')}",
        f"problems: {overall['n']}    accuracy: {overall['accuracy']:.1%}    "
        f"feasible: {overall['feasible']:.1%}    parsed: {overall['parsed']:.1%}",
        "",
        f"{'task':<22} {'n':>4} {'acc':>7} {'steps':>7} {'tools':>7} {'timeout':>8}",
        "-" * 66,
    ]
    for task, row in overall["tasks"].items():
        lines.append(
            f"{task:<22} {row['n']:>4} {row['accuracy']:>7.1%} "
            f"{row['mean_steps']:>7.1f} {row['mean_tool_calls']:>7.1f} "
            f"{row['timeouts']:>8}"
        )
    for task, row in overall["tasks"].items():
        if row["failure_notes"]:
            lines.append("")
            lines.append(f"{task} failure notes:")
            for note, count in row["failure_notes"].items():
                lines.append(f"- {count}x {note}")
    if payload["failures"]:
        lines.append("")
        lines.append("Failure excerpts:")
        for failure in payload["failures"]:
            lines.append(
                f"\n[{failure['id']}] n={failure['n_nodes']} note={failure['note']}"
            )
            if failure["last_code"]:
                lines.append("code:\n" + failure["last_code"])
            if failure["last_observation"]:
                lines.append("observation:\n" + failure["last_observation"])
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="+")
    ap.add_argument("--failures", type=int, default=5,
                    help="failure excerpts per run (default 5)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", help="write the report to this path")
    args = ap.parse_args(argv)
    payloads = [summarize_run(path, max_failures=args.failures) for path in args.run_dir]
    text = "\n\n".join(format_report(p) for p in payloads)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(payloads, indent=2))
    else:
        print(text)


if __name__ == "__main__":
    main()
