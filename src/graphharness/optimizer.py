"""Textual harness optimization on small graphs (Phase 3).

The loop is deliberately close to SkillOpt/GEPA:

1. Run the current harness on a small batch from ``train``.
2. Collect failure trajectories and a few successes.
3. Ask a critic model for a revised system prompt / rule list.
4. Evaluate the candidate on a held-out ``val`` sample.
5. Keep it if validation accuracy improves (or ties with a shorter prompt).

Only the harness changes; model weights and the task data do not.  Every round
is logged and every candidate is checkpointed, so a run can be inspected or
stopped and resumed from ``best_harness.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .agent import Harness
from .cli_common import add_model_arguments, build_model, harness_from_arg
from .data_loader import DEFAULT_DATA_DIR, load_split
from .jsonutil import loads_relaxed
from .models import Model
from .runner import ProblemResult, evaluate_records


@dataclass
class OptimizerConfig:
    rounds: int = 15
    train_per_task: int = 6
    val_per_task: int = 5
    max_steps: int | None = None
    workers: int = 1
    seed: int = 5624
    min_gain: float = 0.0
    max_rules: int = 16
    max_rule_chars: int = 500
    max_prompt_chars: int = 4000
    max_examples: int = 12
    critic_temperature: float = 0.7


def sample_records(records: list[dict], per_task: int, rng: random.Random) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["task"], []).append(record)
    kept = []
    for task in sorted(groups):
        pool = groups[task]
        k = min(per_task, len(pool))
        kept.extend(rng.sample(pool, k))
    return sorted(kept, key=lambda r: r["id"])


def accuracy(results: list[ProblemResult]) -> float:
    return sum(bool(r.score.get("correct")) for r in results) / len(results) if results else 0.0


def _evaluate(harness: Harness, model: Model, records: list[dict],
              config: OptimizerConfig) -> list[ProblemResult]:
    if config.workers <= 1:
        return evaluate_records(harness, model, records, max_steps=config.max_steps)
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = [
            pool.submit(evaluate_records, harness, model, [record], config.max_steps, None)
            for record in records
        ]
        return [result for future in futures for result in future.result()]


def _short(value, limit: int = 500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def _last_tool_output(result: ProblemResult) -> str:
    if result.agent is None:
        return result.error or "(no trajectory)"
    for step in reversed(result.agent.steps):
        if step.get("action") == "code":
            out = []
            if step.get("stdout"):
                out.append("stdout: " + _short(step["stdout"], 700))
            if step.get("stderr"):
                out.append("stderr: " + _short(step["stderr"], 700))
            if step.get("timed_out"):
                out.append("timed out")
            return "\n".join(out) or "(no output)"
    return "(no code was run)"


def format_example(result: ProblemResult) -> str:
    record = result.record or {}
    expected = record.get("answer")
    if isinstance(expected, dict):
        expected = {k: v for k, v in expected.items() if k in {"path", "cycle", "clique",
                                                                "representatives", "length",
                                                                "size", "count", "diameter",
                                                                "reachable"}}
    question = _short(record.get("question", ""), 700)
    return (
        f"TASK: {result.task}  (n={result.n_nodes})\n"
        f"QUESTION: {question}\n"
        f"EXPECTED: {_short(expected, 700)}\n"
        f"MODEL FINAL ANSWER: {_short(result.response or '(none)', 700)}\n"
        f"SCORE: correct={result.score.get('correct')} feasible={result.score.get('feasible')} "
        f"note={result.score.get('note') or '(none)'}\n"
        f"LAST TOOL OUTPUT: {_last_tool_output(result)}"
    )


CRITIC_SYSTEM = """\
You are optimizing the harness of an LLM graph-reasoning agent. The harness is
a system prompt plus a short list of rules; the model can run NetworkX code in a
sandbox and must finish with a bracketed node list.

The harness is optimized ONLY on small graphs (4-50 nodes) but will later be
frozen and evaluated on graphs with 100-2000 nodes. Your edits must therefore
teach transferable strategy, not dataset-specific facts.

Hard constraints:
- Never mention specific node ids, specific answers, or details of the examples.
- Never add task-specific lookup tables, memorized answers, or code.
- Prefer a small number of general rules about graph algorithms and verification.
- Preserve what already works; change only what the evidence supports.
- Rules must remain useful when the graph is much larger than the training graphs.

Return ONLY a JSON object of this form:
{
  "system_prompt": "revised system prompt",
  "rules": ["rule 1", "rule 2"],
  "graph_view": "summary" | "edge_list" | "adjacency_list",
  "reason": "one or two sentences explaining the edit"
}

Keep the full prompt under 4000 characters and use at most 12 rules.
"""


def critic_prompt(harness: Harness, train_results: list[ProblemResult],
                  config: OptimizerConfig) -> str:
    failures = [r for r in train_results if not r.score.get("correct")]
    successes = [r for r in train_results if r.score.get("correct")]
    chosen = failures[:config.max_examples]
    if len(chosen) < config.max_examples:
        chosen += successes[:config.max_examples - len(chosen)]
    examples = "\n\n---\n\n".join(format_example(r) for r in chosen)
    return (
        "## Current harness\n"
        f"{json.dumps(harness.to_dict(), indent=2)}\n\n"
        "## Observed cases on small training graphs\n"
        f"{examples}\n\n"
        "## Your task\n"
        "Propose the next harness. Focus on why the failures happened and on "
        "rules that will still hold at 100-2000 nodes. Do not repeat the same "
        "failed approach. Return only the JSON object."
    )


def extract_proposal(text: str) -> dict | None:
    """Find the last JSON object that looks like a harness proposal."""
    candidates: list[dict] = []
    # Fenced JSON first.
    for block in re.finditer(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE):
        try:
            value = loads_relaxed(block.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    # Then balanced objects in the prose.
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, quote, escape = 0, False, False
        for j in range(i, len(text)):
            ch = text[j]
            if quote:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    quote = False
            else:
                if ch == '"':
                    quote = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            value = loads_relaxed(text[i:j + 1])
                        except json.JSONDecodeError:
                            pass
                        else:
                            if isinstance(value, dict):
                                candidates.append(value)
                        i = j
                        break
        i += 1
    for value in reversed(candidates):
        if "rules" in value or "system_prompt" in value:
            return value
    return None


def validate_proposal(proposal: dict, current: Harness,
                      config: OptimizerConfig) -> tuple[Harness | None, str]:
    system_prompt = proposal.get("system_prompt", current.system_prompt)
    rules = proposal.get("rules", current.rules)
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        return None, "system_prompt is missing or empty"
    if not isinstance(rules, list) or not all(isinstance(r, str) for r in rules):
        return None, "rules must be a list of strings"
    rules = [r.strip() for r in rules if r.strip()]
    if len(rules) > config.max_rules:
        return None, f"too many rules ({len(rules)} > {config.max_rules})"
    too_long = [i for i, r in enumerate(rules) if len(r) > config.max_rule_chars]
    if too_long:
        return None, f"rule(s) too long: {too_long[:5]}"
    if len(system_prompt) > config.max_prompt_chars:
        return None, "system prompt is too long"
    graph_view = proposal.get("graph_view", current.graph_view)
    if graph_view not in {"summary", "edge_list", "adjacency_list"}:
        return None, f"invalid graph_view {graph_view!r}"
    candidate = current.copy(
        system_prompt=system_prompt,
        rules=rules,
        graph_view=graph_view,
        name=f"{current.name}-candidate",
    )
    if len(candidate.system_message()) > config.max_prompt_chars + 4000:
        return None, "full harness prompt is too long"
    reason = str(proposal.get("reason", "")).strip()[:1000]
    return candidate.copy(notes=reason), reason


def _harness_hash(harness: Harness) -> str:
    blob = json.dumps(harness.to_dict(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class OptimizationResult:
    best_harness: Harness
    best_val_accuracy: float
    initial_val_accuracy: float
    history: list[dict] = field(default_factory=list)
    total_tokens: int = 0
    out_dir: str = ""


def optimize(
    start: Harness,
    actor: Model,
    critic: Model,
    train: list[dict],
    val: list[dict],
    out_dir: str | Path,
    config: OptimizerConfig | None = None,
) -> OptimizationResult:
    config = config or OptimizerConfig()
    if not train:
        raise ValueError("training split is empty")
    if not val:
        raise ValueError("validation split is empty")
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    val_records = sample_records(val, config.val_per_task, random.Random(config.seed + 1))
    current = start
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    total_tokens = 0
    history: list[dict] = []

    initial_results = _evaluate(current, actor, val_records, config)
    best_val = accuracy(initial_results)
    total_tokens += sum(r.agent.total_tokens for r in initial_results if r.agent)
    best = current
    current.save(out_dir / "best_harness.json")
    initial_row = {
        "round": 0,
        "accepted": True,
        "role": "initial",
        "val_accuracy": best_val,
        "best_val_accuracy": best_val,
        "harness_hash": _harness_hash(current),
        "tokens": total_tokens,
    }
    history.append(initial_row)
    _append_history(out_dir, initial_row)

    for round_index in range(1, config.rounds + 1):
        train_records = sample_records(
            train, config.train_per_task, random.Random(config.seed * 1000 + round_index)
        )
        train_results = _evaluate(current, actor, train_records, config)
        train_accuracy = accuracy(train_results)
        round_tokens = sum(r.agent.total_tokens for r in train_results if r.agent)

        prompt = critic_prompt(current, train_results, config)
        critique = critic.chat([
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": prompt},
        ], temperature=config.critic_temperature)
        round_tokens += critique.prompt_tokens + critique.completion_tokens
        proposal = extract_proposal(critique.text)
        row = {
            "round": round_index,
            "train_accuracy": train_accuracy,
            "critic_response": critique.text[:4000],
            "tokens": round_tokens,
        }
        if proposal is None:
            row.update(accepted=False, reason="could not parse a JSON proposal",
                       best_val_accuracy=best_val)
        else:
            candidate, reason = validate_proposal(proposal, current, config)
            if candidate is None:
                row.update(accepted=False, reason=f"rejected proposal: {reason}",
                           proposal=proposal, best_val_accuracy=best_val)
            else:
                candidate.save(out_dir / "candidates" / f"round_{round_index:02d}.json")
                val_results = _evaluate(candidate, actor, val_records, config)
                val_accuracy = accuracy(val_results)
                round_tokens += sum(r.agent.total_tokens for r in val_results if r.agent)
                shorter = len(candidate.system_message()) < len(current.system_message())
                accepted = (
                    val_accuracy > best_val + config.min_gain
                    or (val_accuracy == best_val and shorter)
                )
                row.update(
                    accepted=accepted,
                    reason=reason,
                    proposal=proposal,
                    val_accuracy=val_accuracy,
                    best_val_accuracy=max(best_val, val_accuracy),
                    candidate_hash=_harness_hash(candidate),
                    candidate_prompt_chars=len(candidate.system_message()),
                    tokens=round_tokens,
                )
                if accepted:
                    current = candidate
                    best = candidate
                    best_val = val_accuracy
                    best.save(out_dir / "best_harness.json")
                    current.save(out_dir / "checkpoints" / f"round_{round_index:02d}.json")
        total_tokens += round_tokens
        row["total_tokens"] = total_tokens
        history.append(row)
        _append_history(out_dir, row)
        status = "ACCEPT" if row.get("accepted") else "reject"
        print(
            f"round {round_index:>2}/{config.rounds}: train {train_accuracy:.3f} "
            f"val {row.get('val_accuracy', float('nan')):.3f} best {best_val:.3f} "
            f"[{status}] {row.get('reason', '')[:90]}",
            flush=True,
        )

    finished_at = datetime.now(timezone.utc).isoformat()
    result = OptimizationResult(
        best_harness=best,
        best_val_accuracy=best_val,
        initial_val_accuracy=accuracy(initial_results),
        history=history,
        total_tokens=total_tokens,
        out_dir=str(out_dir),
    )
    summary = {
        "start_harness": start.to_dict(),
        "best_harness": best.to_dict(),
        "config": config.__dict__,
        "initial_val_accuracy": result.initial_val_accuracy,
        "best_val_accuracy": best_val,
        "rounds": config.rounds,
        "total_tokens": total_tokens,
        "duration_s": time.perf_counter() - started,
        "started_at": started_at,
        "finished_at": finished_at,
        "history": history,
    }
    (out_dir / "optimization_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    best.save(out_dir / "best_harness.json")
    return result


def _append_history(out_dir: Path, row: dict) -> None:
    with open(out_dir / "history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="baseline_b",
                    help="starting harness name or JSON checkpoint")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--tasks", nargs="*", help="only optimize on these task names")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--train-per-task", type=int, default=6)
    ap.add_argument("--val-per-task", type=int, default=5)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=5624)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--out", default="experiments/optimizer")
    ap.add_argument("--critic-model", help="separate critic model (default: actor model)")
    ap.add_argument("--critic-base-url")
    ap.add_argument("--critic-temperature", type=float, default=0.7)
    ap.add_argument("--critic-max-tokens", type=int, default=1500)
    add_model_arguments(ap)
    args = ap.parse_args(argv)

    start = harness_from_arg(args.start)
    actor = build_model(args)
    critic_args = argparse.Namespace(**vars(args))
    critic_args.model = args.critic_model or args.model
    critic_args.base_url = args.critic_base_url or args.base_url
    critic_args.temperature = args.critic_temperature
    critic_args.max_tokens = args.critic_max_tokens
    critic = build_model(critic_args)
    train = load_split(args.train_split, args.data_dir, tasks=args.tasks)
    val = load_split(args.val_split, args.data_dir, tasks=args.tasks)
    config = OptimizerConfig(
        rounds=args.rounds,
        train_per_task=args.train_per_task,
        val_per_task=args.val_per_task,
        max_steps=args.max_steps,
        workers=args.workers,
        seed=args.seed,
        min_gain=args.min_gain,
        critic_temperature=args.critic_temperature,
    )
    try:
        result = optimize(start, actor, critic, train, val, args.out, config)
    except KeyboardInterrupt:
        print("\ninterrupted; inspect the latest files under " + args.out, file=sys.stderr)
        raise SystemExit(130) from None
    print()
    print(f"best validation accuracy: {result.best_val_accuracy:.3f}")
    print(f"initial validation accuracy: {result.initial_val_accuracy:.3f}")
    print(f"total tokens: {result.total_tokens}")
    print(f"best harness: {Path(result.out_dir) / 'best_harness.json'}")


if __name__ == "__main__":
    main()
