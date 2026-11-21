"""Meta-Harness-lite: LLM-driven, code-level harness evolution.

Unlike the SkillOpt/GEPA loops, a candidate may add a Python helper preamble that
is injected into every sandbox run. This makes the tool loop itself evolvable
while keeping model weights frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .agent import Harness
from .cli_common import add_model_arguments, build_model, harness_from_arg
from .data_loader import DEFAULT_DATA_DIR, load_split
from .models import Model
from .optimizer import (
    OptimizerConfig, _append_history, _evaluate, accuracy, critic_prompt,
    extract_proposal, sample_records, validate_proposal,
)

META_SYSTEM = """\
You are a Meta-Harness optimizer. Improve an LLM graph-reasoning agent by
editing its harness. You may change:
- system_prompt
- rules (general graph-algorithm strategy only)
- graph_view: summary | edge_list | adjacency_list
- tool_backend: python | c
- tool_preamble: optional Python helper code injected into every sandbox run

The helper preamble runs after `G`, `params`, and `nodes` are defined. It may
define reusable functions, but must not contain answers, node-specific
shortcuts, subprocess/network code, or benchmark names.

Prefer reusable mechanisms over dataset-specific rules. Return only JSON:
{"system_prompt": "...", "rules": ["..."], "graph_view": "summary",
 "tool_backend": "python", "tool_preamble": "def ...", "reason": "..."}
"""


@dataclass
class MetaConfig:
    rounds: int = 3
    train_per_task: int = 4
    val_per_task: int = 4
    max_steps: int | None = None
    seed: int = 5624
    critic_temperature: float = 0.7
    max_preamble_chars: int = 3000


def validate_meta(proposal: dict, current: Harness, config: MetaConfig) -> tuple[Harness | None, str]:
    opt = OptimizerConfig(max_steps=config.max_steps, seed=config.seed)
    candidate, reason = validate_proposal(proposal, current, opt)
    if candidate is None:
        return None, reason
    backend = proposal.get("tool_backend", current.tool_backend)
    if backend not in {"python", "c"}:
        return None, f"invalid tool_backend {backend!r}"
    preamble = proposal.get("tool_preamble", current.tool_preamble)
    if not isinstance(preamble, str):
        return None, "tool_preamble must be a string"
    if len(preamble) > config.max_preamble_chars:
        return None, "tool_preamble too long"
    forbidden = ["subprocess", "os.system", "socket", "requests.", "__import__"]
    if any(word in preamble for word in forbidden):
        return None, "tool_preamble contains a forbidden API"
    if re.search(r"\\bnode\\s+\\d+\\b", preamble, re.IGNORECASE):
        return None, "tool_preamble mentions a specific node id"
    return candidate.copy(tool_backend=backend, tool_preamble=preamble,
                          name=f"{current.name}-meta"), str(proposal.get("reason", ""))


def optimize_meta(start: Harness, actor: Model, critic: Model, train: list[dict],
                  val: list[dict], out_dir: str | Path, config: MetaConfig) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()
    opt = OptimizerConfig(max_steps=config.max_steps, seed=config.seed)
    val_records = sample_records(val, config.val_per_task, random.Random(config.seed + 1))
    initial = _evaluate(start, actor, val_records, opt)
    best, best_val = start, accuracy(initial)
    total = sum(r.agent.total_tokens for r in initial if r.agent)
    history = [{"round": 0, "role": "initial", "val_accuracy": best_val,
                "best_val_accuracy": best_val, "tokens": total}]
    _append_history(out_dir, history[0])
    best.save(out_dir / "best_harness.json")
    started = time.perf_counter()
    for round_index in range(1, config.rounds + 1):
        train_records = sample_records(train, config.train_per_task,
                                       random.Random(config.seed * 1000 + round_index))
        train_results = _evaluate(best, actor, train_records, opt)
        total += sum(r.agent.total_tokens for r in train_results if r.agent)
        prompt = critic_prompt(best, train_results, opt) + (
            "\n\nYou may also add a Python helper preamble. Return the JSON "
            "object described in the system message."
        )
        critique = critic.chat([
            {"role": "system", "content": META_SYSTEM},
            {"role": "user", "content": prompt},
        ], temperature=config.critic_temperature)
        total += critique.prompt_tokens + critique.completion_tokens
        proposal = extract_proposal(critique.text)
        row = {"round": round_index, "train_accuracy": accuracy(train_results),
               "critic_response": critique.text[:4000]}
        if proposal is None:
            row.update(accepted=False, reason="could not parse proposal",
                       best_val_accuracy=best_val)
        else:
            candidate, reason = validate_meta(proposal, best, config)
            if candidate is None:
                row.update(accepted=False, reason=reason, proposal=proposal,
                           best_val_accuracy=best_val)
            else:
                candidate.save(out_dir / "candidates" / f"round_{round_index:02d}.json")
                results = _evaluate(candidate, actor, val_records, opt)
                total += sum(r.agent.total_tokens for r in results if r.agent)
                val_acc = accuracy(results)
                accepted = val_acc > best_val + 1e-9
                row.update(accepted=accepted, reason=reason, proposal=proposal,
                           val_accuracy=val_acc, best_val_accuracy=max(best_val, val_acc))
                if accepted:
                    best, best_val = candidate, val_acc
                    best.save(out_dir / "best_harness.json")
                    best.save(out_dir / "checkpoints" / f"round_{round_index:02d}.json")
        row["tokens"] = total
        history.append(row)
        _append_history(out_dir, row)
        print(f"Meta round {round_index}/{config.rounds}: best {best_val:.3f} "
              f"{'ACCEPT' if row.get('accepted') else 'reject'} {row.get('reason','')[:80]}",
              flush=True)
    summary = {"start_harness": start.to_dict(), "best_harness": best.to_dict(),
               "config": config.__dict__, "initial_val_accuracy": accuracy(initial),
               "best_val_accuracy": best_val, "total_tokens": total,
               "duration_s": time.perf_counter() - started,
               "started_at": datetime.now(timezone.utc).isoformat(), "history": history,
               "optimizer": "meta_harness"}
    (out_dir / "meta_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="baseline_a")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--train-per-task", type=int, default=4)
    ap.add_argument("--val-per-task", type=int, default=4)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--seed", type=int, default=5624)
    ap.add_argument("--out", default="experiments/review_meta_harness")
    ap.add_argument("--critic-model")
    add_model_arguments(ap)
    args = ap.parse_args(argv)
    actor = build_model(args)
    critic_args = argparse.Namespace(**vars(args))
    critic_args.model = args.critic_model or args.model
    critic_args.temperature = 0.7
    critic = build_model(critic_args)
    summary = optimize_meta(harness_from_arg(args.start), actor, critic,
                            load_split(args.train_split, args.data_dir),
                            load_split(args.val_split, args.data_dir),
                            args.out, MetaConfig(rounds=args.rounds,
                                                 train_per_task=args.train_per_task,
                                                 val_per_task=args.val_per_task,
                                                 max_steps=args.max_steps,
                                                 seed=args.seed))
    print(json.dumps({k: summary[k] for k in
                      ("initial_val_accuracy", "best_val_accuracy", "total_tokens")},
                     indent=2))


if __name__ == "__main__":
    main()
