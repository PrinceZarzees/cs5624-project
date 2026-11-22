"""GEPA-style prompt evolution baseline.

This is a lightweight, reproducible baseline: each round asks the critic for
several revised system prompts, evaluates each on a fixed small-graph
validation sample, and keeps the best prompt. Rules are held fixed, which
isolates prompt evolution from skill/rule evolution.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .agent import Harness
from .cli_common import add_model_arguments, build_model, harness_from_arg
from .data_loader import DEFAULT_DATA_DIR, load_split
from .models import Model
from .optimizer import (
    CRITIC_SYSTEM,
    OptimizerConfig,
    _append_history,
    _evaluate,
    accuracy,
    critic_prompt,
    extract_proposal,
    sample_records,
)


@dataclass
class GEPAConfig:
    rounds: int = 3
    candidates_per_round: int = 3
    train_per_task: int = 5
    val_per_task: int = 5
    max_steps: int | None = None
    workers: int = 1
    seed: int = 5624
    critic_temperature: float = 0.8


def optimize_gepa(
    start: Harness,
    actor: Model,
    critic: Model,
    train: list[dict],
    val: list[dict],
    out_dir: str | Path,
    config: GEPAConfig | None = None,
) -> dict:
    config = config or GEPAConfig()
    out_dir = Path(out_dir)
    (out_dir / "candidates").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()
    base_eval = OptimizerConfig(max_steps=config.max_steps, workers=config.workers,
                                seed=config.seed)
    val_records = sample_records(val, config.val_per_task, random.Random(config.seed + 1))
    initial = _evaluate(start, actor, val_records, base_eval)
    best = start
    best_val = accuracy(initial)
    total_tokens = sum(r.agent.total_tokens for r in initial if r.agent)
    history = [{"round": 0, "role": "initial", "val_accuracy": best_val,
                "best_val_accuracy": best_val, "tokens": total_tokens}]
    _append_history(out_dir, history[0])
    best.save(out_dir / "best_harness.json")
    started = time.perf_counter()

    for round_index in range(1, config.rounds + 1):
        train_records = sample_records(
            train, config.train_per_task, random.Random(config.seed * 1000 + round_index)
        )
        train_results = _evaluate(best, actor, train_records, base_eval)
        total_tokens += sum(r.agent.total_tokens for r in train_results if r.agent)
        candidates = []
        for candidate_index in range(config.candidates_per_round):
            critique = critic.chat([
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": critic_prompt(best, train_results, base_eval)},
            ], temperature=config.critic_temperature + 0.05 * candidate_index)
            total_tokens += critique.prompt_tokens + critique.completion_tokens
            proposal = extract_proposal(critique.text)
            if proposal is None:
                continue
            system_prompt = proposal.get("system_prompt")
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                continue
            candidate = best.copy(system_prompt=system_prompt.strip(),
                                  name=f"{best.name}-gepa-{round_index}-{candidate_index}")
            if candidate.system_prompt == best.system_prompt:
                continue
            candidate.save(out_dir / "candidates" / f"round_{round_index:02d}_{candidate_index:02d}.json")
            results = _evaluate(candidate, actor, val_records, base_eval)
            total_tokens += sum(r.agent.total_tokens for r in results if r.agent)
            candidates.append({
                "candidate": candidate,
                "accuracy": accuracy(results),
                "proposal": proposal,
            })
        if candidates:
            winner = max(candidates, key=lambda c: c["accuracy"])
            accepted = winner["accuracy"] > best_val
            row = {
                "round": round_index,
                "train_accuracy": accuracy(train_results),
                "candidates": [{"accuracy": c["accuracy"],
                                "reason": c["proposal"].get("reason", "")}
                               for c in candidates],
                "accepted": accepted,
                "reason": winner["proposal"].get("reason", ""),
                "val_accuracy": winner["accuracy"],
                "best_val_accuracy": max(best_val, winner["accuracy"]),
                "tokens": total_tokens,
            }
            if accepted:
                best = winner["candidate"]
                best_val = winner["accuracy"]
                best.save(out_dir / "best_harness.json")
                best.save(out_dir / "checkpoints" / f"round_{round_index:02d}.json")
        else:
            row = {"round": round_index, "accepted": False,
                   "reason": "no valid candidate", "best_val_accuracy": best_val,
                   "tokens": total_tokens}
        history.append(row)
        _append_history(out_dir, row)
        print(f"GEPA round {round_index}/{config.rounds}: "
              f"best {best_val:.3f} {'ACCEPT' if row.get('accepted') else 'reject'}",
              flush=True)

    summary = {
        "start_harness": start.to_dict(), "best_harness": best.to_dict(),
        "config": config.__dict__, "initial_val_accuracy": accuracy(initial),
        "best_val_accuracy": best_val, "total_tokens": total_tokens,
        "duration_s": time.perf_counter() - started,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "history": history, "optimizer": "gepa",
    }
    (out_dir / "gepa_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="baseline_a")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--candidates-per-round", type=int, default=3)
    ap.add_argument("--train-per-task", type=int, default=5)
    ap.add_argument("--val-per-task", type=int, default=5)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--seed", type=int, default=5624)
    ap.add_argument("--out", default="experiments/review_gepa")
    ap.add_argument("--critic-model")
    ap.add_argument("--critic-base-url")
    add_model_arguments(ap)
    args = ap.parse_args(argv)
    start = harness_from_arg(args.start)
    actor = build_model(args)
    critic_args = argparse.Namespace(**vars(args))
    critic_args.model = args.critic_model or args.model
    critic_args.base_url = args.critic_base_url or args.base_url
    critic_args.temperature = 0.8
    critic = build_model(critic_args)
    train = load_split(args.train_split, args.data_dir)
    val = load_split(args.val_split, args.data_dir)
    summary = optimize_gepa(start, actor, critic, train, val, args.out, GEPAConfig(
        rounds=args.rounds, candidates_per_round=args.candidates_per_round,
        train_per_task=args.train_per_task, val_per_task=args.val_per_task,
        max_steps=args.max_steps, seed=args.seed,
    ))
    print(json.dumps({k: summary[k] for k in
                      ("initial_val_accuracy", "best_val_accuracy", "total_tokens")},
                     indent=2))


if __name__ == "__main__":
    main()
