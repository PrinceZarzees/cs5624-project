#!/usr/bin/env python3
"""E9: cost curves and the large-graph counterfactual.

CPU only, arithmetic over measured token/time numbers. No model calls.

Assumptions (also written to E9_costs/cost_assumptions.md):
- USD per million tokens: $0.50 input, $1.50 output.
- The optimizer logs only total tokens; 80/20 prompt/completion split is assumed.
- B200-hour price: $4.00 (cloud reference, not the ARC charge).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/final/E9_costs"
OUT.mkdir(parents=True, exist_ok=True)

IN_PER_M, OUT_PER_M, GPU_HOUR_USD = 0.50, 1.50, 4.00

RUNS = [
    ("SkillOpt", "baseline_a", "experiments/qwen3_4b_opt_a/optimizer"),
    ("SkillOpt", "baseline_b_v2", "experiments/qwen3_4b/optimizer"),
    ("GEPA", "baseline_a", "experiments/review_optimizers/gepa_a"),
    ("RRSI-lite", "baseline_a", "experiments/review_optimizers2/baselinea"),
    ("RRSI-lite", "baseline_b_v2", "experiments/review_optimizers2/baselinebv2"),
    ("Meta-Harness-lite", "baseline_a", "experiments/review_meta_harness"),
]


def load_summary(path: Path):
    for name in ("optimization_summary.json", "gepa_summary.json", "meta_summary.json", "summary.json"):
        p = path / name
        if p.exists():
            return json.loads(p.read_text())
    return {}


def discover_final_runs():
    """Add E3/E4 runs produced by run_e3_lite.sbatch / run_e4.sbatch."""
    known = {(opt, start, rel) for opt, start, rel in RUNS}
    for base, fixed in [("experiments/final/E3_optimizers", None),
                        ("experiments/final/E4_evoskill", "EvoSkill-style")]:
        d = ROOT / base
        if not d.exists():
            continue
        for sub in sorted(p for p in d.iterdir() if p.is_dir()):
            if not (sub / "history.jsonl").exists():
                continue
            name = sub.name
            if fixed:
                opt, start = fixed, name
            elif name.startswith("skillopt"):
                opt, start = "SkillOpt", name.split("_", 1)[-1]
            elif name.startswith("gepa"):
                opt, start = "GEPA", name.split("_", 1)[-1]
            elif name.startswith("rrsi"):
                opt, start = "RRSI-lite", name.split("_", 1)[-1]
            elif name.startswith("meta"):
                opt, start = "Meta-Harness-lite", name.split("_", 1)[-1]
            else:
                opt, start = name, ""
            rel = str(sub.relative_to(ROOT))
            if (opt, start, rel) not in known:
                RUNS.append((opt, start, rel))
                known.add((opt, start, rel))


def main():
    pilot_tokens = []
    for opt, start, rel in list(RUNS):
        d = ROOT / rel
        try:
            summ = load_summary(d)
        except Exception:
            continue
        tok = int(summ.get("total_tokens") or 0)
        if tok > 0:
            pilot_tokens.append(tok)
    pilot_mean = sum(pilot_tokens) / len(pilot_tokens) if pilot_tokens else 0
    discover_final_runs()
    curves = []
    per_run = []
    for opt, start, rel in RUNS:
        d = ROOT / rel
        hist = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines() if l.strip()] if (d / "history.jsonl").exists() else []
        summ = load_summary(d)
        cum, best = 0, None
        for row in hist:
            cum += int(row.get("tokens") or 0)
            best = row.get("best_val_accuracy", best)
            curves.append({"optimizer": opt, "start": start, "round": row.get("round"),
                           "cumulative_tokens": cum, "best_val": best,
                           "accepted": row.get("accepted")})
        total = int(summ.get("total_tokens") or cum)
        if total <= 0:
            continue
        dur = float(summ.get("duration_s") or 0)
        usd_tok = (0.8 * total / 1e6) * IN_PER_M + (0.2 * total / 1e6) * OUT_PER_M
        per_run.append({"optimizer": opt, "start": start, "tokens": total, "wall_s": dur,
                        "gpu_hours": dur / 3600, "usd_tokens": usd_tok,
                        "usd_gpu_hours": dur / 3600 * GPU_HOUR_USD,
                        "initial_val": summ.get("initial_val_accuracy"),
                        "best_val": summ.get("best_val_accuracy")})
    with open(OUT / "curves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curves[0])); w.writeheader(); w.writerows(curves)
    with open(OUT / "per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0])); w.writeheader(); w.writerows(per_run)

    # measured per-problem tokens per split from the frozen scaling run
    ev = list(csv.DictReader(open(OUT / "per_eval.csv")))
    per_problem, per_problem_s = {}, {}
    for r in ev:
        per_problem.setdefault(r["split"], []).append(float(r["mean_tokens"]))
        per_problem_s.setdefault(r["split"], []).append(float(r["mean_duration_s"]))
    per_problem = {k: sum(v) / len(v) for k, v in per_problem.items()}
    per_problem_s = {k: sum(v) / len(v) for k, v in per_problem_s.items()}

    # optimizer protocol: 4 train/task + 10 val/task = 20 + 50 problems per round
    train_n, val_n, rounds = 20, 50, 10
    rows = []
    for scenario, split in [("optimize on small", "test_small"),
                            ("optimize on medium", "test_medium"),
                            ("optimize on large", "test_large")]:
        tpp = per_problem.get(split, 0)
        est = tpp * (train_n + val_n) * rounds
        spp = per_problem_s.get(split, 0)
        est_s = spp * (train_n + val_n) * rounds
        rows.append({"scenario": scenario, "split": split,
                     "tokens_per_problem": round(tpp, 1),
                     "seconds_per_problem": round(spp, 2),
                     "train_per_round": train_n, "val_per_round": val_n, "rounds": rounds,
                     "est_tokens": round(est),
                     "est_wall_s": round(est_s),
                     "est_gpu_hours": round(est / 1e6 * 0.8, 2),
                     "est_usd_tokens": round((0.8 * est / 1e6) * IN_PER_M + (0.2 * est / 1e6) * OUT_PER_M, 2)})
    # fixed denominator: mean tokens of the six original 4B-critic pilot runs
    for r in rows:
        r["ratio_vs_actual_small_run"] = round(r["est_tokens"] / pilot_mean, 2) if pilot_mean else None
    with open(OUT / "counterfactual.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    (OUT / "cost_assumptions.md").write_text(
        "# E9 cost assumptions\n\n"
        f"- Token prices: ${IN_PER_M:.2f}/M input, ${OUT_PER_M:.2f}/M output.\n"
        "- Optimizer logs store total tokens only; assume 80% prompt / 20% completion.\n"
        f"- GPU-hour price: ${GPU_HOUR_USD:.2f}/B200-hour (cloud reference; ARC is a class allocation).\n"
        "- Counterfactual protocol: 20 train + 50 val problems per round, 10 rounds.\n"
        "- `est_gpu_hours` uses 0.8M tokens per B200-hour, measured from the optimizer pilots.\n"
        "- Counterfactual is arithmetic from measured per-problem tokens, not a new run.\n"
        f"- Ratio denominator: mean of the six original 4B-critic pilot runs ({pilot_mean/1e6:.2f}M tokens).\n")
    print(f"wrote curves.csv ({len(curves)} rows), per_run.csv, counterfactual.csv")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
