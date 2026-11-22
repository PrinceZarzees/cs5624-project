"""O6 cost ledger: sum optimizer ledgers and Slurm accounting into cost.csv."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

COLS = ["optimizer", "hda_model", "proposer_model", "rollouts", "rollouts_cached",
        "hda_llm_calls", "hda_prompt_tokens", "hda_completion_tokens",
        "proposer_calls", "proposer_prompt_tokens", "proposer_completion_tokens",
        "proposals", "proposals_parsed", "proposals_rejected_by_validator", "accepted",
        "wall_h", "gpu_h", "cpu_h_eval", "hda_tokens_per_rollout"]

# Slurm accounting; elapsed_s x gpus = GPU-seconds. Split shared jobs by wall_s share.
JOBS = {
    "shared_baseline": dict(cluster="owl", job="1554804", elapsed_s=712, gpus=1, cpus=16),
    "gepa": dict(cluster="falcon", job="926163", elapsed_s=3648, gpus=1, cpus=32),
    "skillopt": dict(cluster="falcon", job="926441", elapsed_s=0, gpus=1, cpus=16),
    "meta_harness": dict(cluster="falcon", job="926441", elapsed_s=0, gpus=1, cpus=16),
    "evoskill": dict(cluster="falcon", job="926441", elapsed_s=0, gpus=1, cpus=16),
}


def _history(d: Path):
    p = d / "history.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def _ledger(d: Path):
    p = d / "ledger.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def optimizer_row(name: str, d: Path, hda_model: str, proposer: str) -> dict:
    hist, led = _history(d), _ledger(d)
    cache = list((d / "rollouts" / "cache").glob("*.json")) if (d / "rollouts" / "cache").exists() else []
    requested = sum(int(x.get("n", 0)) for x in led)
    final = d / "final_optimizer.json"
    fin = json.loads(final.read_text()) if final.exists() else {}
    # proposer accounting: per-step deltas when present, else only the final totals
    pc = sum(int(x.get("proposer_calls", 0) or 0) for x in hist)
    ppt = sum(int(x.get("proposer_prompt_tokens", 0) or 0) for x in hist)
    pct = sum(int(x.get("proposer_completion_tokens", 0) or 0) for x in hist)
    if pc == 0:
        pc = int(fin.get("proposer_calls", 0) or 0)
        ppt = int(fin.get("proposer_prompt_tokens", 0) or 0)
        pct = int(fin.get("proposer_completion_tokens", 0) or 0)
    rejected = sum(1 for x in hist if not (x.get("validator") or {}).get("ok", False))
    job = JOBS.get(name, {})
    elapsed = job.get("elapsed_s", 0)
    gpu_h = elapsed * job.get("gpus", 0) / 3600.0
    wall_h = elapsed / 3600.0
    hda_calls = sum(int(x.get("hda_llm_calls", 0)) for x in led)
    hda_pt = sum(int(x.get("hda_prompt_tokens", 0)) for x in led)
    hda_ct = sum(int(x.get("hda_completion_tokens", 0)) for x in led)
    if name == "shared_baseline":
        pass
    cpu_h = 0.0  # filled from rollout score_s when available; not separately timed
    return {
        "optimizer": name, "hda_model": hda_model, "proposer_model": proposer,
        "rollouts": len(cache) if cache else (requested if name != "shared_baseline" else 60),
        "rollouts_cached": max(0, requested - len(cache)),
        "hda_llm_calls": hda_calls, "hda_prompt_tokens": hda_pt,
        "hda_completion_tokens": hda_ct,
        "proposer_calls": pc, "proposer_prompt_tokens": ppt,
        "proposer_completion_tokens": pct,
        "proposals": len(hist), "proposals_parsed": len(hist) - rejected,
        "proposals_rejected_by_validator": rejected,
        "accepted": sum(1 for x in hist if x.get("accepted")),
        "wall_h": round(wall_h, 2), "gpu_h": round(gpu_h, 3),
        "cpu_h_eval": round(cpu_h, 2),
        "hda_tokens_per_rollout": round((hda_pt + hda_ct) / max(1, len(cache) or requested), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../experiments/hda/optimizers")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.root)
    rows = []

    base = root / "baseline"
    b = json.loads((base / "baseline_A.json").read_text())
    rows.append({
        "optimizer": "shared_baseline", "hda_model": "qwen3-4b-q4",
        "proposer_model": "-", "rollouts": b["n"], "rollouts_cached": 0,
        "hda_llm_calls": b["llm_calls"], "hda_prompt_tokens": b["tokens_prompt"],
        "hda_completion_tokens": b["tokens_completion"], "proposer_calls": 0,
        "proposer_prompt_tokens": 0, "proposer_completion_tokens": 0,
        "proposals": 0, "proposals_parsed": 0, "proposals_rejected_by_validator": 0,
        "accepted": 0, "wall_h": round(712 / 3600, 2), "gpu_h": round(712 / 3600, 3),
        "cpu_h_eval": 0.0,
        "hda_tokens_per_rollout": round((b["tokens_prompt"] + b["tokens_completion"]) / b["n"], 1),
    })

    for name in ("gepa", "skillopt", "meta_harness", "evoskill"):
        d = root / name
        if d.exists():
            rows.append(optimizer_row(name, d, "qwen3-4b-q4" if name == "gepa" else "qwen3-4b-bf16",
                                      "qwen3-4b-q4" if name == "gepa" else "qwen3-4b-bf16"))
    out = Path(a.out) if a.out else root / "cost.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(out)
    for r in rows:
        print(r["optimizer"], r["rollouts"], r["accepted"], r["gpu_h"])


if __name__ == "__main__":
    main()
