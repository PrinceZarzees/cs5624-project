#!/usr/bin/env python3
"""Aggregate E3/E4 optimizer runs into one summary CSV.

Reads the run directories written by optimizer.py / gepa.py / meta_harness.py /
evoskill.py under experiments/final/E3_optimizers and E4_evoskill.
No model calls.

Cost assumptions match scripts/final/e9_costs.py: 80/20 prompt/completion split,
$0.50/$1.50 per M tokens.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
E3 = ROOT / "experiments/final/E3_optimizers"
E4 = ROOT / "experiments/final/E4_evoskill"
IN_PER_M, OUT_PER_M = 0.50, 1.50


def find_summary(d: Path):
    for name in ("optimization_summary.json", "gepa_summary.json",
                 "meta_summary.json", "evoskill_summary.json", "summary.json"):
        p = d / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def history(d: Path):
    p = d / "history.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def label(d: Path, data: dict):
    name = d.name
    known = {"skillopt": "SkillOpt-style", "gepa": "GEPA-style",
             "rrsi": "RRSI-lite", "meta": "Meta-Harness-lite",
             "evoskill": "EvoSkill-style"}
    opt = next((v for k, v in known.items() if name.startswith(k)), name)
    start = data.get("start_harness", {}).get("name") or name.split("_", 1)[-1]
    return opt, start


def main():
    rows = []
    for base, default_opt in [(E3, None), (E4, "EvoSkill-style")]:
        if not base.exists():
            continue
        for d in sorted(p for p in base.iterdir() if p.is_dir()):
            if d.name.endswith("_mixed"):
                continue
            data = find_summary(d)
            if not data:
                continue
            opt, start = label(d, data)
            if default_opt == "EvoSkill-style" and base == E4:
                opt = "EvoSkill-style"
                start = d.name
            hist = history(d)
            if len(data.get("history") or []) > len(hist):
                hist = data["history"]
            tokens = int(data.get("total_tokens") or sum(int(h.get("tokens") or 0) for h in hist))
            wall = float(data.get("duration_s") or 0)
            usd = (0.8 * tokens / 1e6) * IN_PER_M + (0.2 * tokens / 1e6) * OUT_PER_M
            rnds = [h for h in hist if (h.get("round") or 0) >= 1]
            proposals = len(rnds)
            parsed = sum(1 for h in rnds if h.get("proposal") is not None or
                         (h.get("reason") and "no valid candidate" not in h["reason"].lower()
                          and "parse" not in h["reason"].lower()))
            accepted = sum(1 for h in rnds if h.get("accepted"))
            evals = {}
            evdir = E3 / f"eval_{d.name}"
            for split in ("test_small_fresh", "test_medium", "test_large", "test_stress"):
                sp = evdir / split / "summary.json"
                if sp.exists():
                    evals[split] = json.loads(sp.read_text())["overall"]["accuracy"]
            gap = (evals.get("test_small_fresh") - evals.get("test_large")
                   if "test_small_fresh" in evals and "test_large" in evals else "")
            rows.append({
                "optimizer": opt, "start": start,
                "critic": "Qwen2.5-7B-Instruct BF16",
                "rounds": len(rnds), "proposals": proposals,
                "parsed_proposals": parsed, "accepted": accepted,
                "init_val": data.get("initial_val_accuracy"),
                "best_val": data.get("best_val_accuracy"),
                "actor_tokens": "", "critic_tokens": "",
                "total_tokens": tokens, "wall_s": round(wall, 1),
                "est_usd": round(usd, 2),
                "small_fresh": evals.get("test_small_fresh", ""),
                "medium": evals.get("test_medium", ""),
                "large": evals.get("test_large", ""),
                "stress": evals.get("test_stress", ""),
                "gen_gap": gap, "note": "" if accepted else "no accepted edit",
            })
    out = E3 / "summary.csv"
    if not rows:
        print("no optimizer runs found")
        return
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
