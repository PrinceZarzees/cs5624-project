#!/usr/bin/env python3
"""Build the E5 candidate pool (harness JSONs) and deduplicate."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphharness.agent import BASELINE_A, BASELINE_B, BASELINE_B_V2, Harness  # noqa: E402

OUT = ROOT / "experiments/final/E5_proxy"
POOL = OUT / "pool"
POOL.mkdir(parents=True, exist_ok=True)


def key(h: Harness) -> str:
    blob = json.dumps([h.system_prompt, h.rules, h.graph_view, h.allow_tools,
                       h.tool_preamble, h.tool_backend], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)[:80]


def history_val(run_dir: Path, round_no: int):
    hp = run_dir / "history.jsonl"
    if not hp.exists():
        return None
    for line in hp.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("round") == round_no:
            return row.get("val_accuracy")
    return None


def main():
    cands = []  # (id, source, harness, val)
    cands += [("baseline_a", "baselines", BASELINE_A, None),
              ("baseline_b", "baselines", BASELINE_B, None),
              ("baseline_b_v2", "baselines", BASELINE_B_V2, None)]
    # 14 one-rule-removed v2 variants
    abl = json.loads((ROOT / "experiments/qwen3_4b_ablation_val/ablation/ablation_summary.json").read_text())
    by_var = {r["variant"]: r for r in abl["rows"]}
    for i in range(14):
        name = f"baseline_b_v2-no-rule-{i}"
        cands.append((name, "ablation", BASELINE_B_V2.without_rule(i), by_var[name]["accuracy"]))
    # router library
    for p in sorted((ROOT / "experiments/review_router/harnesses").glob("*.json")):
        cands.append((f"library_{p.stem}", "library", Harness.load(p), None))

    globs = [
        "experiments/final/E3_optimizers/*/candidates/round_*.json",
        "experiments/final/E4_evoskill/*/candidates/round_*.json",
        "experiments/qwen3_4b_opt_a/optimizer/candidates/round_*.json",
        "experiments/qwen3_4b/optimizer/candidates/round_*.json",
        "experiments/review_optimizers/*/candidates/round_*.json",
        "experiments/review_optimizers2/*/candidates/round_*.json",
        "experiments/review_meta_harness/candidates/round_*.json",
    ]
    for pattern in globs:
        for p in sorted(ROOT.glob(pattern)):
            run_dir = p.parent.parent
            m = re.search(r"round_(\d+)", p.name)
            val = history_val(run_dir, int(m.group(1))) if m else None
            name = f"{run_dir.parent.name}_{run_dir.name}_{p.stem}"
            cands.append((name, str(run_dir.relative_to(ROOT)), Harness.load(p), val))

    seen, rows = set(), []
    for cid, source, h, val in cands:
        k = key(h)
        if k in seen:
            continue
        seen.add(k)
        cid = slug(cid)
        h.save(POOL / f"{cid}.json")
        rows.append({"id": cid, "source": source, "path": f"pool/{cid}.json",
                     "val_accuracy": "" if val is None else val})
    with open(OUT / "pool.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} unique candidates")
    from collections import Counter
    print(Counter(r["source"].split("/")[0] for r in rows))


if __name__ == "__main__":
    main()
