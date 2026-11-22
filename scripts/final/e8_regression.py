#!/usr/bin/env python3
"""E8: family x size table and a logistic regression for correctness.

CPU only. Reads the frozen full-scaling scores plus the original splits.
Writes experiments/final/E8_characteristics/{family_x_size.csv,regression.csv}.
Regression: correct ~ log(n) + density + family + task + harness (no interactions),
fitted with Newton-Raphson (statsmodels is not installed on this node).
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphharness.data_loader import load_split, to_graph  # noqa: E402
from graphharness.stats import wilson_ci  # noqa: E402

OUT = ROOT / "experiments/final/E8_characteristics"
SPLITS = ["test_small", "test_medium", "test_large", "test_stress"]
HARNESSES = ["baseline_a", "baseline_b", "baseline_b_v2"]
FAMILIES = ["ba", "er", "ws", "tree", "mixed"]
TASKS = ["shortest_path", "connected_components", "cycle_detection", "diameter", "max_clique"]


def size_bin(n: int) -> str:
    if n <= 50:
        return "4-50"
    if n <= 300:
        return "100-300"
    if n <= 1000:
        return "500-1000"
    return "2000"


def records_by_id():
    out = {}
    for split in SPLITS:
        for rec in load_split(split, str(ROOT / "data")):
            g = to_graph(rec)
            n = g.number_of_nodes()
            m = g.number_of_edges()
            out[rec["id"]] = {
                "task": rec["task"],
                "n": n,
                "density": (2.0 * m / (n * (n - 1))) if n > 1 else 0.0,
                "family": rec.get("meta", {}).get("family", "unknown"),
                "size_bin": size_bin(n),
                "split": split,
            }
    return out


def load_rows():
    recs = records_by_id()
    rows = []
    for h in HARNESSES:
        for split in SPLITS:
            path = ROOT / f"experiments/qwen3_4b_scaling_full/scaling/{h}/{split}/scores.jsonl"
            if not path.exists():
                continue
            for line in open(path):
                if not line.strip():
                    continue
                s = json.loads(line)
                if s["id"] not in recs:
                    continue
                r = dict(recs[s["id"]])
                r["harness"] = h
                r["correct"] = bool(s.get("correct"))
                rows.append(r)
    return rows


def family_size_table(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["harness"], r["family"], r["size_bin"])].append(r)
    out = []
    for (h, fam, sb), items in sorted(groups.items()):
        k = sum(x["correct"] for x in items)
        lo, hi = wilson_ci(k, len(items))
        out.append({
            "harness": h, "family": fam, "size_bin": sb, "n": len(items),
            "accuracy": k / len(items), "wilson_lo": lo, "wilson_hi": hi,
            "small_n_flag": int(len(items) < 10),
        })
    return out


def fit_logit(X, y, max_iter=100, ridge=1e-6):
    """Newton-Raphson logistic regression. Returns beta, standard errors."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    eta = X @ beta
    p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    W = p * (1 - p)
    H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
    cov = np.linalg.inv(H)
    return beta, np.sqrt(np.diag(cov))


def regression(rows, use_family=False):
    fams = sorted({r["family"] for r in rows} - {"er"}) if use_family else []
    tasks = sorted({r["task"] for r in rows} - {"shortest_path"}) if not use_family else []
    harnesses = sorted({r["harness"] for r in rows} - {"baseline_a"})
    names = ["intercept", "log_n", "density"]
    names += [f"family[{f}]" for f in fams]
    names += [f"task[{t}]" for t in tasks]
    names += [f"harness[{h}]" for h in harnesses]
    X = np.zeros((len(rows), len(names)))
    y = np.array([1.0 if r["correct"] else 0.0 for r in rows])
    for i, r in enumerate(rows):
        X[i, 0] = 1.0
        X[i, 1] = math.log(r["n"])
        # density is scaled by 10 so the coefficient is per 0.1 density
        X[i, 2] = r["density"] * 10.0
        j = 3
        for f in fams:
            if r["family"] == f:
                X[i, j] = 1.0
            j += 1
        for t in tasks:
            if r["task"] == t:
                X[i, j] = 1.0
            j += 1
        for h in harnesses:
            if r["harness"] == h:
                X[i, j] = 1.0
            j += 1
    beta, se = fit_logit(X, y)
    out = []
    for name, b, s in zip(names, beta, se):
        out.append({
            "term": name, "coef": b, "se": s,
            "ci_lo": b - 1.96 * s, "ci_hi": b + 1.96 * s,
            "odds_ratio": math.exp(b),
        })
    return out, len(rows)


def main():
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)

    fam = family_size_table(rows)
    with open(OUT / "family_x_size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fam[0]))
        w.writeheader()
        w.writerows(fam)

    # family is nested inside task by construction (tree==cycle_detection,
    # mixed==connected_components), so the two cannot be identified jointly.
    reg, n = regression(rows, use_family=False)
    with open(OUT / "regression.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reg[0]))
        w.writeheader()
        w.writerows(reg)
    reg_fam, _ = regression(rows, use_family=True)
    with open(OUT / "regression_family.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reg_fam[0]))
        w.writeheader()
        w.writerows(reg_fam)

    lines = ["# E8 characteristics", "",
             f"- Rows: {n} scored problems (A/B/v2 x 4 scaling splits).",
             "- `family_x_size.csv`: family x node-size bin per harness, Wilson CI, n, small-n flag.",
             "- `regression.csv`: logistic `correct ~ log(n) + 10*density + task + harness`.",
             "- `regression_family.csv`: same with family instead of task.",
             "- family and task are perfectly nested by construction: `tree` only occurs",
             "  with cycle_detection and `mixed` only with connected_components, so a model",
             "  containing both is not identified. Report one at a time.",
             "- Reference levels: task=shortest_path / family=er, harness=baseline_a.",
             "",
             "| term | coef | 95% CI | odds ratio |", "|---|---:|---|---:|"]
    for r in reg:
        lines.append(f"| {r['term']} | {r['coef']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] | {r['odds_ratio']:.3f} |")
    (OUT / "NOTES.md").write_text("\n".join(lines) + "\n")
    print(f"wrote family_x_size.csv ({len(fam)} rows) and regression.csv ({len(reg)} terms) from {n} problems")


if __name__ == "__main__":
    main()
