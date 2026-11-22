#!/usr/bin/env python3
"""E5: same-candidate proxy transfer, val -> test_large.

15 candidates: full v2 plus the 14 one-rule-removed variants. Both labels
already exist in the frozen ablation runs, so no model calls are made.

Reports the learned static proxy (leave-one-out) against val and against
test_large, plus Spearman/Kendall/pairwise/top-3 for each target.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphharness.agent import BASELINE_B_V2, get_harness  # noqa: E402
from graphharness.rubric_proxy import pairwise_accuracy, spearman, static_features  # noqa: E402

OUT = ROOT / "experiments/final/E5_proxy"


def kendall(x, y):
    n = len(x)
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = (x[i] - x[j]) * (y[i] - y[j])
            if a > 0:
                conc += 1
            elif a < 0:
                disc += 1
    return (conc - disc) / (conc + disc) if conc + disc else 0.0


def loo_predictions(X, y):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Xs = (X - mu) / sd
    pred = []
    for i in range(len(y)):
        mask = np.arange(len(y)) != i
        w = np.linalg.solve(Xs[mask].T @ Xs[mask] + np.eye(Xs.shape[1]), Xs[mask].T @ y[mask])
        pred.append(float(Xs[i] @ w))
    return pred


def main():
    val = json.loads((ROOT / "experiments/qwen3_4b_ablation_val/ablation/ablation_summary.json").read_text())
    large = json.loads((ROOT / "experiments/qwen3_4b_ablation_full/ablation/ablation_summary.json").read_text())
    large_by = {r["variant"]: r["accuracy"] for r in large["rows"]}
    rows = []
    for r in val["rows"]:
        h = BASELINE_B_V2 if r["rule_index"] is None else BASELINE_B_V2.without_rule(r["rule_index"])
        text = (h.system_prompt + "\n" + "\n".join(h.rules)).lower()
        feats = static_features(h) if static_features is not None else None
        if isinstance(feats, dict):
            feats = [float(v) for _, v in sorted(feats.items())]
        if feats is None:
            feats = [len(h.system_message()), len(h.rules),
                     sum(k in text for k in ["verify", "check", "print", "networkx", "diameter",
                                             "clique", "component", "exact", "heuristic"]),
                     int("print(" in text), int("diameter" in text), int("clique" in text),
                     int("component" in text), int("verify" in text or "check" in text),
                     int(h.allow_tools), int(h.graph_view == "edge_list"),
                     int(bool(re.search(r"\bnode\s+\d+\b", text)))]
        rows.append({"id": r["variant"], "val_accuracy": r["accuracy"],
                     "large_accuracy": large_by[r["variant"]], "features": feats})
    X = np.array([r["features"] for r in rows], float)
    proxy = loo_predictions(X, np.array([r["val_accuracy"] for r in rows]))
    for r, p in zip(rows, proxy):
        r["proxy"] = p

    rng = random.Random(0)
    def boot_rho(p, y, B=10000):
        vals = []
        for _ in range(B):
            idx = [rng.randrange(len(y)) for _ in range(len(y))]
            vals.append(spearman([p[i] for i in idx], [y[i] for i in idx]))
        vals.sort()
        return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]

    summary = []
    for target in ["val_accuracy", "large_accuracy"]:
        y = [r[target] for r in rows]
        lo, hi = boot_rho(proxy, y)
        ranked = sorted(rows, key=lambda r: r["proxy"], reverse=True)
        best = max(rows, key=lambda r: r[target])
        summary.append({
            "proxy": "learned_static_LOO", "target": target.replace("_accuracy", ""), "n": len(rows),
            "spearman": spearman(proxy, y), "rho_lo": lo, "rho_hi": hi,
            "kendall": kendall(proxy, y),
            "pairwise": pairwise_accuracy(proxy, y),
            "top1_hit": int(ranked[0]["id"] == best["id"]),
            "top3_hit": int(best["id"] in [r["id"] for r in ranked[:3]]),
            "best_actual": best["id"], "best_proxy": ranked[0]["id"],
        })

    with open(OUT / "labels_same_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "val_accuracy", "large_accuracy", "proxy"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    with open(OUT / "summary_same_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    # merge the earlier proxy pilots (18 optimizer candidates, val target)
    extra = []
    learned = ROOT / "experiments/review_rubric_proxy/learned_proxy_results.json"
    if learned.exists():
        d = json.loads(learned.read_text())
        extra.append({"proxy": "learned_static_LOO", "target": "val", "n": d["n"],
                      "spearman": d["spearman_loo"], "rho_lo": "", "rho_hi": "",
                      "kendall": "", "pairwise": d["pairwise_loo"],
                      "top1_hit": "", "top3_hit": "",
                      "best_actual": "", "best_proxy": "",
                      "source": "earlier-18"})
    cheap = ROOT / "experiments/review_rubric_proxy/cheap_proxy_results.json"
    if cheap.exists():
        d = json.loads(cheap.read_text())
        extra.append({"proxy": "cheap_rollout_k1", "target": "val", "n": d["n"],
                      "spearman": d["spearman"], "rho_lo": "", "rho_hi": "",
                      "kendall": "", "pairwise": d["pairwise"],
                      "top1_hit": "", "top3_hit": int(d["top3_recall"]),
                      "best_actual": "", "best_proxy": "",
                      "source": "earlier-18, k=1"})
    for r in summary:
        r["source"] = "same-15"
    allrows = extra + summary
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
