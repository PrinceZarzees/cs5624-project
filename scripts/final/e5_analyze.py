#!/usr/bin/env python3
"""E5: compare proxies against val and proxy_large oracle accuracy.

Reads experiments/final/E5_proxy/{pool.csv,runs/,judge_4b.json,judge_7b.json}.
No model calls.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphharness.agent import Harness  # noqa: E402
from graphharness.rubric_proxy import pairwise_accuracy, spearman, static_features  # noqa: E402

E5 = ROOT / "experiments/final/E5_proxy"


def kendall(x, y):
    c = d = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            v = (x[i] - x[j]) * (y[i] - y[j])
            c += v > 0
            d += v < 0
    return (c - d) / (c + d) if c + d else 0.0


def rollout(scores_path: Path, k: int):
    if not scores_path.exists():
        return None
    rows = [json.loads(l) for l in scores_path.read_text().splitlines() if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(bool(r.get("correct")))
    vals = [v for xs in by.values() for v in xs[:k]]
    return sum(vals) / len(vals) if vals else None


def oracle(name: str):
    p = E5 / "runs" / name / "val_k10/scores.jsonl"
    q = E5 / "runs" / name / "proxy_large_k10/scores.jsonl"
    if not (p.exists() and q.exists()):
        return None, None
    def acc(path):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        return sum(bool(r.get("correct")) for r in rows) / len(rows) if rows else None
    return acc(p), acc(q)


def ridge_loo(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1
    Xs = (X - mu) / sd
    pred = []
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        w = np.linalg.solve(Xs[m].T @ Xs[m] + lam * np.eye(Xs.shape[1]), Xs[m].T @ y[m])
        pred.append(float(Xs[i] @ w))
    return pred


def ridge_cv(X, y, folds=5, lam=1.0, seed=0):
    idx = list(range(len(y))); rng = random.Random(seed); rng.shuffle(idx)
    pred = [0.0] * len(y)
    for f in range(folds):
        test = idx[f::folds]
        tr = [i for i in range(len(y)) if i not in test]
        mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
        Xs = (X - mu) / sd
        w = np.linalg.solve(Xs[tr].T @ Xs[tr] + lam * np.eye(X.shape[1]), Xs[tr].T @ y[tr])
        for i in test:
            pred[i] = float(Xs[i] @ w)
    return pred


def boot_rho(p, y, B=5000, seed=0):
    rng = random.Random(seed); vals = []
    for _ in range(B):
        idx = [rng.randrange(len(y)) for _ in range(len(y))]
        vals.append(spearman([p[i] for i in idx], [y[i] for i in idx]))
    vals.sort()
    return vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def metrics(proxy, truth):
    ranked = sorted(range(len(truth)), key=lambda i: proxy[i], reverse=True)
    best = max(range(len(truth)), key=lambda i: truth[i])
    top3 = ranked[:3]
    return {
        "spearman": spearman(proxy, truth),
        "kendall": kendall(proxy, truth),
        "pairwise": pairwise_accuracy(proxy, truth),
        "top1_hit": int(ranked[0] == best),
        "top3_hit": int(best in top3),
        "regret": truth[best] - truth[ranked[0]],
    }


def main():
    pool = list(csv.DictReader(open(E5 / "pool.csv")))
    judges = {}
    for tag in ("4b", "7b"):
        p = E5 / f"judge_{tag}.json"
        if p.exists():
            judges[tag] = {r["id"]: r.get("proxy") for r in json.loads(p.read_text())}
            judges[tag + "_scores"] = {r["id"]: r.get("scores") for r in json.loads(p.read_text())}
    rows = []
    for r in pool:
        val, large = oracle(r["id"])
        if val is None:
            continue
        h = Harness.load(E5 / r["path"])
        st = static_features(h)
        rows.append({
            "id": r["id"], "val": val, "large": large,
            "P1": judges.get("4b", {}).get(r["id"]),
            "P2": judges.get("7b", {}).get(r["id"]),
            "static": [float(v) for _, v in sorted(st.items())],
            "P5_1": rollout(E5 / "runs" / r["id"] / "val_k10/scores.jsonl", 1),
            "P5_3": rollout(E5 / "runs" / r["id"] / "val_k10/scores.jsonl", 3),
            "P5_5": rollout(E5 / "runs" / r["id"] / "val_k10/scores.jsonl", 5),
            "P6_1": rollout(E5 / "runs" / r["id"] / "proxy_medium_k3/scores.jsonl", 1),
            "P6_3": rollout(E5 / "runs" / r["id"] / "proxy_medium_k3/scores.jsonl", 3),
            "rubric": judges.get("7b_scores", {}).get(r["id"]),
        })
    if not rows:
        print("no labelled candidates yet")
        return

    X = np.array([r["static"] for r in rows], float)
    yv = np.array([r["val"] for r in rows], float)
    for r, p in zip(rows, ridge_loo(X, yv)):
        r["P3"] = p
    if all(r.get("rubric") for r in rows):
        keys = sorted({k for r in rows for k in r["rubric"]})
        XR = np.array([[float(r["rubric"].get(k, 0)) for k in keys] for r in rows])
        for r, p in zip(rows, ridge_cv(XR, yv)):
            r["P4"] = p
    XH = np.array([r["static"] + [float(r["P5_3"] or 0)] for r in rows])
    for r, p in zip(rows, ridge_loo(XH, yv)):
        r["P7"] = p

    proxies = ["P1", "P2", "P3", "P4", "P5_1", "P5_3", "P5_5", "P6_1", "P6_3", "P7"]
    with open(E5 / "proxy_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "val", "large"] + proxies)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    summary = []
    for pname in proxies:
        usable = [r for r in rows if r.get(pname) is not None]
        if len(usable) < 5:
            continue
        for target in ("val", "large"):
            truth = [float(r[target]) for r in usable]
            proxy = [float(r[pname]) for r in usable]
            lo, hi = boot_rho(proxy, truth)
            m = metrics(proxy, truth)
            summary.append({"proxy": pname, "target": target, "n": len(usable),
                            "spearman": m["spearman"], "rho_lo": lo, "rho_hi": hi,
                            "kendall": m["kendall"], "pairwise": m["pairwise"],
                            "top1_hit": m["top1_hit"], "top3_hit": m["top3_hit"],
                            "regret": m["regret"]})
    with open(E5 / "summary_full.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    print(f"analysed {len(rows)} candidates")
    for s in summary:
        print(f"{s['proxy']:5s} {s['target']:5s} n={s['n']:2d} rho={s['spearman']:+.3f} "
              f"[{s['rho_lo']:+.2f},{s['rho_hi']:+.2f}] pw={s['pairwise']:.2f} top3={s['top3_hit']}")


if __name__ == "__main__":
    main()
