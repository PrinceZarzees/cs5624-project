#!/usr/bin/env python3
"""Learned harness router and exact additive decomposition helpers (E7).

Features are cheap graph statistics + task. The label is the best library
harness on a validation problem. The model is one-vs-rest logistic regression
fitted with Newton iterations (sklearn is not installed).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphharness.data_loader import load_split, to_graph  # noqa: E402

TASKS = ["shortest_path", "connected_components", "cycle_detection", "diameter", "max_clique"]
FEATURES = ["log_n", "density", "avg_degree", "max_degree", "clustering",
            "n_components", "is_forest"] + [f"task[{t}]" for t in TASKS]


def features(record: dict) -> list[float]:
    g = to_graph(record)
    n = g.number_of_nodes()
    m = g.number_of_edges()
    deg = [d for _, d in g.degree()]
    dens = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
    try:
        clus = float(np.mean(list(__import__("networkx").clustering(g).values())))
    except Exception:
        clus = 0.0
    import networkx as nx
    ncomp = nx.number_connected_components(g)
    forest = int(nx.is_forest(g))
    vec = [math.log(max(2, n)), dens, (sum(deg) / n if n else 0.0),
           max(deg) if deg else 0.0, clus, ncomp, forest]
    vec += [1.0 if record["task"] == t else 0.0 for t in TASKS]
    return vec


def fit_logit(X, y, l2=1.0, iters=200, sample_weight=None):
    beta = np.zeros(X.shape[1])
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        W = p * (1 - p) * sample_weight
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        g = X.T @ ((y - p) * sample_weight) - l2 * beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


class LearnedRouter:
    def __init__(self, classes, betas, mean, std):
        self.classes = list(classes)
        self.betas = np.asarray(betas, float)
        self.mean = np.asarray(mean, float)
        self.std = np.asarray(std, float)

    @classmethod
    def train(cls, records, labels, classes):
        X = np.array([features(r) for r in records], float)
        mean, std = X.mean(0), X.std(0)
        std[std == 0] = 1.0
        Xs = (X - mean) / std
        counts = {c: max(1, labels.count(c)) for c in classes}
        K = len(classes); N = len(labels)
        sw = np.array([N / (K * counts[y]) for y in labels])
        betas = [fit_logit(Xs, np.array([1.0 if y == c else 0.0 for y in labels]),
                           sample_weight=sw)
                 for c in classes]
        return cls(classes, betas, mean, std)

    def predict(self, record):
        x = (np.array(features(record)) - self.mean) / self.std
        scores = [float(x @ b) for b in self.betas]
        return self.classes[int(np.argmax(scores))]

    def choose(self, record):
        return self.predict(record)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "classes": self.classes, "betas": self.betas.tolist(),
            "mean": self.mean.tolist(), "std": self.std.tolist(),
            "features": FEATURES}) + "\n")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["classes"], d["betas"], d["mean"], d["std"])


def exact_additive(task_name, H):
    """Exact answer on one connected component (NetworkX branch-and-bound)."""
    import networkx as nx
    if task_name == "connected_components":
        return {"representatives": [next(iter(H.nodes))]}
    if task_name == "max_clique":
        clique, _ = nx.max_weight_clique(H, weight=None)
        clique = sorted(clique, key=str)
        return {"size": len(clique), "clique": clique}
    if task_name in {"max_independent_set", "min_vertex_cover"}:
        clique, _ = nx.max_weight_clique(nx.complement(H), weight=None)
        indep = sorted(clique, key=str)
        if task_name == "max_independent_set":
            return {"size": len(indep), "nodes": indep}
        cover = sorted(set(H.nodes) - set(indep), key=str)
        return {"size": len(cover), "nodes": cover}
    raise ValueError(task_name)


def load_matrix(scores_glob: str, split: str):
    """Per-problem correctness matrix for every library harness."""
    records = {r["id"]: r for r in load_split(split, str(ROOT / "data"))}
    tables = {}
    for path in glob.glob(scores_glob):
        name = Path(path).parts[-2]
        tables[name] = {r["id"]: bool(r.get("correct")) for r in map(json.loads, open(path)) if r}
    priority = ["baseline_b_v2", "tree", "sparse", "dense", "large"]
    names = [n for n in priority if n in tables] + [n for n in tables if n not in priority]
    ids = [pid for pid in records if all(pid in tables[n] for n in names)]
    recs = [records[pid] for pid in ids]
    Y = np.array([[1.0 if tables[n][pid] else 0.0 for n in names] for pid in ids])
    return recs, Y, names


def train_accuracy(recs, Y, names):
    X = np.array([features(r) for r in recs], float)
    mean, std = X.mean(0), X.std(0)
    std[std == 0] = 1.0
    Xs = (X - mean) / std
    betas = [fit_logit(Xs, Y[:, k], l2=0.5) for k in range(len(names))]
    return LearnedRouter(names, betas, mean, std)


def load_labels(scores_glob: str, split: str):
    """Join per-harness scores and pick the best correct harness per problem."""
    records = {r["id"]: r for r in load_split(split, str(ROOT / "data"))}
    tables = {}
    for path in glob.glob(scores_glob):
        name = Path(path).parts[-2]
        tables[name] = {r["id"]: bool(r.get("correct")) for r in map(json.loads, open(path)) if r}
    priority = ["baseline_b_v2", "tree", "sparse", "dense", "large"]
    names = [n for n in priority if n in tables] + [n for n in tables if n not in priority]
    recs, labels = [], []
    for pid, rec in records.items():
        correct = [n for n in names if tables[n].get(pid)]
        if not correct:
            continue
        recs.append(rec)
        labels.append(correct[0])
    return recs, labels, names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="glob such as 'experiments/final/E7_router/val/*/scores.jsonl'")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", default="experiments/final/E7_router/router.json")
    args = ap.parse_args(argv)
    recs, Y, names = load_matrix(args.scores, args.split)
    router = train_accuracy(recs, Y, names)
    # 5-fold CV: downstream accuracy of the picked harness
    idx = np.arange(len(recs)); rng = np.random.default_rng(0); rng.shuffle(idx)
    acc, oracle = [], []
    for f in range(5):
        test = idx[f::5]
        tr = [i for i in range(len(recs)) if i not in set(test.tolist())]
        r = train_accuracy([recs[i] for i in tr], Y[tr], names)
        for i in test:
            pick = names.index(r.predict(recs[i]))
            acc.append(float(Y[i, pick])); oracle.append(float(Y[i].max()))
    router.save(args.out)
    best = [names[int(y.argmax())] for y in Y]
    meta = {"n": len(recs), "classes": names,
            "class_counts": {c: best.count(c) for c in names},
            "cv_router_accuracy": sum(acc) / len(acc),
            "cv_oracle_any_accuracy": sum(oracle) / len(oracle),
            "features": FEATURES}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
