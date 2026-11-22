#!/usr/bin/env python3
"""Build the E7 disconnected additive set: 120 problems, seed 56243.

Sizes 400/1000/2000, components 5/10/20, four additive tasks, 10 problems per
task per size. Answers are merged from exact per-component solutions.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import networkx as nx  # noqa: E402
from graphharness.evaluate_answer import score_response  # noqa: E402
from graphharness.generators import base_graph, ensure_connected, graph_to_json  # noqa: E402
from graphharness.tasks import get_task  # noqa: E402
from graphharness.verify_data import truth_text  # noqa: E402

TASKS = ["max_clique", "max_independent_set", "min_vertex_cover", "connected_components"]
LAYOUT = [(400, 5, 80), (1000, 10, 100), (2000, 20, 100)]


def build_graph(rng, n_components, comp_size):
    comps = []
    for _ in range(n_components):
        H = ensure_connected(base_graph(comp_size, rng.choice(["er", "ba", "ws"]), rng), rng)
        comps.append(H)
    G = nx.Graph()
    offset = 0
    maps = []
    for H in comps:
        mapping = {v: v + offset for v in H.nodes}
        maps.append(mapping)
        G.add_nodes_from(mapping.values())
        G.add_edges_from((mapping[u], mapping[v]) for u, v in H.edges)
        offset += H.number_of_nodes()
    return G, comps, maps


def merge(task_name, sub_truth, maps):
    if task_name == "max_clique":
        best = None
        for t, m in zip(sub_truth, maps):
            clique = [m[v] for v in t["clique"]]
            if best is None or len(clique) > len(best["clique"]):
                best = {"size": len(clique), "clique": clique}
        return best
    if task_name in {"max_independent_set", "min_vertex_cover"}:
        nodes = sorted({m[v] for t, m in zip(sub_truth, maps) for v in t["nodes"]}, key=str)
        return {"size": len(nodes), "nodes": nodes}
    reps = sorted({m[t["representatives"][0]] for t, m in zip(sub_truth, maps)}, key=str)
    return {"count": len(sub_truth), "representatives": reps}


def main():
    rng = random.Random(56243)
    records = []
    for n_nodes, n_comp, comp_size in LAYOUT:
        for task_name in TASKS:
            task = get_task(task_name)
            for i in range(10):
                G, comps, maps = build_graph(rng, n_comp, comp_size)
                sub_truth = [task.solve(H, {}) for H in comps]
                answer = merge(task_name, sub_truth, maps)
                rec = {
                    "id": f"{task_name}/disc{n_nodes}/{i:03d}", "source": "synthetic",
                    "task": task_name, "split": "disconnected120",
                    "graph": graph_to_json(G), "params": {},
                    "question": task.question({}), "answer": answer,
                    "meta": {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                             "n_components": n_comp, "family": "disconnected",
                             "size_group": n_nodes},
                }
                records.append(rec)
    bad = [r["id"] for r in records if not score_response(r, truth_text(r))["correct"]]
    out = ROOT / "data_final/disconnected120.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (ROOT / "data_final/manifest_disconnected120.json").write_text(json.dumps(
        {"seed": 56243, "layout": LAYOUT, "count": len(records), "bad_truth": len(bad)}, indent=2) + "\n")
    print(f"wrote {out} ({len(records)} records, bad truth {len(bad)})")
    for r in records[:1]:
        print(r["id"], r["meta"], r["answer"])


if __name__ == "__main__":
    main()
