"""Build a disconnected additive-task set for the router/decomposition experiment."""
from __future__ import annotations
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import networkx as nx
from graphharness.generators import base_graph, ensure_connected, graph_to_json
from graphharness.tasks import get_task

def main():
    rng = random.Random(99)
    out = Path("data/disconnected_additive.jsonl")
    out.parent.mkdir(exist_ok=True)
    tasks = ["max_clique", "max_independent_set", "min_vertex_cover", "connected_components"]
    records = []
    for task_name in tasks:
        task = get_task(task_name)
        for i in range(10):
            comps = []
            for c in range(5):
                H = ensure_connected(base_graph(80, rng.choice(["er", "ba", "ws"]), rng), rng)
                comps.append(H)
            # Relabel components to globally unique ids.
            offset = 0
            G = nx.Graph()
            for H in comps:
                mapping = {v: v + offset for v in H.nodes}
                G.add_nodes_from(mapping.values())
                G.add_edges_from((mapping[u], mapping[v]) for u, v in H.edges)
                offset += H.number_of_nodes()
            sub_truth = [task.solve(H, {}) for H in comps]
            if task_name == "max_clique":
                best = max(sub_truth, key=lambda t: t["size"])
                answer = best
            elif task_name == "max_independent_set":
                nodes = sorted({v for t in sub_truth for v in t["nodes"]}, key=str)
                answer = {"size": len(nodes), "nodes": nodes}
            elif task_name == "min_vertex_cover":
                nodes = sorted({v for t in sub_truth for v in t["nodes"]}, key=str)
                answer = {"size": len(nodes), "nodes": nodes}
            else:
                reps = sorted({t["representatives"][0] for t in sub_truth}, key=str)
                answer = {"count": len(comps), "representatives": reps}
            records.append({
                "id": f"{task_name}/disconnected/{i:03d}", "source": "synthetic",
                "task": task_name, "split": "disconnected", "graph": graph_to_json(G),
                "params": {}, "question": task.question({}), "answer": answer,
                "meta": {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                         "n_components": 5, "family": "disconnected"},
            })
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print("wrote", out, len(records))

if __name__ == "__main__":
    main()
