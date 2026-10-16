"""Convert GraphArena problems into our common JSONL format.

GraphArena's processed data is a set of pickle files (``Distance_easy.pkl``,
``MCP_hard.pkl``, ...) inside ``dataset.zip`` from the Google Drive link in
https://github.com/squareRoot3/GraphArena. Download and unzip it, then:

    python -m graphharness.grapharena_adapter --dataset-dir path/to/dataset --out data

This writes ``data/ga_easy.jsonl`` and ``data/ga_hard.jsonl``.

Only the GraphArena tasks that match our tasks are converted:

    Distance  -> shortest_path
    Connected -> connected_components
    Diameter  -> diameter
    MCP       -> max_clique

Nodes are named by GraphArena's ``name`` attribute (e.g. real author names).
Ground truth is recomputed with our own solvers so it is checked exactly the
same way as synthetic data; GraphArena's own answer is kept in ``meta``.

Note: GraphArena's "hard" graphs have at most 50 nodes, so they are still
"small" for this project. Larger test graphs come from ``build_data``.

Security note: pickle files can run code when loaded. Only load the official
GraphArena files.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import networkx as nx

from .data_loader import DEFAULT_DATA_DIR, save_jsonl
from .generators import graph_to_json
from .tasks import as_json_value, get_task

GA_TO_OURS = {
    "Distance": "shortest_path",
    "Connected": "connected_components",
    "Diameter": "diameter",
    "MCP": "max_clique",
}


def convert_problem(problem: dict, ga_task: str, difficulty: str, index: int) -> dict | None:
    """Convert one GraphArena problem dict. Returns None if it cannot be used."""
    task_name = GA_TO_OURS[ga_task]
    task = get_task(task_name)
    G0: nx.Graph = problem["graph"]

    names = {v: str(d.get("name", v)) for v, d in G0.nodes(data=True)}
    if len(set(names.values())) != len(names):
        return None  # two nodes share a name; answers would be ambiguous

    G = nx.Graph()
    G.add_nodes_from(names.values())
    G.add_edges_from((names[u], names[v]) for u, v in G0.edges())

    if task_name == "shortest_path":
        params = {"source": names[problem["source"]], "target": names[problem["target"]]}
    else:
        params = {}
    if task_name == "diameter" and not nx.is_connected(G):
        return None

    answer = task.solve(G, params)
    split = f"ga_{difficulty}"
    return {
        "id": f"{task_name}/{split}/{index:05d}",
        "source": "grapharena",
        "task": task_name,
        "split": split,
        "graph": graph_to_json(G),
        "params": as_json_value(params),
        "question": task.question(params),
        "answer": as_json_value(answer),
        "meta": {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "task_kind": task.kind,
            "grapharena_task": ga_task,
            "grapharena_id": problem.get("id", index),
            "grapharena_exact_answer": as_json_value(problem.get("exact_answer")),
            "original_prompt": problem.get("problem_text"),
        },
    }


def convert_dataset(dataset_dir: Path, difficulty: str, limit: int | None = None) -> list[dict]:
    records = []
    for ga_task in GA_TO_OURS:
        path = dataset_dir / f"{ga_task}_{difficulty}.pkl"
        if not path.exists():
            print(f"  skip {path.name} (not found)")
            continue
        with open(path, "rb") as f:
            problems = pickle.load(f)
        kept = 0
        for i, p in enumerate(problems):
            if limit is not None and kept >= limit:
                break
            rec = convert_problem(p, ga_task, difficulty, i)
            if rec is not None:
                records.append(rec)
                kept += 1
        print(f"  {ga_task:<10} -> {GA_TO_OURS[ga_task]:<22} {kept} problems")
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True, help="folder with GraphArena *.pkl files")
    ap.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--limit", type=int, help="max problems per task (default: all)")
    args = ap.parse_args(argv)

    for difficulty in ("easy", "hard"):
        print(f"GraphArena {difficulty}:")
        records = convert_dataset(Path(args.dataset_dir), difficulty, args.limit)
        if records:
            path = Path(args.out) / f"ga_{difficulty}.jsonl"
            save_jsonl(records, path)
            print(f"  wrote {len(records)} problems -> {path}")


if __name__ == "__main__":
    main()
