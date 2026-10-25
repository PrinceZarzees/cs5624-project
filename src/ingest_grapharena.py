"""
ingest_grapharena.py
====================
Ingestion pipeline for the GraphArena benchmark (Tang et al., ICLR 2025).
Source: Official GraphArena dataset release (`squareRoot3/GraphArena`).

Extracts:
  - Tasks: Distance (shortest path), Connected (connected components)
  - Difficulties: easy, hard
  - Formats into standard GraphInstance and TaskInstance JSONL records.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from .graph_generator import GraphInstance
from .tasks import TaskInstance


def parse_grapharena_item(
    item: Dict[str, Any],
    task_name: str,
    difficulty: str,
    index: int,
) -> Tuple[GraphInstance, TaskInstance]:
    """Parse a single GraphArena problem entry into GraphInstance + TaskInstance."""
    orig_g = item["graph"]
    orig_nodes = list(orig_g.nodes())
    node_to_idx = {node: i for i, node in enumerate(orig_nodes)}
    idx_to_name = {
        i: orig_g.nodes[node].get("name", str(node))
        for i, node in enumerate(orig_nodes)
    }

    # Relabel edges to contiguous 0-indexed integers
    edges = [
        [node_to_idx[u], node_to_idx[v]]
        for u, v in orig_g.edges()
    ]
    nnodes = len(orig_nodes)
    nedges = len(edges)

    graph_id = f"grapharena_{task_name.lower()}_{difficulty}_{index:04d}"
    graph_inst = GraphInstance(
        graph_id=graph_id,
        model=f"grapharena_{task_name.lower()}_{difficulty}",
        num_nodes=nnodes,
        num_edges=nedges,
        nodes=list(range(nnodes)),
        edges=edges,
        directed=False,
        scale="small" if nnodes <= 50 else ("medium" if nnodes <= 300 else "large"),
    )

    # Canonical task type mapping
    task_map = {
        "Distance": "shortest_path",
        "Connected": "connected_components",
        "Diameter": "diameter",
        "MCP": "max_clique",
    }
    canonical_task_type = task_map.get(task_name, task_name.lower())
    task_id = f"{graph_id}_{canonical_task_type}"

    # Path or representative nodes mapped to integer IDs
    orig_path = item.get("path")
    clean_path = None
    if orig_path:
        clean_path = [node_to_idx[n] for n in orig_path if n in node_to_idx]

    task_inst = TaskInstance(
        task_id=task_id,
        graph_id=graph_id,
        task_type=canonical_task_type,
        question=item["problem_text"],
        answer=int(item["exact_answer"]),
        metadata={
            "source_benchmark": "GraphArena",
            "original_task": task_name,
            "difficulty": difficulty,
            "path": clean_path,
            "node_names": idx_to_name,
        },
    )
    return graph_inst, task_inst


def ingest_grapharena(
    raw_dir: str = "data/raw_grapharena",
    output_dir: str = "data",
) -> Dict[str, Any]:
    """
    Ingest extracted GraphArena PKL files into JSONL dataset splits.

    Parameters
    ----------
    raw_dir : str
        Directory where PKL files reside.
    output_dir : str
        Directory to save JSONL output files.

    Returns
    -------
    summary dict
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}
    all_graphs: Dict[str, GraphInstance] = {}

    targets = [
        ("Distance", "easy", "Distance_easy.pkl"),
        ("Distance", "hard", "Distance_hard.pkl"),
        ("Connected", "easy", "Connected_easy.pkl"),
        ("Connected", "hard", "Connected_hard.pkl"),
    ]

    for task_name, difficulty, fname in targets:
        fpath = raw_path / fname
        if not fpath.exists():
            print(f"Warning: file {fpath} not found, skipping...")
            continue

        print(f"Ingesting GraphArena: {fname} ...")
        with open(fpath, "rb") as f:
            data = pickle.load(f)

        tasks: List[TaskInstance] = []
        for idx, item in enumerate(data):
            g_inst, t_inst = parse_grapharena_item(item, task_name, difficulty, idx)
            all_graphs[g_inst.graph_id] = g_inst
            tasks.append(t_inst)

        split_task_file = out_path / f"tasks_grapharena_{task_name.lower()}_{difficulty}.jsonl"
        with open(split_task_file, "w") as f:
            for t in tasks:
                f.write(json.dumps(t.to_dict()) + "\n")

        print(f"  Parsed {len(tasks)} tasks -> {split_task_file}")
        summary[f"grapharena_{task_name.lower()}_{difficulty}"] = {
            "n_tasks": len(tasks),
            "file": str(split_task_file),
        }

    # Save combined tasks by difficulty
    for diff in ["easy", "hard"]:
        diff_tasks = []
        for task_name in ["Distance", "Connected"]:
            split_task_file = out_path / f"tasks_grapharena_{task_name.lower()}_{diff}.jsonl"
            if split_task_file.exists():
                with open(split_task_file) as f:
                    for line in f:
                        diff_tasks.append(json.loads(line.strip()))

        if diff_tasks:
            comb_file = out_path / f"tasks_grapharena_{diff}.jsonl"
            with open(comb_file, "w") as f:
                for t in diff_tasks:
                    f.write(json.dumps(t) + "\n")
            summary[f"grapharena_{diff}_combined"] = {
                "n_tasks": len(diff_tasks),
                "file": str(comb_file),
            }

    # Save all unique graphs
    graph_file = out_path / "graphs_grapharena.jsonl"
    with open(graph_file, "w") as f:
        for g in all_graphs.values():
            f.write(json.dumps(g.to_dict()) + "\n")
    print(f"Saved {len(all_graphs)} graphs to {graph_file}")
    summary["grapharena_graphs"] = {
        "n_graphs": len(all_graphs),
        "file": str(graph_file),
    }

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest GraphArena benchmark dataset.")
    parser.add_argument("--raw-dir", type=str, default="data/raw_grapharena")
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    print("=" * 60)
    print("CS 5624 — Ingesting GraphArena Benchmark Dataset")
    print("=" * 60)
    summary = ingest_grapharena(raw_dir=args.raw_dir, output_dir=args.output_dir)
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
