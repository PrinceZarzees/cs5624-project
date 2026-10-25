"""
data_loader.py
==============
Unified data loading and saving utilities for the CS 5624 project.

Supports:
  - Loading/saving graph instances from/to JSONL files.
  - Loading/saving task instances from/to JSONL files.
  - Generating complete task datasets from graph datasets.
  - Providing graph representations suitable for LLM prompts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .graph_generator import GraphInstance, generate_full_dataset, split_small_graphs
from .tasks import (
    TaskInstance, TASK_REGISTRY, generate_task_instance, get_task_names,
)


# ---------------------------------------------------------------------------
# JSONL I/O for GraphInstance
# ---------------------------------------------------------------------------

def save_graphs_jsonl(graphs: List[GraphInstance], filepath: str) -> None:
    """Save a list of GraphInstance objects to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for g in graphs:
            f.write(json.dumps(g.to_dict()) + "\n")


def load_graphs_jsonl(filepath: str) -> List[GraphInstance]:
    """Load GraphInstance objects from a JSONL file."""
    graphs = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                graphs.append(GraphInstance.from_dict(json.loads(line)))
    return graphs


# ---------------------------------------------------------------------------
# JSONL I/O for TaskInstance
# ---------------------------------------------------------------------------

def save_tasks_jsonl(tasks: List[TaskInstance], filepath: str) -> None:
    """Save a list of TaskInstance objects to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_tasks_jsonl(filepath: str) -> List[TaskInstance]:
    """Load TaskInstance objects from a JSONL file."""
    tasks = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(TaskInstance.from_dict(json.loads(line)))
    return tasks


# ---------------------------------------------------------------------------
# Graph → Prompt Representation
# ---------------------------------------------------------------------------

def graph_to_adjacency_text(graph: GraphInstance, max_edges: int = 500) -> str:
    """
    Convert a graph to a textual adjacency list representation
    suitable for inclusion in an LLM prompt.

    For large graphs, truncates and adds a note.
    """
    lines = [f"Graph with {graph.num_nodes} nodes and {graph.num_edges} edges."]
    lines.append(f"Nodes: {graph.nodes}")

    if len(graph.edges) <= max_edges:
        lines.append("Edges (undirected):")
        for u, v in graph.edges:
            lines.append(f"  {u} -- {v}")
    else:
        lines.append(f"Edges (showing first {max_edges} of {len(graph.edges)}):")
        for u, v in graph.edges[:max_edges]:
            lines.append(f"  {u} -- {v}")
        lines.append(f"  ... ({len(graph.edges) - max_edges} more edges)")

    return "\n".join(lines)


def graph_to_edge_list_text(graph: GraphInstance) -> str:
    """Convert a graph to a compact edge list string."""
    edges_str = ", ".join(f"({u},{v})" for u, v in graph.edges)
    return (
        f"Nodes: {graph.nodes}\n"
        f"Edges: [{edges_str}]"
    )


def graph_to_summary_stats(graph: GraphInstance) -> str:
    """
    Convert a graph to summary statistics (useful for very large graphs
    where full adjacency would overflow context).
    """
    G = graph.to_networkx()
    degrees = dict(G.degree())
    deg_vals = list(degrees.values())

    lines = [
        f"Graph Summary:",
        f"  Nodes: {graph.num_nodes}",
        f"  Edges: {graph.num_edges}",
        f"  Density: {nx.density(G):.4f}",
        f"  Average degree: {sum(deg_vals) / len(deg_vals):.2f}" if deg_vals else "  Average degree: 0",
        f"  Min degree: {min(deg_vals)}" if deg_vals else "  Min degree: 0",
        f"  Max degree: {max(deg_vals)}" if deg_vals else "  Max degree: 0",
        f"  Connected components: {nx.number_connected_components(G)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Complete dataset generation pipeline
# ---------------------------------------------------------------------------

def generate_task_dataset(
    graphs: List[GraphInstance],
    task_types: Optional[List[str]] = None,
    seed: int = 42,
) -> List[TaskInstance]:
    """
    Generate task instances for all graphs × all task types.

    Parameters
    ----------
    graphs : list of GraphInstance
        The input graphs.
    task_types : list of str, optional
        Task types to generate. Default: all polynomial tasks + max_clique.
        Hamiltonian path is skipped for large graphs by default.
    seed : int
        Random seed.

    Returns
    -------
    list of TaskInstance
    """
    if task_types is None:
        task_types = ["shortest_path", "connected_components", "cycle_detection",
                      "max_clique"]

    tasks = []
    for graph in graphs:
        G = graph.to_networkx()
        for task_type in task_types:
            # Skip Hamiltonian path for graphs > 20 nodes (too expensive exact)
            if task_type == "hamiltonian_path" and graph.num_nodes > 20:
                continue

            task = generate_task_instance(
                graph_id=graph.graph_id,
                G=G,
                task_type=task_type,
                seed=seed,
            )
            tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Benchmark Data Loaders (GraphQA & GraphArena)
# ---------------------------------------------------------------------------

def load_graphqa(
    data_dir: str = "data",
    split: str = "test",
) -> List[TaskInstance]:
    """
    Load GraphQA tasks for a given split ('train', 'validation', 'test').
    """
    path = Path(data_dir) / f"tasks_graphqa_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"GraphQA split file not found: {path}")
    return load_tasks_jsonl(str(path))


def load_grapharena(
    data_dir: str = "data",
    difficulty: str = "easy",
    task: Optional[str] = None,
) -> List[TaskInstance]:
    """
    Load GraphArena tasks.

    Parameters
    ----------
    data_dir : str
        Data directory path.
    difficulty : str
        'easy' or 'hard'.
    task : str, optional
        'distance', 'connected', or None for combined.
    """
    path = Path(data_dir)
    if task:
        fpath = path / f"tasks_grapharena_{task.lower()}_{difficulty}.jsonl"
    else:
        fpath = path / f"tasks_grapharena_{difficulty}.jsonl"

    if not fpath.exists():
        raise FileNotFoundError(f"GraphArena dataset file not found: {fpath}")
    return load_tasks_jsonl(str(fpath))


def get_full_dataset_summary(data_dir: str = "data") -> Dict[str, Any]:
    """
    Build a comprehensive summary across all datasets in the repository:
    Synthetic (Small, Medium, Large, Stress), GraphQA, and GraphArena.
    """
    d = Path(data_dir)
    summary: Dict[str, Any] = {
        "synthetic": {},
        "graphqa": {},
        "grapharena": {},
    }

    # Synthetic splits
    for split in ["small_train", "small_val", "medium", "large", "stress"]:
        task_f = d / f"tasks_{split}.jsonl"
        graph_f = d / f"graphs_{split}.jsonl" if (d / f"graphs_{split}.jsonl").exists() else d / f"graphs_{split.split('_')[0]}.jsonl"
        if task_f.exists():
            tasks = load_tasks_jsonl(str(task_f))
            summary["synthetic"][split] = {
                "n_tasks": len(tasks),
                "task_types": sorted(list(set(t.task_type for t in tasks))),
                "task_file": str(task_f),
                "graph_file": str(graph_f) if graph_f.exists() else None,
            }

    # GraphQA splits
    for split in ["train", "validation", "test"]:
        task_f = d / f"tasks_graphqa_{split}.jsonl"
        if task_f.exists():
            tasks = load_tasks_jsonl(str(task_f))
            summary["graphqa"][split] = {
                "n_tasks": len(tasks),
                "task_types": sorted(list(set(t.task_type for t in tasks))),
                "file": str(task_f),
            }

    # GraphArena splits
    for diff in ["easy", "hard"]:
        task_f = d / f"tasks_grapharena_{diff}.jsonl"
        if task_f.exists():
            tasks = load_tasks_jsonl(str(task_f))
            summary["grapharena"][diff] = {
                "n_tasks": len(tasks),
                "task_types": sorted(list(set(t.task_type for t in tasks))),
                "file": str(task_f),
            }

    return summary


# ---------------------------------------------------------------------------
# Full pipeline: generate graphs + tasks + save everything
# ---------------------------------------------------------------------------

def build_full_dataset(
    output_dir: str = "data",
    seed: int = 42,
    graph_counts: Optional[Dict[str, int]] = None,
    task_types: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    End-to-end dataset building:
      1. Generate graphs at all scales.
      2. Split small graphs into train/val.
      3. Generate task instances for each split.
      4. Save everything as JSONL.

    Returns a summary dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if task_types is None:
        task_types = ["shortest_path", "connected_components",
                      "cycle_detection", "max_clique", "hamiltonian_path"]

    # 1. Generate graphs
    dataset = generate_full_dataset(counts=graph_counts, seed=seed)

    # 2. Save graphs and split small
    summary = {}
    splits = {}

    # Small → train / val
    if "small" in dataset:
        train, val = split_small_graphs(dataset["small"], seed=seed)
        splits["small_train"] = train
        splits["small_val"] = val
        save_graphs_jsonl(dataset["small"], str(out / "graphs_small.jsonl"))
        save_graphs_jsonl(train, str(out / "graphs_small_train.jsonl"))
        save_graphs_jsonl(val, str(out / "graphs_small_val.jsonl"))

    # Medium, Large, Stress
    for scale in ["medium", "large", "stress"]:
        if scale in dataset:
            splits[scale] = dataset[scale]
            save_graphs_jsonl(dataset[scale],
                              str(out / f"graphs_{scale}.jsonl"))

    # 3. Generate tasks for each split
    for split_name, graphs in splits.items():
        tasks = generate_task_dataset(graphs, task_types=task_types, seed=seed)
        save_tasks_jsonl(tasks, str(out / f"tasks_{split_name}.jsonl"))
        summary[split_name] = {
            "n_graphs": len(graphs),
            "n_tasks": len(tasks),
            "task_types": list(set(t.task_type for t in tasks)),
        }
        print(f"  {split_name}: {len(graphs)} graphs → {len(tasks)} tasks")

    # 4. Save comprehensive summary across all datasets
    full_summary = get_full_dataset_summary(output_dir)
    full_summary["synthetic_generation"] = summary
    with open(out / "dataset_summary.json", "w") as f:
        json.dump(full_summary, f, indent=2)

    return full_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Build the complete benchmark dataset (graphs + tasks)."
    )
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("CS 5624 — Building Full Benchmark Dataset")
    print("=" * 60)

    summary = build_full_dataset(output_dir=args.output_dir, seed=args.seed)

    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("=" * 60)
    for split, info in summary.items():
        print(f"  {split}: {info['n_graphs']} graphs, {info['n_tasks']} tasks "
              f"({', '.join(info['task_types'])})")
    print("Done!")


if __name__ == "__main__":
    main()

