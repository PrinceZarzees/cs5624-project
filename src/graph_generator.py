"""
graph_generator.py
==================
Synthetic benchmark graph generator for the Harness Optimization project.

Generates graphs at four scales using three random graph models:
  - Erdős–Rényi (G(n,p))
  - Barabási–Albert (preferential attachment)
  - Watts–Strogatz (small-world)

Each graph is serialized as a JSON-friendly dictionary containing:
  - "graph_id":   unique identifier
  - "model":      generation model name
  - "num_nodes":  |V|
  - "num_edges":  |E|
  - "nodes":      list of node labels
  - "edges":      list of [u, v] pairs (undirected)
  - "directed":   False (all benchmarks use undirected graphs)
  - "scale":      one of "small", "medium", "large", "stress"
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Scale definitions
# ---------------------------------------------------------------------------

SCALE_RANGES: Dict[str, Tuple[int, int]] = {
    "small":  (4, 50),        # Training / Validation
    "medium": (100, 300),     # Test 1
    "large":  (500, 1000),    # Test 2
    "stress": (1500, 2000),   # Test 3 (stress test)
}


# ---------------------------------------------------------------------------
# Graph model factories
# ---------------------------------------------------------------------------

def _er_graph(n: int, rng: random.Random) -> nx.Graph:
    """Erdős–Rényi G(n, p) with p chosen to give ~2*n edges on average."""
    # p = 2*n / C(n,2) = 4/(n-1), clamped to [0.05, 0.6]
    if n <= 1:
        return nx.empty_graph(n)
    p = min(0.6, max(0.05, 4.0 / (n - 1)))
    seed = rng.randint(0, 2**31)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    return G


def _ba_graph(n: int, rng: random.Random) -> nx.Graph:
    """Barabási–Albert preferential attachment (m = 2 or 3)."""
    m = min(3, max(1, n // 10)) if n > 3 else 1
    m = min(m, n - 1)
    seed = rng.randint(0, 2**31)
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    return G


def _ws_graph(n: int, rng: random.Random) -> nx.Graph:
    """Watts–Strogatz small-world (k = 4, p = 0.3)."""
    k = min(4, n - 1)
    if k % 2 != 0:
        k = max(2, k - 1)
    if k < 2:
        k = 2
    if n <= k:
        return nx.complete_graph(n)
    seed = rng.randint(0, 2**31)
    G = nx.watts_strogatz_graph(n, k, 0.3, seed=seed)
    return G


GRAPH_MODELS = {
    "erdos_renyi":    _er_graph,
    "barabasi_albert": _ba_graph,
    "watts_strogatz":  _ws_graph,
}


# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------

@dataclass
class GraphInstance:
    """A single benchmark graph with metadata."""
    graph_id: str
    model: str
    num_nodes: int
    num_edges: int
    nodes: List[int]
    edges: List[List[int]]
    directed: bool = False
    scale: str = "small"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_networkx(self) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from([tuple(e) for e in self.edges])
        return G

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GraphInstance":
        return cls(**d)

    @classmethod
    def from_networkx(cls, G: nx.Graph, graph_id: str, model: str,
                      scale: str) -> "GraphInstance":
        nodes = sorted(G.nodes())
        edges = [[int(u), int(v)] for u, v in G.edges()]
        return cls(
            graph_id=graph_id,
            model=model,
            num_nodes=G.number_of_nodes(),
            num_edges=G.number_of_edges(),
            nodes=[int(n) for n in nodes],
            edges=edges,
            directed=False,
            scale=scale,
        )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _make_graph_id(scale: str, model: str, idx: int) -> str:
    """Deterministic human-readable graph ID."""
    return f"{scale}_{model}_{idx:04d}"


def generate_graphs(
    scale: str,
    count: int,
    models: Optional[List[str]] = None,
    seed: int = 42,
) -> List[GraphInstance]:
    """
    Generate `count` graphs uniformly split across the requested models.

    Parameters
    ----------
    scale : str
        One of "small", "medium", "large", "stress".
    count : int
        Total number of graphs to generate (split evenly across models).
    models : list of str, optional
        Subset of GRAPH_MODELS keys. Default: all three.
    seed : int
        Master RNG seed for reproducibility.

    Returns
    -------
    list of GraphInstance
    """
    if models is None:
        models = list(GRAPH_MODELS.keys())

    lo, hi = SCALE_RANGES[scale]
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    per_model = count // len(models)
    remainder = count % len(models)

    instances: List[GraphInstance] = []
    global_idx = 0

    for mi, model_name in enumerate(models):
        factory = GRAPH_MODELS[model_name]
        n_graphs = per_model + (1 if mi < remainder else 0)

        for _ in range(n_graphs):
            n = rng.randint(lo, hi)
            G = factory(n, rng)

            # Relabel nodes to random permutation to avoid label leakage
            mapping = {old: new for old, new in
                       zip(G.nodes(), np_rng.permutation(G.number_of_nodes()))}
            G = nx.relabel_nodes(G, {k: int(v) for k, v in mapping.items()})

            gid = _make_graph_id(scale, model_name, global_idx)
            inst = GraphInstance.from_networkx(G, gid, model_name, scale)
            instances.append(inst)
            global_idx += 1

    return instances


# ---------------------------------------------------------------------------
# Full dataset generation
# ---------------------------------------------------------------------------

DEFAULT_COUNTS = {
    "small":  300,   # 200 train + 100 val
    "medium": 60,
    "large":  30,
    "stress": 15,
}


def generate_full_dataset(
    counts: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Dict[str, List[GraphInstance]]:
    """
    Generate the complete benchmark dataset across all four scales.

    Returns a dict: scale_name -> list of GraphInstance.
    """
    if counts is None:
        counts = DEFAULT_COUNTS

    dataset: Dict[str, List[GraphInstance]] = {}
    for scale, count in counts.items():
        print(f"Generating {count} graphs at scale '{scale}' "
              f"(N ∈ [{SCALE_RANGES[scale][0]}, {SCALE_RANGES[scale][1]}]) ...")
        dataset[scale] = generate_graphs(scale, count, seed=seed)
    return dataset


# ---------------------------------------------------------------------------
# Train / Val split for small graphs
# ---------------------------------------------------------------------------

def split_small_graphs(
    small_graphs: List[GraphInstance],
    train_ratio: float = 0.67,
    seed: int = 42,
) -> Tuple[List[GraphInstance], List[GraphInstance]]:
    """Split small-scale graphs into train and validation sets."""
    rng = random.Random(seed)
    indices = list(range(len(small_graphs)))
    rng.shuffle(indices)
    n_train = int(len(small_graphs) * train_ratio)
    train = [small_graphs[i] for i in indices[:n_train]]
    val = [small_graphs[i] for i in indices[n_train:]]
    return train, val


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Generate full dataset and save to data/ directory."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic benchmark graphs for CS 5624 project."
    )
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory for JSONL files.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--small-count", type=int, default=300)
    parser.add_argument("--medium-count", type=int, default=60)
    parser.add_argument("--large-count", type=int, default=30)
    parser.add_argument("--stress-count", type=int, default=15)
    args = parser.parse_args()

    counts = {
        "small":  args.small_count,
        "medium": args.medium_count,
        "large":  args.large_count,
        "stress": args.stress_count,
    }

    dataset = generate_full_dataset(counts=counts, seed=args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save each scale as a JSONL file
    for scale, graphs in dataset.items():
        fpath = out_dir / f"graphs_{scale}.jsonl"
        with open(fpath, "w") as f:
            for g in graphs:
                f.write(json.dumps(g.to_dict()) + "\n")
        print(f"  Saved {len(graphs)} graphs to {fpath}")

    # Additionally split small into train/val
    if "small" in dataset:
        train, val = split_small_graphs(dataset["small"], seed=args.seed)

        for split_name, split_data in [("train", train), ("val", val)]:
            fpath = out_dir / f"graphs_small_{split_name}.jsonl"
            with open(fpath, "w") as f:
                for g in split_data:
                    f.write(json.dumps(g.to_dict()) + "\n")
            print(f"  Saved {len(split_data)} graphs to {fpath}")

    print("\nDataset generation complete!")
    print(f"Total graphs: {sum(len(g) for g in dataset.values())}")


if __name__ == "__main__":
    main()
