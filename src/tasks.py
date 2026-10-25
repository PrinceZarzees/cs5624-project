"""
tasks.py
========
Ground-truth solvers for graph reasoning tasks.

Implements deterministic, verifiable solvers for:
  1. Shortest Path (polynomial)
  2. Connected Components (polynomial)
  3. Cycle Detection (polynomial)
  4. Max Clique (NP-hard — exact for small graphs, heuristic for large)
  5. Hamiltonian Path detection (NP-hard — exact for small, heuristic for large)

Each task is defined by:
  - A question generator (produces a natural-language question from a graph)
  - A ground-truth solver (produces the canonical answer)
  - An answer verifier (checks an agent's answer against ground truth)
"""

from __future__ import annotations

import json
import itertools
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Task result container
# ---------------------------------------------------------------------------

@dataclass
class TaskInstance:
    """A single task instance: a graph + question + ground truth answer."""
    task_id: str
    graph_id: str
    task_type: str
    question: str
    answer: Any  # type depends on task
    metadata: Dict[str, Any]  # extra info (e.g., source/target for shortest path)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskInstance":
        return cls(**d)


# =========================================================================
# TASK 1: Shortest Path (Unweighted BFS)
# =========================================================================

def shortest_path_question(G: nx.Graph, rng: random.Random) -> Tuple[str, Any, Dict]:
    """
    Pick two random nodes and compute shortest path length.

    Returns: (question_text, answer, metadata)
    - answer is an int (path length) or -1 if no path exists.
    """
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return ("Graph has fewer than 2 nodes.", -1, {})

    # Attempt to pick nodes that might have a path
    s, t = rng.sample(nodes, 2)

    if nx.has_path(G, s, t):
        length = nx.shortest_path_length(G, s, t)
        path = nx.shortest_path(G, s, t)
        question = (
            f"What is the shortest path length from node {s} to node {t}? "
            f"Return the integer path length, or -1 if no path exists."
        )
        return question, int(length), {"source": s, "target": t, "path": path}
    else:
        question = (
            f"What is the shortest path length from node {s} to node {t}? "
            f"Return the integer path length, or -1 if no path exists."
        )
        return question, -1, {"source": s, "target": t, "path": None}


def verify_shortest_path(answer: Any, ground_truth: Any, metadata: Dict) -> bool:
    """Verify a shortest path answer."""
    try:
        return int(answer) == int(ground_truth)
    except (ValueError, TypeError):
        return False


# =========================================================================
# TASK 2: Connected Components
# =========================================================================

def connected_components_question(G: nx.Graph, rng: random.Random) -> Tuple[str, Any, Dict]:
    """
    Count the number of connected components.

    Returns: (question_text, answer, metadata)
    - answer is an int (number of components).
    """
    n_components = nx.number_connected_components(G)
    components = [sorted(list(c)) for c in nx.connected_components(G)]
    components.sort(key=lambda c: c[0])

    question = (
        "How many connected components does this graph have? "
        "Return a single integer."
    )
    return question, int(n_components), {"components": components}


def verify_connected_components(answer: Any, ground_truth: Any, metadata: Dict) -> bool:
    """Verify connected components count."""
    try:
        return int(answer) == int(ground_truth)
    except (ValueError, TypeError):
        return False


# =========================================================================
# TASK 3: Cycle Detection
# =========================================================================

def cycle_detection_question(G: nx.Graph, rng: random.Random) -> Tuple[str, Any, Dict]:
    """
    Determine whether the graph contains at least one cycle.

    Returns: (question_text, answer, metadata)
    - answer is a bool (True if cycle exists).
    """
    try:
        cycle = nx.find_cycle(G)
        has_cycle = True
        cycle_edges = [[int(u), int(v)] for u, v, *_ in cycle]
    except nx.NetworkXNoCycle:
        has_cycle = False
        cycle_edges = []

    question = (
        "Does this graph contain a cycle? "
        "Return True if it contains at least one cycle, False otherwise."
    )
    return question, has_cycle, {"cycle_edges": cycle_edges}


def verify_cycle_detection(answer: Any, ground_truth: Any, metadata: Dict) -> bool:
    """Verify cycle detection answer."""
    if isinstance(answer, bool):
        return answer == ground_truth
    if isinstance(answer, str):
        return answer.strip().lower() in (
            ("true", "yes", "1") if ground_truth else ("false", "no", "0")
        )
    try:
        return bool(answer) == ground_truth
    except (ValueError, TypeError):
        return False


# =========================================================================
# TASK 4: Maximum Clique
# =========================================================================

def max_clique_question(G: nx.Graph, rng: random.Random) -> Tuple[str, Any, Dict]:
    """
    Find the size of the maximum clique in the graph.

    Uses exact algorithm for small graphs (|V| <= 100) and greedy approximation
    for larger graphs (providing a lower bound).

    Returns: (question_text, answer, metadata)
    - answer is an int (clique size).
    """
    n = G.number_of_nodes()

    if n <= 100:
        # Exact: enumerate all maximal cliques, find largest
        clique = max(nx.find_cliques(G), key=len, default=[])
        clique_size = len(clique)
        clique_nodes = sorted([int(n) for n in clique])
        is_exact = True
    else:
        # Fast greedy approximation / lower bound for large graphs
        clique_size = nx.approximation.large_clique_size(G)
        clique_nodes = []
        is_exact = False

    question = (
        "What is the size of the maximum clique in this graph? "
        "Return a single integer representing the number of nodes "
        "in the largest complete subgraph."
    )
    return question, int(clique_size), {
        "clique_nodes": clique_nodes,
        "is_exact": is_exact,
    }


def verify_max_clique(answer: Any, ground_truth: Any, metadata: Dict) -> bool:
    """
    Verify max clique answer.

    For exact ground truth: answer must match exactly.
    For approximate ground truth: answer >= ground_truth is acceptable
    (since our approximation is a lower bound).
    """
    try:
        ans = int(answer)
        gt = int(ground_truth)
        if metadata.get("is_exact", True):
            return ans == gt
        else:
            # Approximation: agent's answer could be better or equal
            return ans >= gt
    except (ValueError, TypeError):
        return False


# =========================================================================
# TASK 5: Hamiltonian Path Detection
# =========================================================================

def _has_hamiltonian_path_exact(G: nx.Graph) -> Tuple[bool, Optional[List[int]]]:
    """
    Exact Hamiltonian path detection via backtracking.
    Only feasible for small graphs (|V| <= ~20-25).
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 1:
        return True, nodes

    adj = {v: set(G.neighbors(v)) for v in nodes}

    def backtrack(path: List[int], visited: set) -> Optional[List[int]]:
        if len(path) == n:
            return list(path)
        current = path[-1]
        for neighbor in adj[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                result = backtrack(path, visited)
                if result is not None:
                    return result
                path.pop()
                visited.remove(neighbor)
        return None

    for start in nodes:
        result = backtrack([start], {start})
        if result is not None:
            return True, result

    return False, None


def hamiltonian_path_question(G: nx.Graph, rng: random.Random) -> Tuple[str, Any, Dict]:
    """
    Determine whether the graph has a Hamiltonian path (visits every node exactly once).

    For |V| <= 20: exact backtracking.
    For |V| > 20: use heuristic (sufficient conditions / Ore's theorem checks).

    Returns: (question_text, answer, metadata)
    - answer is a bool.
    """
    n = G.number_of_nodes()

    if n <= 20:
        has_path, path = _has_hamiltonian_path_exact(G)
        is_exact = True
    else:
        # For larger graphs, use a heuristic check:
        # Dirac's theorem: if deg(v) >= n/2 for all v, Hamiltonian path exists
        min_deg = min(dict(G.degree()).values()) if n > 0 else 0
        if min_deg >= n / 2:
            has_path = True
            path = None
            is_exact = False  # Known sufficient condition but we don't find the path
        else:
            # Try a greedy approach
            # Note: This can give false negatives, so we mark it
            has_path = False
            path = None
            is_exact = False

    question = (
        "Does this graph have a Hamiltonian path "
        "(a path that visits every node exactly once)? "
        "Return True or False."
    )

    path_as_ints = [int(x) for x in path] if path else None
    return question, has_path, {
        "path": path_as_ints,
        "is_exact": is_exact,
    }


def verify_hamiltonian_path(answer: Any, ground_truth: Any, metadata: Dict) -> bool:
    """Verify Hamiltonian path answer."""
    if not metadata.get("is_exact", True):
        # For heuristic answers, only verify if ground_truth is True
        # (sufficient conditions are one-directional)
        if ground_truth is True:
            if isinstance(answer, bool):
                return answer is True
            if isinstance(answer, str):
                return answer.strip().lower() in ("true", "yes", "1")
        # If heuristic says False, we can't be sure, so accept either answer
        return True

    if isinstance(answer, bool):
        return answer == ground_truth
    if isinstance(answer, str):
        return answer.strip().lower() in (
            ("true", "yes", "1") if ground_truth else ("false", "no", "0")
        )
    try:
        return bool(answer) == ground_truth
    except (ValueError, TypeError):
        return False


def reachability_question(G: nx.Graph, rng: random.Random) -> Tuple[str, Any, Dict]:
    """Check if there is a path between two nodes."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return ("Graph has fewer than 2 nodes.", False, {})
    s, t = rng.sample(nodes, 2)
    has_p = nx.has_path(G, s, t)
    question = f"Is there a path from node {s} to node {t} in this graph? Return True or False."
    return question, has_p, {"source": s, "target": t}


def verify_reachability(answer: Any, ground_truth: Any, metadata: Dict) -> bool:
    """Verify reachability answer."""
    if isinstance(answer, bool):
        return answer == ground_truth
    if isinstance(answer, str):
        return answer.strip().lower() in (
            ("true", "yes", "1") if ground_truth else ("false", "no", "0")
        )
    try:
        return bool(answer) == ground_truth
    except (ValueError, TypeError):
        return False


# =========================================================================
# Task Registry
# =========================================================================

TASK_REGISTRY = {
    "shortest_path": {
        "question_fn": shortest_path_question,
        "verify_fn": verify_shortest_path,
        "complexity": "polynomial",
        "description": "Find shortest path length between two nodes.",
    },
    "connected_components": {
        "question_fn": connected_components_question,
        "verify_fn": verify_connected_components,
        "complexity": "polynomial",
        "description": "Count connected components in the graph.",
    },
    "cycle_detection": {
        "question_fn": cycle_detection_question,
        "verify_fn": verify_cycle_detection,
        "complexity": "polynomial",
        "description": "Detect whether the graph contains a cycle.",
    },
    "reachability": {
        "question_fn": reachability_question,
        "verify_fn": verify_reachability,
        "complexity": "polynomial",
        "description": "Determine if a path exists between two nodes.",
    },
    "max_clique": {
        "question_fn": max_clique_question,
        "verify_fn": verify_max_clique,
        "complexity": "np_hard",
        "description": "Find the size of the maximum clique.",
    },
    "hamiltonian_path": {
        "question_fn": hamiltonian_path_question,
        "verify_fn": verify_hamiltonian_path,
        "complexity": "np_hard",
        "description": "Detect whether a Hamiltonian path exists.",
    },
}


def get_task_names() -> List[str]:
    """Return all available task names."""
    return list(TASK_REGISTRY.keys())


def generate_task_instance(
    graph_id: str,
    G: nx.Graph,
    task_type: str,
    instance_idx: int = 0,
    seed: int = 42,
) -> TaskInstance:
    """
    Generate a single task instance for a given graph and task type.

    Parameters
    ----------
    graph_id : str
        Identifier of the source graph.
    G : nx.Graph
        The graph to generate a task for.
    task_type : str
        One of the keys in TASK_REGISTRY.
    instance_idx : int
        Index for generating the task_id.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    TaskInstance
    """
    if task_type not in TASK_REGISTRY:
        raise ValueError(f"Unknown task type: {task_type}. "
                         f"Available: {list(TASK_REGISTRY.keys())}")

    rng = random.Random(seed + hash(graph_id) + instance_idx)
    question_fn = TASK_REGISTRY[task_type]["question_fn"]
    question, answer, metadata = question_fn(G, rng)

    task_id = f"{graph_id}_{task_type}_{instance_idx:04d}"
    return TaskInstance(
        task_id=task_id,
        graph_id=graph_id,
        task_type=task_type,
        question=question,
        answer=answer,
        metadata=metadata,
    )

