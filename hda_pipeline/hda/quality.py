"""Feasibility validators and pure normalized-quality scores for phase-2 tasks.

The evaluator receives a candidate node set (not just an objective value), so a
program cannot claim a large score without a valid solution.
"""
from __future__ import annotations

import re
from typing import Iterable


def parse_nodes(text: str) -> list[int] | None:
    """Parse a node list from '[1, 2, 3]', '1 2 3', or one-per-line output."""
    if not text:
        return None
    m = re.search(r"\[(.*?)\]", text, re.S)
    body = m.group(1) if m else text
    nums = re.findall(r"-?\d+", body)
    if m is None and not nums:
        return None
    try:
        return [int(x) for x in nums]
    except ValueError:
        return None


def _edge_set(edges: Iterable) -> set[tuple[int, int]]:
    out = set()
    for e in edges:
        u, v = int(e[0]), int(e[1])
        out.add((u, v) if u <= v else (v, u))
    return out


def validate_clique(nodes, n, edges) -> bool:
    if nodes is None or len(set(nodes)) != len(nodes):
        return False
    if any(v < 0 or v >= n for v in nodes):
        return False
    E = _edge_set(edges)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1:]:
            a, b = (u, v) if u <= v else (v, u)
            if (a, b) not in E:
                return False
    return True


def validate_independent_set(nodes, n, edges) -> bool:
    if nodes is None or len(set(nodes)) != len(nodes):
        return False
    if any(v < 0 or v >= n for v in nodes):
        return False
    E = _edge_set(edges)
    S = set(nodes)
    for u, v in E:
        if u in S and v in S:
            return False
    return True


def validate_vertex_cover(nodes, n, edges) -> bool:
    if nodes is None or len(set(nodes)) != len(nodes):
        return False
    if any(v < 0 or v >= n for v in nodes):
        return False
    C = set(nodes)
    for e in edges:
        if int(e[0]) not in C and int(e[1]) not in C:
            return False
    return True


VALIDATORS = {
    "max_clique": validate_clique,
    "max_independent_set": validate_independent_set,
    "min_vertex_cover": validate_vertex_cover,
}


def normalized_quality(task: str, value: float, reference: float) -> float:
    """Pure normalized quality in [0,1]; maximization vs minimization."""
    if reference is None or reference <= 0 or value is None:
        return 0.0
    if task in {"max_clique", "max_independent_set"}:
        q = value / reference
    elif task == "min_vertex_cover":
        q = reference / value if value > 0 else 0.0
    else:
        raise KeyError(task)
    return max(0.0, min(1.0, q))
