"""Graph tasks: question text, exact solver, and answer checker for each task.

Every task works on an undirected networkx graph and uses the same answer
convention as GraphArena: the model writes its final answer as a list of
nodes inside square brackets, e.g. ``[3, 17, 42]``.

Each task defines three things:

* ``make_params(G, rng)``  - pick the task inputs (e.g. source/target nodes).
* ``solve(G, params)``     - exact ground truth, computed with NetworkX.
* ``check(G, params, truth, nodes)`` - check a parsed answer (a list of nodes,
  already mapped back to the graph's own node ids) and return a ``Check``.

Tasks are registered in ``TASKS`` by name.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx


@dataclass
class Check:
    """Result of checking one answer.

    feasible: the answer is a valid object for the task (a real path, a real
        clique, ...), even if it is not the best one.
    correct:  the answer is valid AND optimal / matches the ground truth.
    value:    the task's score for the answer (path length, clique size, ...).
    note:     short reason when the answer is wrong, for error analysis.
    """

    feasible: bool
    correct: bool
    value: float | None = None
    note: str = ""


@dataclass
class Task:
    name: str
    kind: str  # "P" (polynomial) or "NP" (NP-hard)
    question: Callable[[dict], str]
    make_params: Callable[[nx.Graph, random.Random], dict]
    solve: Callable[[nx.Graph, dict], dict]
    check: Callable[[nx.Graph, dict, dict, list], Check]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _is_simple_path(G: nx.Graph, nodes: list) -> str:
    """Return "" if ``nodes`` is a simple path in G, else the reason it is not."""
    if len(nodes) == 0:
        return "empty answer"
    if len(set(nodes)) != len(nodes):
        return "path repeats a node"
    for a, b in zip(nodes, nodes[1:]):
        if not G.has_edge(a, b):
            return f"no edge between {a} and {b}"
    return ""


def _pick_reachable_pair(G: nx.Graph, rng: random.Random):
    """Pick two different nodes that are connected by some path."""
    comps = [list(c) for c in nx.connected_components(G) if len(c) >= 2]
    if not comps:
        raise ValueError("graph has no edge, cannot pick a reachable pair")
    # weight components by size so the pair is usually in the big component
    comp = rng.choices(comps, weights=[len(c) for c in comps])[0]
    comp.sort(key=str)
    s, t = rng.sample(comp, 2)
    return s, t


# --------------------------------------------------------------------------
# shortest path (GraphArena: Distance)
# --------------------------------------------------------------------------

def _sp_question(p):
    s, t = p["source"], p["target"]
    return (
        f"Find a shortest path (the fewest edges) from node {s} to node {t}. "
        f"Give the path as a list of nodes that starts with {s} and ends with {t}, "
        f"for example: [{s}, ..., {t}]."
    )


def _sp_params(G, rng):
    s, t = _pick_reachable_pair(G, rng)
    return {"source": s, "target": t}


def _sp_solve(G, p):
    try:
        path = nx.shortest_path(G, p["source"], p["target"])
    except nx.NetworkXNoPath:
        return {"length": -1, "path": []}
    return {"length": len(path) - 1, "path": path}


def _sp_check(G, p, truth, nodes):
    if truth["length"] == -1:
        if not nodes:
            return Check(True, True, -1)
        return Check(False, False, None, "gave a path where no path exists")
    why = _is_simple_path(G, nodes)
    if why:
        return Check(False, False, None, why)
    if nodes[0] != p["source"] or nodes[-1] != p["target"]:
        return Check(False, False, None, "path does not start/end at the given nodes")
    length = len(nodes) - 1
    if length != truth["length"]:
        return Check(True, False, length, f"path has {length} edges, shortest is {truth['length']}")
    return Check(True, True, length)


# --------------------------------------------------------------------------
# reachability (GraphQA)
# --------------------------------------------------------------------------

def _reach_question(p):
    s, t = p["source"], p["target"]
    return f"Is there a path from node {s} to node {t}? If yes, give one path as a list; otherwise give []."


def _reach_params(G, rng):
    s, t = _pick_reachable_pair(G, rng)
    return {"source": s, "target": t}


def _reach_solve(G, p):
    try:
        return {"reachable": True, "path": nx.shortest_path(G, p["source"], p["target"])}
    except nx.NetworkXNoPath:
        return {"reachable": False, "path": []}


def _reach_check(G, p, truth, nodes):
    if not truth["reachable"]:
        if not nodes:
            return Check(True, True, 0)
        return Check(False, False, None, "gave a path where no path exists")
    why = _is_simple_path(G, nodes)
    if why:
        return Check(False, False, None, why)
    if nodes[0] != p["source"] or nodes[-1] != p["target"]:
        return Check(False, False, None, "path does not start/end at the given nodes")
    return Check(True, True, len(nodes) - 1)


# --------------------------------------------------------------------------
# connected components (GraphArena: Connected)
# --------------------------------------------------------------------------

def _cc_question(p):
    return (
        "Find all connected components of the graph and give exactly one node "
        "from each component. Give the answer as a list of nodes, for example: "
        "[a, b, c]."
    )


def _cc_params(G, rng):
    return {}


def _cc_solve(G, p):
    comps = [sorted(c, key=str) for c in nx.connected_components(G)]
    comps.sort(key=lambda c: str(c[0]))
    return {"count": len(comps), "representatives": [c[0] for c in comps]}


def _cc_check(G, p, truth, nodes):
    if len(set(nodes)) != len(nodes):
        return Check(False, False, None, "answer repeats a node")
    comp_of = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for v in comp:
            comp_of[v] = i
    seen = set()
    for v in nodes:
        if comp_of[v] in seen:
            return Check(False, False, None, f"two answer nodes are in the same component (e.g. {v})")
        seen.add(comp_of[v])
    k = len(nodes)
    if k != truth["count"]:
        return Check(True, False, k, f"gave {k} components, graph has {truth['count']}")
    return Check(True, True, k)


# --------------------------------------------------------------------------
# cycle detection (GraphQA: cycle check, but the model must show a cycle)
# --------------------------------------------------------------------------

def _cyc_question(p):
    return (
        "Does the graph contain a cycle? If it does, give one cycle as a list of "
        "nodes in the order you visit them, for example: [a, b, c] (the edge from "
        "the last node back to the first is implied). If the graph has no cycle, "
        "answer with an empty list: []."
    )


def _cyc_params(G, rng):
    return {}


def _cyc_solve(G, p):
    try:
        edges = nx.find_cycle(G)
        cycle = [u for u, _ in edges]
        return {"has_cycle": True, "cycle": cycle}
    except nx.NetworkXNoCycle:
        return {"has_cycle": False, "cycle": []}


def _cyc_check(G, p, truth, nodes):
    if len(nodes) == 0:
        if truth["has_cycle"]:
            return Check(True, False, 0, "said no cycle, but the graph has one")
        return Check(True, True, 0)
    # allow the model to repeat the first node at the end: [a, b, c, a]
    if len(nodes) >= 2 and nodes[0] == nodes[-1]:
        nodes = nodes[:-1]
    if len(nodes) < 3:
        return Check(False, False, None, "a cycle needs at least 3 nodes")
    why = _is_simple_path(G, nodes)
    if why:
        return Check(False, False, None, why)
    if not G.has_edge(nodes[-1], nodes[0]):
        return Check(False, False, None, "no edge from the last node back to the first")
    return Check(True, True, len(nodes))


# --------------------------------------------------------------------------
# diameter (GraphArena: Diameter)
# --------------------------------------------------------------------------

def _diam_question(p):
    return (
        "The graph is connected. Find its diameter: the largest shortest-path "
        "distance between any two nodes. Give a shortest path between two nodes "
        "that are this far apart, as a list of nodes, for example: [a, ..., b]."
    )


def _diam_params(G, rng):
    if not nx.is_connected(G):
        raise ValueError("diameter task needs a connected graph")
    return {}


def _diam_solve(G, p):
    if G.number_of_nodes() == 1:
        v = next(iter(G.nodes))
        return {"diameter": 0, "path": [v]}
    d = nx.diameter(G, usebounds=True)
    # find one pair of nodes that are d apart (a witness path)
    for v in sorted(G.nodes, key=str):
        dist = nx.single_source_shortest_path_length(G, v)
        far = [u for u, k in dist.items() if k == d]
        if far:
            u = sorted(far, key=str)[0]
            return {"diameter": d, "path": nx.shortest_path(G, v, u)}
    raise RuntimeError("could not find a diameter path")  # should not happen


def _diam_check(G, p, truth, nodes):
    why = _is_simple_path(G, nodes)
    if why:
        return Check(False, False, None, why)
    length = len(nodes) - 1
    true_dist = nx.shortest_path_length(G, nodes[0], nodes[-1])
    if length != true_dist:
        return Check(False, False, None,
                     f"path has {length} edges but its end nodes are only {true_dist} apart")
    if length != truth["diameter"]:
        return Check(True, False, length, f"distance {length}, diameter is {truth['diameter']}")
    return Check(True, True, length)


# --------------------------------------------------------------------------
# maximum clique (GraphArena: MCP) - NP-hard
# --------------------------------------------------------------------------

def _mc_question(p):
    return (
        "Find a maximum clique: the largest set of nodes in which every pair of "
        "nodes is connected by an edge. Give the clique as a list of nodes, for "
        "example: [a, b, c]."
    )


def _mc_params(G, rng):
    return {}


def _mc_solve(G, p):
    clique, _ = nx.max_weight_clique(G, weight=None)
    clique = sorted(clique, key=str)
    return {"size": len(clique), "clique": clique}


def _mc_check(G, p, truth, nodes):
    if len(nodes) == 0:
        return Check(False, False, None, "empty answer")
    if len(set(nodes)) != len(nodes):
        return Check(False, False, None, "answer repeats a node")
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if not G.has_edge(a, b):
                return Check(False, False, None, f"{a} and {b} are not connected, not a clique")
    k = len(nodes)
    if k != truth["size"]:
        return Check(True, False, k, f"clique of size {k}, maximum is {truth['size']}")
    return Check(True, True, k)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

TASKS: dict[str, Task] = {
    t.name: t
    for t in [
        Task("shortest_path", "P", _sp_question, _sp_params, _sp_solve, _sp_check),
        Task("reachability", "P", _reach_question, _reach_params, _reach_solve, _reach_check),
        Task("connected_components", "P", _cc_question, _cc_params, _cc_solve, _cc_check),
        Task("cycle_detection", "P", _cyc_question, _cyc_params, _cyc_solve, _cyc_check),
        Task("diameter", "P", _diam_question, _diam_params, _diam_solve, _diam_check),
        Task("max_clique", "NP", _mc_question, _mc_params, _mc_solve, _mc_check),
    ]
}


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise KeyError(f"unknown task {name!r}; known tasks: {sorted(TASKS)}") from None


def as_json_value(x: Any) -> Any:
    """Make solver output JSON-safe (tuples -> lists)."""
    if isinstance(x, (list, tuple)):
        return [as_json_value(v) for v in x]
    if isinstance(x, dict):
        return {k: as_json_value(v) for k, v in x.items()}
    return x
