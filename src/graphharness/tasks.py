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
    check: Callable[[nx.Graph, dict, dict, Any], Check]
    answer_kind: str = "nodes"  # nodes | number | boolean | string | mapping | two_lists


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
# common neighbors (GraphArena: Neighbor)
# --------------------------------------------------------------------------

def _cn_question(p):
    u, v = p["node1"], p["node2"]
    return (
        f"Find the common neighbors of node {u} and node {v}: nodes that are "
        f"adjacent to both {u} and {v}. Give the answer as a list of nodes, "
        "for example: [a, b, c]. Use [] if there are none."
    )


def _cn_params(G, rng):
    nodes = sorted(G.nodes, key=str)
    pairs = [(u, v) for i, u in enumerate(nodes) for v in nodes[i + 1:]]
    rng.shuffle(pairs)
    for u, v in pairs:
        if set(G.neighbors(u)) & set(G.neighbors(v)):
            return {"node1": u, "node2": v}
    return {"node1": nodes[0], "node2": nodes[1]} if len(nodes) >= 2 else {}


def _cn_solve(G, p):
    u, v = p["node1"], p["node2"]
    common = sorted(set(G.neighbors(u)) & set(G.neighbors(v)), key=str)
    return {"count": len(common), "nodes": common}


def _cn_check(G, p, truth, nodes):
    if len(set(nodes)) != len(nodes):
        return Check(False, False, None, "answer repeats a node")
    u, v = p["node1"], p["node2"]
    nbrs_u, nbrs_v = set(G.neighbors(u)), set(G.neighbors(v))
    expected = nbrs_u & nbrs_v
    if any(n not in G for n in nodes):
        return Check(False, False, None, "answer contains an unknown node")
    if any(n not in expected for n in nodes):
        return Check(False, False, None, "answer contains a node that is not a common neighbor")
    if set(nodes) != expected:
        return Check(True, False, len(nodes),
                     f"gave {len(nodes)} common neighbors, graph has {len(expected)}")
    return Check(True, True, len(nodes))


# --------------------------------------------------------------------------
# maximum independent set (GraphArena: MIS) - NP-hard
# --------------------------------------------------------------------------

def _mis_question(p):
    return (
        "Find a maximum independent set: the largest set of nodes with no edge "
        "between any two of them. Give the set as a list of nodes, for example: "
        "[a, b, c]."
    )


def _mis_params(G, rng):
    return {}


def _mis_solve(G, p):
    if G.number_of_nodes() > 100:
        raise ValueError("exact maximum independent set is limited to 100 nodes")
    clique, _ = nx.max_weight_clique(nx.complement(G), weight=None)
    nodes = sorted(clique, key=str)
    return {"size": len(nodes), "nodes": nodes}


def _mis_check(G, p, truth, nodes):
    if len(set(nodes)) != len(nodes):
        return Check(False, False, None, "answer repeats a node")
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if G.has_edge(a, b):
                return Check(False, False, None, f"{a} and {b} are adjacent")
    k = len(nodes)
    if k != truth["size"]:
        return Check(True, False, k, f"independent set of size {k}, maximum is {truth['size']}")
    return Check(True, True, k)


# --------------------------------------------------------------------------
# minimum vertex cover (GraphArena: MVC) - NP-hard
# --------------------------------------------------------------------------

def _mvc_question(p):
    return (
        "Find a minimum vertex cover: the smallest set of nodes such that every "
        "edge has at least one endpoint in the set. Give the set as a list of "
        "nodes, for example: [a, b, c]."
    )


def _mvc_params(G, rng):
    return {}


def _mvc_solve(G, p):
    if G.number_of_nodes() > 100:
        raise ValueError("exact minimum vertex cover is limited to 100 nodes")
    clique, _ = nx.max_weight_clique(nx.complement(G), weight=None)
    cover = sorted(set(G.nodes) - set(clique), key=str)
    return {"size": len(cover), "nodes": cover}


def _mvc_check(G, p, truth, nodes):
    if len(set(nodes)) != len(nodes):
        return Check(False, False, None, "answer repeats a node")
    cover = set(nodes)
    for u, v in G.edges:
        if u not in cover and v not in cover:
            return Check(False, False, None, f"edge {u}-{v} is not covered")
    k = len(nodes)
    if k != truth["size"]:
        return Check(True, False, k, f"vertex cover of size {k}, minimum is {truth['size']}")
    return Check(True, True, k)


# --------------------------------------------------------------------------
# Hamiltonian path (proposal NP-hard alternative)
# --------------------------------------------------------------------------

def _hp_question(p):
    return (
        "Does the graph contain a Hamiltonian path, a path that visits every "
        "node exactly once? If yes, give one such path as a list of nodes; if "
        "not, answer with []."
    )


def _hp_params(G, rng):
    return {}


def _hp_find(G):
    n = G.number_of_nodes()
    if n == 0:
        return []
    if n == 1:
        return [next(iter(G.nodes))]
    if not nx.is_connected(G):
        return None
    # Fast path for a simple path graph.
    if all(d <= 2 for _, d in G.degree()):
        ends = [v for v, d in G.degree() if d == 1]
        if len(ends) == 2:
            return nx.shortest_path(G, ends[0], ends[1])
    nodes = sorted(G.nodes, key=str)
    visited = set()

    def dfs(path):
        if len(path) == n:
            return list(path)
        last = path[-1]
        for nxt in sorted(G.neighbors(last), key=str):
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                found = dfs(path)
                if found is not None:
                    return found
                path.pop()
                visited.remove(nxt)
        return None

    for start in sorted(nodes, key=lambda v: (G.degree(v), str(v))):
        visited = {start}
        found = dfs([start])
        if found is not None:
            return found
    return None


def _hp_solve(G, p):
    n = G.number_of_nodes()
    if n > 100:
        # Only path-shaped large graphs are solved exactly; arbitrary large
        # Hamiltonian-path instances are outside this project's data range.
        if nx.is_connected(G) and all(d <= 2 for _, d in G.degree()):
            path = _hp_find(G)
            return {"exists": path is not None, "path": path or []}
        raise ValueError("exact Hamiltonian path is limited to 100 nodes")
    path = _hp_find(G)
    return {"exists": path is not None, "path": path or []}


def _hp_check(G, p, truth, nodes):
    if not truth["exists"]:
        if not nodes:
            return Check(True, True, 0)
        return Check(False, False, None, "gave a path but no Hamiltonian path exists")
    if not nodes:
        return Check(True, False, 0, "said no path, but a Hamiltonian path exists")
    why = _is_simple_path(G, nodes)
    if why:
        return Check(False, False, None, why)
    if len(nodes) != G.number_of_nodes():
        return Check(True, False, len(nodes),
                     f"path has {len(nodes)} nodes, graph has {G.number_of_nodes()}")
    return Check(True, True, len(nodes))



# --------------------------------------------------------------------------
# graph edit distance (GraphArena: GED) - NP-hard
# --------------------------------------------------------------------------

def _ged_question(p):
    return (
        "You are given two node-labelled graphs, Graph A and Graph B. Find a "
        "node mapping from Graph A to Graph B with minimum graph edit cost "
        "(node relabel cost 1 if labels differ, plus 1 per edge present in only "
        "one graph). Give the mapping as a list of Graph B node ids, one per "
        "Graph A node in order, for example: [1, 0, 2]. You may also give only "
        "the minimum edit cost as a number."
    )


def _ged_params(G, rng):
    return {}


def _node_label(G, v):
    return str(G.nodes[v].get("label", 0))


def _ged_order(G):
    try:
        return sorted(G.nodes)
    except TypeError:
        return sorted(G.nodes, key=str)


def _ged_cost(g1: nx.Graph, g2: nx.Graph, mapping: list[int]) -> int:
    order = _ged_order(g1)
    pos = {v: i for i, v in enumerate(order)}
    mapped = {u: mapping[pos[u]] for u in order}
    cost = 0
    for u in order:
        if _node_label(g1, u) != _node_label(g2, mapped[u]):
            cost += 1
    for u, v in g1.edges:
        if not g2.has_edge(mapped[u], mapped[v]):
            cost += 1
    g1_edges = {(mapped[u], mapped[v]) for u, v in g1.edges}
    for u, v in g2.edges:
        if (u, v) not in g1_edges and (v, u) not in g1_edges:
            cost += 1
    return cost

def _ged_solve(G, p):
    g1, g2 = G
    if g1.number_of_nodes() != g2.number_of_nodes() or g1.number_of_nodes() > 9:
        raise ValueError("exact GED is limited to equal-size graphs with at most 9 nodes")
    paths, cost = nx.optimal_edit_paths(
        g1, g2, node_subst_cost=lambda a, b: 0 if a.get("label") == b.get("label") else 1
    )
    order = _ged_order(g1)
    pos = {v: i for i, v in enumerate(order)}
    mapping = [None] * len(order)
    for u, v in paths[0][0]:
        mapping[pos[u]] = v
    return {"cost": cost, "mapping": mapping}


def _ged_check(G, p, truth, parsed):
    g1, g2 = G
    if isinstance(parsed, (int, float)):
        cost = int(parsed)
        return Check(True, cost == truth["cost"], cost,
                     f"cost {cost}, minimum is {truth['cost']}")
    if not isinstance(parsed, list) or len(parsed) != g1.number_of_nodes():
        return Check(False, False, None, "mapping must have one entry per Graph A node")
    if set(parsed) != set(g2.nodes) or len(set(parsed)) != len(parsed):
        return Check(False, False, None, "mapping must use every Graph B node exactly once")
    cost = _ged_cost(g1, g2, parsed)
    return Check(True, cost == truth["cost"], cost,
                 f"mapping cost {cost}, minimum is {truth['cost']}")


# --------------------------------------------------------------------------
# maximum common subgraph (GraphArena: MCS) - NP-hard
# --------------------------------------------------------------------------

def _mcs_question(p):
    return (
        "You are given two graphs, Graph A and Graph B. Find a maximum common "
        "subgraph: node sets from A and B of equal size whose induced subgraphs "
        "are isomorphic. Give two lists, first the nodes in Graph A and then the "
        "nodes in Graph B, for example: [0, 2, 3], [1, 2, 4]."
    )


def _mcs_params(G, rng):
    return {}


def _mcs_solve(G, p):
    g1, g2 = G
    if g1.number_of_nodes() > 10 or g2.number_of_nodes() > 10:
        raise ValueError("exact maximum common subgraph is limited to 10 nodes")
    ismags = nx.isomorphism.ISMAGS(g1, g2)
    mapping = next(iter(ismags.largest_common_subgraph(symmetry=True)))
    nodes1 = sorted(mapping.keys(), key=str)
    nodes2 = [mapping[v] for v in nodes1]
    return {"size": len(nodes1), "nodes1": nodes1, "nodes2": nodes2}


def _mcs_check(G, p, truth, parsed):
    g1, g2 = G
    if not isinstance(parsed, list) or len(parsed) != 2:
        return Check(False, False, None, "expected two node lists")
    nodes1, nodes2 = parsed
    if len(nodes1) != len(nodes2) or not nodes1 or not nodes2:
        return Check(False, False, None, "the two lists must be non-empty and equal length")
    if any(n not in g1 for n in nodes1) or any(n not in g2 for n in nodes2):
        return Check(False, False, None, "unknown node in one of the lists")
    if not nx.is_isomorphic(g1.subgraph(nodes1), g2.subgraph(nodes2)):
        return Check(False, False, None, "the induced subgraphs are not isomorphic")
    k = len(nodes1)
    if k != truth["size"]:
        return Check(True, False, k,
                     f"common subgraph of size {k}, maximum is {truth['size']}")
    return Check(True, True, k)


# --------------------------------------------------------------------------
# travelling salesman (GraphArena: TSP) - NP-hard
# --------------------------------------------------------------------------

def _tsp_question(p):
    return (
        "Find a shortest tour that visits every node exactly once and returns "
        "to the starting node in this weighted undirected graph. Give the tour "
        "as a list of nodes that starts and ends at the same node, for example: "
        "[a, b, c, a]."
    )


def _tsp_params(G, rng):
    return {}


def _tsp_cost(G, route):
    return sum(G.edges[route[i], route[i + 1]].get("weight", 1)
               for i in range(len(route) - 1))


def _tsp_solve(G, p):
    nodes = sorted(G.nodes, key=str)
    n = len(nodes)
    if n > 12:
        raise ValueError("exact TSP is limited to 12 nodes")
    if n <= 1:
        return {"cost": 0, "route": nodes + nodes}
    start = nodes[0]
    others = nodes[1:]
    # Held-Karp dynamic programming.
    dp = {}
    for j, v in enumerate(others):
        dp[(1 << j, j)] = (G.edges[start, v].get("weight", 1), [start, v])
    for mask in range(1, 1 << len(others)):
        for j in range(len(others)):
            if not (mask & (1 << j)):
                continue
            key = (mask, j)
            if key not in dp:
                continue
            cost, path = dp[key]
            for k, v in enumerate(others):
                if mask & (1 << k):
                    continue
                nkey = (mask | (1 << k), k)
                ncost = cost + G.edges[others[j], v].get("weight", 1)
                if nkey not in dp or ncost < dp[nkey][0]:
                    dp[nkey] = (ncost, path + [v])
    full = (1 << len(others)) - 1
    best_cost, best_path = min(
        (dp[(full, j)][0] + G.edges[others[j], start].get("weight", 1),
         dp[(full, j)][1] + [start])
        for j in range(len(others))
    )
    return {"cost": best_cost, "route": best_path}


def _tsp_check(G, p, truth, nodes):
    if not nodes:
        return Check(False, False, None, "empty tour")
    if len(nodes) != G.number_of_nodes() + 1 or nodes[0] != nodes[-1]:
        return Check(False, False, None, "tour must visit all nodes and return to the start")
    if set(nodes[:-1]) != set(G.nodes):
        return Check(False, False, None, "tour does not visit every node exactly once")
    cost = _tsp_cost(G, nodes)
    if cost != truth["cost"]:
        return Check(True, False, cost, f"tour cost {cost}, optimal is {truth['cost']}")
    return Check(True, True, cost)



# --------------------------------------------------------------------------
# GraphQA simple graph statistics
# --------------------------------------------------------------------------

def _nc_question(p):
    return "How many nodes are in the graph? Answer with a number."


def _nc_params(G, rng):
    return {}


def _nc_solve(G, p):
    return {"count": G.number_of_nodes()}


def _nc_check(G, p, truth, parsed):
    try:
        value = int(parsed)
    except (TypeError, ValueError):
        return Check(False, False, None, "answer is not a number")
    return Check(True, value == truth["count"], value,
                 f"answered {value}, graph has {truth['count']} nodes")


def _ec_question(p):
    return "How many edges are in the graph? Answer with a number."


def _ec_params(G, rng):
    return {}


def _ec_solve(G, p):
    return {"count": G.number_of_edges()}


def _ec_check(G, p, truth, parsed):
    try:
        value = int(parsed)
    except (TypeError, ValueError):
        return Check(False, False, None, "answer is not a number")
    return Check(True, value == truth["count"], value,
                 f"answered {value}, graph has {truth['count']} edges")


def _deg_question(p):
    return f"What is the degree of node {p['node']}? Answer with a number."


def _deg_params(G, rng):
    return {"node": rng.choice(sorted(G.nodes, key=str))}


def _deg_solve(G, p):
    return {"degree": G.degree(p["node"])}


def _deg_check(G, p, truth, parsed):
    try:
        value = int(parsed)
    except (TypeError, ValueError):
        return Check(False, False, None, "answer is not a number")
    return Check(True, value == truth["degree"], value,
                 f"answered {value}, degree is {truth['degree']}")


def _ee_question(p):
    return f"Is node {p['node1']} connected to node {p['node2']} by an edge? Answer yes or no."


def _ee_params(G, rng):
    nodes = sorted(G.nodes, key=str)
    if len(nodes) < 2:
        return {"node1": nodes[0], "node2": nodes[0]}
    return {"node1": rng.choice(nodes), "node2": rng.choice(nodes)}


def _ee_solve(G, p):
    return {"exists": G.has_edge(p["node1"], p["node2"])}


def _ee_check(G, p, truth, parsed):
    if not isinstance(parsed, bool):
        return Check(False, False, None, "answer is not yes/no")
    return Check(True, parsed == truth["exists"], int(parsed),
                 f"answered {parsed}, correct is {truth['exists']}")


def _con_question(p):
    return (
        f"List all nodes connected to node {p['node']} (excluding the node "
        f"itself), in alphabetical order."
    )


def _con_params(G, rng):
    return {"node": rng.choice(sorted(G.nodes, key=str))}


def _con_solve(G, p):
    # GraphQA's "connected to" means directly adjacent neighbours.
    nodes = sorted(set(G.neighbors(p["node"])), key=str)
    return {"nodes": nodes}


def _con_check(G, p, truth, nodes):
    if len(set(nodes)) != len(nodes):
        return Check(False, False, None, "answer repeats a node")
    expected = set(truth["nodes"])
    if any(n not in G for n in nodes):
        return Check(False, False, None, "answer contains an unknown node")
    if set(nodes) != expected:
        return Check(True, False, len(nodes),
                     f"gave {len(nodes)} nodes, expected {len(expected)}")
    return Check(True, True, len(nodes))


def _discon_question(p):
    return (
        f"List all nodes that are not connected to node {p['node']} (excluding "
        f"the node itself), in alphabetical order."
    )


def _discon_params(G, rng):
    return {"node": rng.choice(sorted(G.nodes, key=str))}


def _discon_solve(G, p):
    nodes = sorted(set(G.nodes) - {p["node"]} - set(G.neighbors(p["node"])), key=str)
    return {"nodes": nodes}


def _discon_check(G, p, truth, nodes):
    return _con_check(G, p, truth, nodes)


def _tri_question(p):
    return "How many triangles are in the graph? Answer with a number."


def _tri_params(G, rng):
    return {}


def _tri_solve(G, p):
    return {"count": sum(nx.triangles(G).values()) // 3}


def _tri_check(G, p, truth, parsed):
    try:
        value = int(parsed)
    except (TypeError, ValueError):
        return Check(False, False, None, "answer is not a number")
    return Check(True, value == truth["count"], value,
                 f"answered {value}, graph has {truth['count']} triangles")


def _nclass_question(p):
    labels = p.get("labels") or {}
    known = ", ".join(f"node {n} likes {label}" for n, label in sorted(labels.items(), key=lambda x: str(x[0])))
    return (
        f"Known labels: {known}. Which label belongs to node {p['node']}? "
        "Answer with the label."
    )


def _nclass_params(G, rng):
    return {"node": rng.choice(sorted(G.nodes, key=str)), "labels": {}}


def _nclass_solve(G, p):
    return {"label": str(p.get("answer", p.get("label", "")))}


def _nclass_check(G, p, truth, parsed):
    if parsed is None:
        return Check(False, False, None, "answer is empty")
    value = str(parsed).strip().strip("\"'`*.;: ").lower()
    return Check(True, value == str(truth["label"]).lower(), value,
                 f"answered {value!r}, expected {truth['label']!r}")


def _flow_question(p):
    return (
        f"What is the maximum flow from node {p['source']} to node "
        f"{p['target']}? Answer with a number."
    )


def _flow_params(G, rng):
    nodes = sorted(G.nodes, key=str)
    return {"source": nodes[0], "target": nodes[-1], "answer": 0}


def _flow_solve(G, p):
    return {"flow": p.get("answer")}


def _flow_check(G, p, truth, parsed):
    try:
        value = int(parsed)
    except (TypeError, ValueError):
        return Check(False, False, None, "answer is not a number")
    return Check(True, value == truth["flow"], value,
                 f"answered {value}, expected {truth['flow']}")


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
        Task("common_neighbors", "P", _cn_question, _cn_params, _cn_solve, _cn_check),
        Task("max_independent_set", "NP", _mis_question, _mis_params, _mis_solve, _mis_check),
        Task("min_vertex_cover", "NP", _mvc_question, _mvc_params, _mvc_solve, _mvc_check),
        Task("hamiltonian_path", "NP", _hp_question, _hp_params, _hp_solve, _hp_check),
        Task("graph_edit_distance", "NP", _ged_question, _ged_params, _ged_solve, _ged_check,
             answer_kind="mapping"),
        Task("max_common_subgraph", "NP", _mcs_question, _mcs_params, _mcs_solve, _mcs_check,
             answer_kind="two_lists"),
        Task("traveling_salesman", "NP", _tsp_question, _tsp_params, _tsp_solve, _tsp_check),
        Task("node_count", "P", _nc_question, _nc_params, _nc_solve, _nc_check, answer_kind="number"),
        Task("edge_count", "P", _ec_question, _ec_params, _ec_solve, _ec_check, answer_kind="number"),
        Task("node_degree", "P", _deg_question, _deg_params, _deg_solve, _deg_check, answer_kind="number"),
        Task("edge_existence", "P", _ee_question, _ee_params, _ee_solve, _ee_check, answer_kind="boolean"),
        Task("connected_nodes", "P", _con_question, _con_params, _con_solve, _con_check),
        Task("disconnected_nodes", "P", _discon_question, _discon_params, _discon_solve, _discon_check),
        Task("triangle_counting", "P", _tri_question, _tri_params, _tri_solve, _tri_check, answer_kind="number"),
        Task("node_classification", "P", _nclass_question, _nclass_params, _nclass_solve, _nclass_check, answer_kind="string"),
        Task("maximum_flow", "P", _flow_question, _flow_params, _flow_solve, _flow_check, answer_kind="number"),
    ]
}


# Tasks with a deterministic synthetic generator. Benchmark-only tasks
# (two-graph GED/MCS, weighted TSP, source-label classification/flow) are
# populated by the GraphArena/GraphQA adapters instead.
SYNTHETIC_TASKS = tuple(
    name for name in TASKS
    if name not in {"graph_edit_distance", "max_common_subgraph",
                    "traveling_salesman", "node_classification", "maximum_flow"}
)


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
