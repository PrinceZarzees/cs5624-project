"""Phase-1 task registry: polynomial-time graph tasks with exact, unique answers.

Each task defines the input signature (directed/weighted/#query args), the
statement shown to the HDA in the "statement given" condition, and a NetworkX
reference used only by the evaluator (never by the HDA). The C++ files in
reference/ are the fast ground truth; tests/test_reference.py cross-checks them.
"""
from __future__ import annotations
from dataclasses import dataclass
import networkx as nx


@dataclass(frozen=True)
class Task:
    name: str
    directed: bool
    weighted: bool
    n_query: int
    needs_connected: bool
    statement: str
    max_n: int = 100_000          # largest graph used in the stress split


IO_NOTE = ("Print exactly one line. YES/NO answers are upper case; numeric answers "
           "are plain integers.")

TASKS: dict[str, Task] = {t.name: t for t in [
    Task("shortest_path", False, True, 2, False,
         "Given an undirected graph with positive integer edge weights and two nodes s and t. "
         "The query line is `2 s t`: k=2 is the NUMBER of arguments (not a type code), "
         "then s and t. Print the total weight of a minimum-weight path from s to t, or -1 "
         "if t is unreachable from s."),
    Task("reachability", True, False, 2, False,
         "Given a directed graph and two nodes s and t. The query line is `2 s t`: k=2 is the "
         "NUMBER of arguments (not a type code), then s and t. Print YES if there is a directed "
         "path from s to t, otherwise NO."),
    Task("connected_components", False, False, 0, False,
         "Given an undirected graph, print the number of connected components "
         "(an isolated node is its own component)."),
    Task("cycle_detection", False, False, 0, False,
         "Given an undirected simple graph, print YES if it contains at least one cycle, otherwise NO."),
    Task("diameter", False, False, 0, True,
         "Given a connected undirected unweighted graph, print its diameter: the largest "
         "shortest-path distance (in edges) between any two nodes.", max_n=5_000),
    Task("common_neighbors", False, False, 2, False,
         "Given an undirected graph and two nodes u and v. The query line is `2 u v`: k=2 is the "
         "NUMBER of arguments (not a type code), then u and v. Print the number of nodes "
         "adjacent to both u and v."),
    Task("triangle_count", False, False, 0, False,
         "Given an undirected simple graph, print the number of triangles "
         "(unordered triples of mutually adjacent nodes)."),
    Task("bipartite_check", False, False, 0, False,
         "Given an undirected graph, print YES if its nodes can be split into two sets so that "
         "every edge joins the two sets (the graph is bipartite), otherwise NO."),
    Task("mst_weight", False, True, 0, True,
         "Given a connected undirected graph with positive integer edge weights, print the "
         "total weight of a minimum spanning tree."),
    Task("max_flow", True, True, 2, False,
         "Given a directed graph whose edge weights are capacities and two nodes s and t "
         "(query 2 s t), print the value of a maximum flow from s to t.", max_n=20_000),
]}


def nx_graph(n, edges, directed, weighted):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(n))
    for e in edges:
        if weighted:
            G.add_edge(e[0], e[1], weight=e[2])
        else:
            G.add_edge(e[0], e[1])
    return G


def nx_answer(task: str, n, edges, query, directed, weighted) -> str:
    """Slow reference (NetworkX). Used for cross-checking only."""
    G = nx_graph(n, edges, directed, weighted)
    yn = lambda b: "YES" if b else "NO"
    if task == "shortest_path":
        s, t = query
        try:
            return str(nx.dijkstra_path_length(G, s, t))
        except nx.NetworkXNoPath:
            return "-1"
    if task == "reachability":
        return yn(nx.has_path(G, query[0], query[1]))
    if task == "connected_components":
        return str(nx.number_connected_components(G))
    if task == "cycle_detection":
        return yn(G.number_of_edges() > n - nx.number_connected_components(G))
    if task == "diameter":
        return str(nx.diameter(G))
    if task == "common_neighbors":
        return str(len(set(G[query[0]]) & set(G[query[1]])))
    if task == "triangle_count":
        return str(sum(nx.triangles(G).values()) // 3)
    if task == "bipartite_check":
        return yn(nx.is_bipartite(G))
    if task == "mst_weight":
        return str(int(nx.minimum_spanning_tree(G).size(weight="weight")))
    if task == "max_flow":
        s, t = query
        if s == t:
            return "0"
        return str(int(nx.maximum_flow_value(G, s, t, capacity="weight")))
    raise KeyError(task)
