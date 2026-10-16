"""Synthetic graph generators.

Three random graph families are used, as in the project plan:
Erdos-Renyi ("er"), Barabasi-Albert ("ba") and Watts-Strogatz ("ws").
Some tasks need a special shape (e.g. several components, or a 50/50 mix of
graphs with and without cycles), so ``make_graph_for_task`` adds that on top.

After a graph is built, its nodes are relabeled with random integer ids and its
edge order is shuffled. This stops the agent (and the harness optimizer) from
learning shortcuts like "the answer nodes are 0..5" on small graphs.
"""

from __future__ import annotations

import random

import networkx as nx

FAMILIES = ("er", "ba", "ws")


def _seed(rng: random.Random) -> int:
    return rng.randrange(2**31)


def base_graph(n: int, family: str, rng: random.Random) -> nx.Graph:
    """A sparse random graph with n nodes from one family."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        G = nx.Graph()
        G.add_node(0)
        return G
    if family == "er":
        avg_degree = rng.uniform(1.5, 4.0)
        p = min(0.5, avg_degree / (n - 1))  # cap keeps tiny graphs from being complete
        return nx.fast_gnp_random_graph(n, p, seed=_seed(rng))
    if family == "ba":
        m = rng.randint(1, min(3, n - 1))
        return nx.barabasi_albert_graph(n, m, seed=_seed(rng))
    if family == "ws":
        k = rng.choice([2, 4, 6])
        half = n // 2
        k = min(k, max(2, half - half % 2))  # even, and small enough to stay sparse
        if k >= n:
            return nx.path_graph(n)
        p = rng.uniform(0.05, 0.3)
        return nx.watts_strogatz_graph(n, k, p, seed=_seed(rng))
    raise ValueError(f"unknown family {family!r}")


def ensure_connected(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Join all components into one by adding one random edge between them."""
    comps = [sorted(c) for c in nx.connected_components(G)]
    rng.shuffle(comps)
    for a, b in zip(comps, comps[1:]):
        G.add_edge(rng.choice(a), rng.choice(b))
    return G


def random_tree(n: int, rng: random.Random) -> nx.Graph:
    """A uniformly random tree on n nodes (random Pruefer sequence)."""
    if n == 1:
        G = nx.Graph()
        G.add_node(0)
        return G
    if n == 2:
        return nx.path_graph(2)
    seq = [rng.randrange(n) for _ in range(n - 2)]
    return nx.from_prufer_sequence(seq)


def graph_with_components(n: int, n_comps: int, rng: random.Random) -> nx.Graph:
    """A graph with exactly ``n_comps`` connected components (n_comps <= n)."""
    n_comps = max(1, min(n_comps, n))
    # split n into n_comps positive parts
    cuts = sorted(rng.sample(range(1, n), n_comps - 1)) if n_comps > 1 else []
    sizes = [b - a for a, b in zip([0] + cuts, cuts + [n])]
    G = nx.Graph()
    offset = 0
    for size in sizes:
        part = ensure_connected(base_graph(size, rng.choice(FAMILIES), rng), rng)
        G.add_edges_from((u + offset, v + offset) for u, v in part.edges)
        G.add_nodes_from(v + offset for v in part.nodes)
        offset += size
    return G


def plant_clique(G: nx.Graph, k: int, rng: random.Random) -> nx.Graph:
    nodes = rng.sample(sorted(G.nodes), k)
    G.add_edges_from((a, b) for i, a in enumerate(nodes) for b in nodes[i + 1:])
    return G


def relabel_randomly(G: nx.Graph, rng: random.Random) -> nx.Graph:
    """Give nodes random distinct integer ids in [0, 10*n)."""
    n = G.number_of_nodes()
    new_ids = rng.sample(range(10 * max(n, 1)), n)
    old = sorted(G.nodes)
    H = nx.relabel_nodes(G, dict(zip(old, new_ids)), copy=True)
    return H


def make_graph_for_task(task: str, n: int, rng: random.Random) -> tuple[nx.Graph, dict]:
    """Build a graph shaped for ``task``. Returns (graph, info about how it was made)."""
    family = rng.choice(FAMILIES)
    info: dict = {"family": family}

    if task in ("shortest_path", "diameter"):
        G = ensure_connected(base_graph(n, family, rng), rng)

    elif task == "connected_components":
        n_comps = rng.randint(1, max(1, min(12, n // 3)))
        G = graph_with_components(n, n_comps, rng)
        info = {"family": "mixed", "n_components": n_comps}

    elif task == "cycle_detection":
        # half the graphs have no cycle (a tree); the other half are a tree
        # with a few extra edges, so the cycle is hidden in a mostly tree-like graph
        G = random_tree(n, rng)
        extra = 0
        if n >= 3 and rng.random() < 0.5:
            extra = rng.randint(1, 3)
            tries = 0
            while extra > 0 and tries < 100:
                a, b = rng.sample(range(n), 2)
                if not G.has_edge(a, b):
                    G.add_edge(a, b)
                    extra -= 1
                tries += 1
        info = {"family": "tree", "extra_edges": G.number_of_edges() - (n - 1)}

    elif task == "max_clique":
        G = base_graph(n, family, rng)
        if n >= 3 and rng.random() < 0.5:
            k = rng.randint(3, min(n, 10))
            plant_clique(G, k, rng)
            info["planted_clique"] = k

    else:
        G = base_graph(n, family, rng)

    return relabel_randomly(G, rng), info


def graph_to_json(G: nx.Graph, rng: random.Random | None = None) -> dict:
    """Store a graph as plain lists. Edge order is shuffled if rng is given."""
    edges = [[u, v] for u, v in G.edges]
    if rng is not None:
        rng.shuffle(edges)
        for e in edges:
            if rng.random() < 0.5:
                e.reverse()
    return {"directed": False, "nodes": sorted(G.nodes, key=str), "edges": edges}


def graph_from_json(g: dict) -> nx.Graph:
    G = nx.DiGraph() if g.get("directed") else nx.Graph()
    G.add_nodes_from(g["nodes"])
    G.add_edges_from(tuple(e) for e in g["edges"])
    return G
