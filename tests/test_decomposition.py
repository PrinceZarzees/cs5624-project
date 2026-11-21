import random
import networkx as nx
import pytest
from graphharness.learned_router import LearnedRouter, exact_additive


def random_disconnected(seed, comps=4, size=25):
    rng = random.Random(seed)
    G = nx.Graph()
    off = 0
    for _ in range(comps):
        H = nx.gnp_random_graph(size, 0.25, seed=rng.randrange(10**9))
        while not nx.is_connected(H):
            H = nx.gnp_random_graph(size, 0.25, seed=rng.randrange(10**9))
        G.add_nodes_from((v + off) for v in H.nodes)
        G.add_edges_from((u + off, v + off) for u, v in H.edges)
        off += size
    return G


@pytest.mark.parametrize("task", ["max_clique", "max_independent_set", "min_vertex_cover", "connected_components"])
def test_component_merge_is_exact(task):
    for seed in range(20):
        G = random_disconnected(seed)
        comps = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        subs = [exact_additive(task, H) for H in comps]
        glob = exact_additive(task, G)
        if task == "max_clique":
            assert max(s["size"] for s in subs) == glob["size"]
        elif task in {"max_independent_set", "min_vertex_cover"}:
            assert sum(s["size"] for s in subs) == glob["size"]
        else:
            assert len(subs) == nx.number_connected_components(G)


def test_router_roundtrip(tmp_path):
    records = [{"id": str(i), "task": "diameter", "graph": {"directed": False,
               "nodes": list(range(10)), "edges": [[i, i + 1] for i in range(9)]},
               "params": {}} for i in range(6)]
    labels = ["a", "a", "b", "b", "a", "b"]
    r = LearnedRouter.train(records, labels, ["a", "b"])
    p = tmp_path / "r.json"; r.save(p)
    r2 = LearnedRouter.load(p)
    assert r2.predict(records[0]) in {"a", "b"}
