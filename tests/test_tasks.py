"""Check the exact solvers against brute force, and the answer checkers on
right and wrong answers."""

import itertools
import random

import networkx as nx
import pytest

from graphharness.generators import make_graph_for_task
from graphharness.tasks import TASKS, get_task


def small_graphs(task, count=40, seed=0, lo=4, hi=10):
    rng = random.Random(seed)
    for _ in range(count):
        G, _ = make_graph_for_task(task, rng.randint(lo, hi), rng)
        yield G, rng


# ---------- solvers agree with brute force ----------

def test_max_clique_matches_brute_force():
    task = get_task("max_clique")
    for G, rng in small_graphs("max_clique"):
        best = 0
        nodes = list(G.nodes)
        for k in range(1, len(nodes) + 1):
            for sub in itertools.combinations(nodes, k):
                if all(G.has_edge(a, b) for a, b in itertools.combinations(sub, 2)):
                    best = k
                    break
        assert task.solve(G, {})["size"] == best


def test_cycle_detection_matches_edge_count():
    # a graph has no cycle exactly when it is a forest: edges = nodes - components
    task = get_task("cycle_detection")
    seen = set()
    for G, rng in small_graphs("cycle_detection", count=60):
        forest = G.number_of_edges() == G.number_of_nodes() - nx.number_connected_components(G)
        assert task.solve(G, {})["has_cycle"] == (not forest)
        seen.add(forest)
    assert seen == {True, False}, "generator should make graphs with and without cycles"


def test_diameter_matches_all_pairs():
    task = get_task("diameter")
    for G, rng in small_graphs("diameter"):
        d = max(max(dist.values()) for _, dist in nx.all_pairs_shortest_path_length(G))
        assert task.solve(G, {})["diameter"] == d


def test_components_count():
    task = get_task("connected_components")
    for G, rng in small_graphs("connected_components", lo=4, hi=40):
        assert task.solve(G, {})["count"] == nx.number_connected_components(G)


def test_disconnected_shortest_path_is_scored_correctly():
    task = get_task("shortest_path")
    G = nx.Graph([(1, 2)])
    G.add_nodes_from([3, 4])
    params = {"source": 1, "target": 4}
    truth = task.solve(G, params)
    assert truth == {"length": -1, "path": []}
    assert task.check(G, params, truth, []).correct


def test_reachability_task():
    task = get_task("reachability")
    G = nx.path_graph(4)
    params = {"source": 0, "target": 3}
    truth = task.solve(G, params)
    assert truth["reachable"]
    assert task.check(G, params, truth, [0, 1, 2, 3]).correct


# ---------- the true answer is always accepted ----------

def truth_nodes(task_name, truth):
    return {
        "shortest_path": lambda t: t["path"],
        "reachability": lambda t: t["path"],
        "connected_components": lambda t: t["representatives"],
        "cycle_detection": lambda t: t["cycle"],
        "diameter": lambda t: t["path"],
        "max_clique": lambda t: t["clique"],
        "common_neighbors": lambda t: t["nodes"],
        "max_independent_set": lambda t: t["nodes"],
        "min_vertex_cover": lambda t: t["nodes"],
        "hamiltonian_path": lambda t: t["path"],
        "node_count": lambda t: t["count"],
        "edge_count": lambda t: t["count"],
        "node_degree": lambda t: t["degree"],
        "edge_existence": lambda t: t["exists"],
        "connected_nodes": lambda t: t["nodes"],
        "disconnected_nodes": lambda t: t["nodes"],
        "triangle_counting": lambda t: t["count"],
    }[task_name](truth)


MULTI_GRAPH = {"graph_edit_distance", "max_common_subgraph", "traveling_salesman",
               "node_classification", "maximum_flow"}


@pytest.mark.parametrize("task_name", [t for t in sorted(TASKS) if t not in MULTI_GRAPH])
def test_truth_is_correct(task_name):
    task = get_task(task_name)
    for G, rng in small_graphs(task_name, lo=4, hi=30):
        params = task.make_params(G, rng)
        truth = task.solve(G, params)
        c = task.check(G, params, truth, truth_nodes(task_name, truth))
        assert c.feasible and c.correct, c.note


# ---------- wrong answers are rejected ----------

@pytest.fixture
def path_graph():
    # 0-1-2-3-4 plus a shortcut 0-2 and a separate edge 10-11
    G = nx.path_graph(5)
    G.add_edge(0, 2)
    G.add_edge(10, 11)
    return G


def test_shortest_path_checks(path_graph):
    t = get_task("shortest_path")
    p = {"source": 0, "target": 4}
    truth = t.solve(path_graph, p)
    assert truth["length"] == 3
    assert t.check(path_graph, p, truth, [0, 2, 3, 4]).correct
    longer = t.check(path_graph, p, truth, [0, 1, 2, 3, 4])
    assert longer.feasible and not longer.correct
    assert not t.check(path_graph, p, truth, [0, 3, 4]).feasible       # 0-3 is not an edge
    assert not t.check(path_graph, p, truth, [2, 3, 4]).feasible       # wrong start
    assert not t.check(path_graph, p, truth, []).feasible


def test_components_checks(path_graph):
    t = get_task("connected_components")
    truth = t.solve(path_graph, {})
    assert truth["count"] == 2
    assert t.check(path_graph, {}, truth, [3, 11]).correct
    assert not t.check(path_graph, {}, truth, [0, 1]).feasible        # same component
    missing = t.check(path_graph, {}, truth, [0])
    assert missing.feasible and not missing.correct


def test_cycle_checks(path_graph):
    t = get_task("cycle_detection")
    truth = t.solve(path_graph, {})
    assert truth["has_cycle"]
    assert t.check(path_graph, {}, truth, [0, 1, 2]).correct
    assert t.check(path_graph, {}, truth, [0, 1, 2, 0]).correct       # first node repeated
    assert not t.check(path_graph, {}, truth, [0, 1, 2, 3]).feasible  # 3-0 not an edge
    assert not t.check(path_graph, {}, truth, []).correct             # said "no cycle"
    tree = nx.path_graph(4)
    tree_truth = t.solve(tree, {})
    assert t.check(tree, {}, tree_truth, []).correct


def test_diameter_checks():
    t = get_task("diameter")
    G = nx.path_graph(5)
    G.add_edge(0, 2)          # distance 0..4 is now 3
    truth = t.solve(G, {})
    assert truth["diameter"] == 3
    assert t.check(G, {}, truth, [0, 2, 3, 4]).correct
    assert not t.check(G, {}, truth, [0, 1, 2, 3, 4]).feasible        # not a shortest path
    short = t.check(G, {}, truth, [1, 2, 3])
    assert short.feasible and not short.correct


def test_max_clique_checks():
    t = get_task("max_clique")
    G = nx.complete_graph(4)
    G.add_edge(3, 4)
    truth = t.solve(G, {})
    assert truth["size"] == 4
    assert t.check(G, {}, truth, [0, 1, 2, 3]).correct
    assert not t.check(G, {}, truth, [0, 1, 4]).feasible
    small = t.check(G, {}, truth, [3, 4])
    assert small.feasible and not small.correct


def test_common_neighbors_check():
    t = get_task("common_neighbors")
    G = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
    p = {"node1": 0, "node2": 1}
    truth = t.solve(G, p)
    assert truth["nodes"] == [2]
    assert t.check(G, p, truth, [2]).correct
    assert not t.check(G, p, truth, [2, 3]).correct
    assert not t.check(G, p, truth, [3]).feasible


def test_mis_and_mvc_brute_force():
    mis, mvc = get_task("max_independent_set"), get_task("min_vertex_cover")
    for G, _ in small_graphs("max_independent_set", count=15):
        nodes = list(G.nodes)
        best = 0
        for k in range(1, len(nodes) + 1):
            for sub in itertools.combinations(nodes, k):
                if all(not G.has_edge(a, b) for a, b in itertools.combinations(sub, 2)):
                    best = max(best, k)
        assert mis.solve(G, {})["size"] == best
        assert mvc.solve(G, {})["size"] == len(nodes) - best


def test_hamiltonian_path_known_cases():
    t = get_task("hamiltonian_path")
    path = nx.path_graph(5)
    truth = t.solve(path, {})
    assert truth["exists"]
    assert t.check(path, {}, truth, truth["path"]).correct
    disconnected = nx.Graph([(0, 1), (2, 3)])
    truth = t.solve(disconnected, {})
    assert not truth["exists"]
    assert t.check(disconnected, {}, truth, []).correct
    assert not t.check(disconnected, {}, truth, [0, 1]).feasible


def _labeled_graph(edges, labels):
    G = nx.Graph()
    G.add_nodes_from((i, {"label": label}) for i, label in enumerate(labels))
    G.add_edges_from(edges)
    return G


def test_ged_known_mapping():
    t = get_task("graph_edit_distance")
    g1 = _labeled_graph([(0, 1), (1, 2)], ["C", "O", "N"])
    g2 = _labeled_graph([(0, 1), (1, 2)], ["C", "O", "N"])
    truth = t.solve((g1, g2), {})
    assert truth["cost"] == 0
    assert t.check((g1, g2), {}, truth, truth["mapping"]).correct
    assert t.check((g1, g2), {}, truth, 0).correct
    assert not t.check((g1, g2), {}, truth, 99).correct


def test_mcs_known_subgraph():
    t = get_task("max_common_subgraph")
    g1 = nx.path_graph(4)
    g2 = nx.cycle_graph(4)
    truth = t.solve((g1, g2), {})
    assert truth["size"] == 3
    assert t.check((g1, g2), {}, truth, [truth["nodes1"], truth["nodes2"]]).correct


def test_tsp_known_tour():
    t = get_task("traveling_salesman")
    G = nx.complete_graph(4)
    for u, v in G.edges:
        G.edges[u, v]["weight"] = 1
    truth = t.solve(G, {})
    assert truth["cost"] == 4
    assert t.check(G, {}, truth, [0, 1, 2, 3, 0]).correct
    assert t.check(G, {}, truth, [0, 1, 2, 3]).feasible is False


def test_graphqa_simple_scalar_tasks():
    G = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
    cases = {
        "node_count": (4, "4"),
        "edge_count": (4, "4"),
        "triangle_counting": (1, "1"),
    }
    for name, (value, text) in cases.items():
        t = get_task(name)
        truth = t.solve(G, {})
        assert truth[list(truth)[0]] == value
        assert t.check(G, {}, truth, value).correct

    deg = get_task("node_degree")
    truth = deg.solve(G, {"node": 2})
    assert truth["degree"] == 3
    assert deg.check(G, {"node": 2}, truth, 3).correct

    ee = get_task("edge_existence")
    assert ee.check(G, {"node1": 0, "node2": 1}, ee.solve(G, {"node1": 0, "node2": 1}), True).correct
    assert not ee.check(G, {"node1": 0, "node2": 3}, ee.solve(G, {"node1": 0, "node2": 3}), True).correct

    con = get_task("connected_nodes")
    p = {"node": 2}
    truth = con.solve(G, p)
    assert set(truth["nodes"]) == {0, 1, 3}
    assert con.check(G, p, truth, truth["nodes"]).correct

    dis = get_task("disconnected_nodes")
    G2 = nx.Graph([(0, 1), (2, 3)])
    p = {"node": 0}
    truth = dis.solve(G2, p)
    assert set(truth["nodes"]) == {2, 3}
    assert dis.check(G2, p, truth, truth["nodes"]).correct
