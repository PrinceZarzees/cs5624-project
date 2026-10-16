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


# ---------- the true answer is always accepted ----------

def truth_nodes(task_name, truth):
    return {
        "shortest_path": lambda t: t["path"],
        "connected_components": lambda t: t["representatives"],
        "cycle_detection": lambda t: t["cycle"],
        "diameter": lambda t: t["path"],
        "max_clique": lambda t: t["clique"],
    }[task_name](truth)


@pytest.mark.parametrize("task_name", sorted(TASKS))
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
