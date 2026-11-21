import random

import networkx as nx

from graphharness.agent import BASELINE_A
from graphharness.generators import graph_to_json
from graphharness.models import ScriptedModel
from graphharness.router import FeatureRouter, run_decomposed, run_routed
from graphharness.tasks import get_task


def test_feature_router_prefers_class_specific_harness():
    library = {
        "baseline_b_v2": BASELINE_A,
        "dense": BASELINE_A.copy(name="dense"),
        "large": BASELINE_A.copy(name="large"),
    }
    router = FeatureRouter(library)
    G = nx.complete_graph(10)
    rec = {"id": "x", "task": "max_clique", "graph": graph_to_json(G),
           "params": {}, "question": "", "answer": {"size": 10, "clique": list(G.nodes)},
           "meta": {}}
    assert router.choose(rec) == "dense"
    rec["graph"] = graph_to_json(nx.path_graph(600))
    assert router.choose(rec) == "large"


def test_routed_run_scores_a_simple_answer():
    G = nx.path_graph(4)
    task = get_task("shortest_path")
    rec = {"id": "sp", "task": "shortest_path", "split": "test",
           "graph": graph_to_json(G), "params": {"source": 0, "target": 3},
           "question": task.question({"source": 0, "target": 3}),
           "answer": {"length": 3, "path": [0, 1, 2, 3]}, "meta": {}}
    scores = run_routed({"baseline_b_v2": BASELINE_A},
                        ScriptedModel(["<answer>[0, 1, 2, 3]</answer>"]), [rec])
    assert scores[0]["correct"]


def test_decomposed_max_clique_uses_component_answers():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)])
    task = get_task("max_clique")
    truth = task.solve(G, {})
    rec = {"id": "mc", "task": "max_clique", "split": "test",
           "graph": graph_to_json(G), "params": {}, "question": task.question({}),
           "answer": truth, "meta": {}}
    model = ScriptedModel(["<answer>[0, 1, 2]</answer>", "<answer>[3, 4, 5]</answer>"])
    scores = run_decomposed({"baseline_b_v2": BASELINE_A}, model, [rec])
    assert scores[0]["correct"]
    assert scores[0]["decomposition"] == "components"
