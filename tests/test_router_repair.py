import networkx as nx
import pytest
from graphharness.agent import BASELINE_B_V2
from graphharness.models import CallableModel
from graphharness.router import run_decomposed
from graphharness.learned_router import exact_additive
from graphharness.tasks import get_task


def make_record(task, G):
    return {"id": f"{task}/t", "task": task, "source": "synthetic", "split": "t",
            "graph": {"directed": False, "nodes": list(G.nodes),
                      "edges": [list(e) for e in G.edges]},
            "params": {}, "question": "", "answer": get_task(task).solve(G, {}), "meta": {}}


def disconnected():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5)])
    return G


@pytest.mark.parametrize("task", ["max_clique", "max_independent_set", "min_vertex_cover", "connected_components"])
def test_repair_recovers_exact_answer(task):
    G = disconnected()
    rec = make_record(task, G)
    # model always answers with an empty/wrong list; repair must fix each component
    model = CallableModel(lambda messages: "<answer>[]</answer>")
    out = run_decomposed({"baseline_b_v2": BASELINE_B_V2}, model, [rec], max_steps=1,
                         repair=True)
    assert out and out[0]["correct"], out
    assert out[0]["repaired_components"] >= 1
