import json

import networkx as nx

from graphharness.graphqa_adapter import convert_item, convert_dataset
from graphharness.evaluate_answer import score_response


def item(question, answer=1):
    return {"question": question, "answer": answer, "algorithm": "er"}


def test_convert_item_uses_current_schema():
    question = (
        "G describes a graph among nodes 0, 1, 2, 3. "
        "The edges in G are: (0, 1) (1, 2) (2, 3). "
        "Q: What is the length of the shortest path from node 0 to node 3?"
    )
    record = convert_item(item(question, 3), "shortest_path", "test", 0)
    assert record["source"] == "graphqa"
    assert record["params"] == {"source": 0, "target": 3}
    assert score_response(record, "<answer>[0, 1, 2, 3]</answer>")["correct"]


def test_convert_dataset_writes_all_requested_splits(tmp_path):
    raw = tmp_path / "raw"
    for task in ("shortest_path", "cycle_check", "reachability"):
        task_dir = raw / task
        task_dir.mkdir(parents=True)
        question = (
            "G describes a graph among nodes 0, 1, 2. "
            "The edges in G are: (0, 1) (1, 2). "
            "Q: Is there a path from node 0 to node 2?"
            if task == "reachability"
            else (
                "G describes a graph among nodes 0, 1, 2. "
                "The edges in G are: (0, 1) (1, 2). "
                "Q: Is there a cycle in this graph?"
                if task == "cycle_check"
                else (
                    "G describes a graph among nodes 0, 1, 2. "
                    "The edges in G are: (0, 1) (1, 2). "
                    "Q: What is the length of the shortest path from node 0 to node 2?"
                )
            )
        )
        for split in ("train", "validation", "test"):
            (task_dir / f"{task}_zero_shot_{split}.json").write_text(
                json.dumps([item(question)])
            )
    counts = convert_dataset(raw, tmp_path / "out",
                             tasks=("shortest_path", "cycle_check", "reachability"), limit=1)
    assert counts == {"train": 3, "validation": 3, "test": 3}
    assert len((tmp_path / "out" / "graphqa_test.jsonl").read_text().splitlines()) == 3


def test_convert_item_new_graphqa_tasks():
    base = ("G describes a graph among nodes 0, 1, 2, 3. "
            "The edges in G are: (0, 1) (1, 2) (2, 3). ")
    cases = [
        ("node_count", "Q: How many nodes are in the graph?", "4.", 4, "<answer>4</answer>"),
        ("edge_count", "Q: How many edges are in the graph?", "3.", 3, "<answer>3</answer>"),
        ("node_degree", "Q: What is the degree of node 2?", "2.", 2, "<answer>2</answer>"),
        ("edge_existence", "Q: Is node 0 connected to node 1?", "Yes.", True, "<answer>yes</answer>"),
        ("connected_nodes", "Q: List all the nodes connected to 1.", "0, 2.", [0, 2], "<answer>[0, 2]</answer>"),
        ("disconnected_nodes", "Q: List all the nodes that are not connected to 1.", "3.", [3], "<answer>[3]</answer>"),
        ("triangle_counting", "Q: How many triangles are in the graph?", "0.", 0, "<answer>0</answer>"),
    ]
    for source_task, tail, source_answer, expected, response in cases:
        rec = convert_item(item(base + tail, source_answer), source_task, "test", 0)
        if source_task in {"node_count", "edge_count"}:
            assert rec["answer"]["count"] == expected
        elif source_task == "node_degree":
            assert rec["answer"]["degree"] == expected
        elif source_task == "edge_existence":
            assert rec["answer"]["exists"] is expected
        elif source_task == "triangle_counting":
            assert rec["answer"]["count"] == expected
        assert score_response(rec, response)["correct"], (source_task, score_response(rec, response))


def test_convert_item_semantic_graphqa_tasks():
    base = ("G describes a graph among nodes 0, 1, 2, 3. "
            "The edges in G are: (0, 1) (1, 2) (2, 3). ")
    nc = base + ("Node 0 likes golf. Node 1 likes golf. Node 2 likes soccer. "
                 "Q: Does node 3 like golf or soccer?")
    rec = convert_item(item(nc, "soccer"), "node_classification", "test", 0)
    assert score_response(rec, "<answer>soccer</answer>")["correct"]
    mf = base + "Q: What is the maximum capacity of the flow from node 0 to node 3?"
    rec = convert_item(item(mf, "3."), "maximum_flow", "test", 0)
    assert score_response(rec, "<answer>3</answer>")["correct"]
