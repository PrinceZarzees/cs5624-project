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
    counts = convert_dataset(raw, tmp_path / "out", limit=1)
    assert counts == {"train": 3, "validation": 3, "test": 3}
    assert len((tmp_path / "out" / "graphqa_test.jsonl").read_text().splitlines()) == 3
