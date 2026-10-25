import json

import networkx as nx

from graphharness.evaluate_answer import extract_list, main, score_response, summarize, to_nodes
from graphharness.generators import graph_to_json
from graphharness.tasks import get_task


def make_record(G, task_name, params=None, rid="r1"):
    params = params or {}
    t = get_task(task_name)
    return {
        "id": rid, "task": task_name, "split": "test", "source": "synthetic",
        "graph": graph_to_json(G), "params": params,
        "question": t.question(params), "answer": t.solve(G, params), "meta": {},
    }


# ---------- reading the answer ----------

def test_extract_last_list():
    assert extract_list("first [1, 2] then final [3, 4]") == "3, 4"


def test_extract_prefers_answer_tags():
    text = "I think [9, 9]. <answer>[1, 2]</answer> and later [5]"
    assert extract_list(text) == "1, 2"


def test_extract_none():
    assert extract_list("no list here") is None
    assert extract_list(None) is None


def test_extract_empty_list():
    assert extract_list("<answer>[]</answer>") == ""


def test_to_nodes_ints_and_prefixes():
    nodes, unknown = to_nodes(" 12, 'node 7', \"3\" ", [3, 7, 12])
    assert nodes == [12, 7, 3] and unknown == []


def test_to_nodes_unknown():
    nodes, unknown = to_nodes("1, 99", [1, 2])
    assert nodes == [1] and unknown == ["99"]


def test_to_nodes_names_with_commas():
    graph_nodes = ["Smith, John", "Smith", "Ann Lee"]
    nodes, unknown = to_nodes("Smith, John, Ann Lee, Smith", graph_nodes)
    assert nodes == ["Smith, John", "Ann Lee", "Smith"] and unknown == []


def test_to_nodes_case_insensitive():
    nodes, unknown = to_nodes("ann lee", ["Ann Lee"])
    assert nodes == ["Ann Lee"] and unknown == []


# ---------- scoring a full record ----------

def test_score_correct_and_wrong():
    G = nx.cycle_graph(6)
    rec = make_record(G, "shortest_path", {"source": 0, "target": 3})
    good = score_response(rec, "Go 0 to 1 to 2 to 3. <answer>[0, 1, 2, 3]</answer>")
    assert good["parsed"] and good["feasible"] and good["correct"]
    bad = score_response(rec, "<answer>[0, 3]</answer>")
    assert bad["parsed"] and not bad["feasible"]
    made_up = score_response(rec, "[0, 77, 3]")
    assert not made_up["feasible"] and "unknown" in made_up["note"]
    nothing = score_response(rec, "I don't know")
    assert not nothing["parsed"]
    missing = score_response(rec, None)
    assert missing["note"] == "no prediction"


def test_summary_and_cli(tmp_path):
    G = nx.path_graph(4)
    recs = [make_record(G, "cycle_detection", rid="a"), make_record(G, "max_clique", rid="b")]
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in recs))
    preds = tmp_path / "preds.jsonl"
    preds.write_text(json.dumps({"id": "a", "response": "[]"}) + "\n")   # b missing
    out = tmp_path / "scores.jsonl"
    main(["--data", str(data), "--predictions", str(preds), "--out", str(out)])
    scores = [json.loads(line) for line in out.read_text().splitlines()]
    assert [s["correct"] for s in scores] == [True, False]
    rows = {r["task"]: r for r in summarize(scores)}
    assert rows["ALL"]["accuracy"] == 0.5
