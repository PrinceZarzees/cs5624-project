import json
import pickle
import zipfile

import networkx as nx

from graphharness import build_data
from graphharness.data_loader import load_jsonl, load_split, to_graph
from graphharness.encoding import build_prompt
from graphharness.evaluate_answer import score_response
from graphharness import grapharena_adapter
from graphharness.grapharena_adapter import convert_dataset

TINY_CONFIG = {
    "seed": 7,
    "tasks": ["shortest_path", "connected_components", "cycle_detection", "diameter", "max_clique"],
    "splits": {
        "train": {"n_range": [4, 12], "per_task": 15},
        "val": {"n_range": [4, 12], "per_task": 8},
        "test_medium": {"n_range": [100, 150], "per_task": 3},
    },
}


def build(tmp_path, name="d", extra=()):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps(TINY_CONFIG))
    out = tmp_path / name
    build_data.main(["--config", str(cfg), "--out", str(out), *extra])
    return out


def answer_text(rec):
    a = rec["answer"]
    nodes = {
        "shortest_path": a.get("path"), "diameter": a.get("path"),
        "connected_components": a.get("representatives"),
        "cycle_detection": a.get("cycle"), "max_clique": a.get("clique"),
    }[rec["task"]]
    return "<answer>[" + ", ".join(str(v) for v in nodes) + "]</answer>"


def test_build_is_reproducible(tmp_path):
    a = build(tmp_path, "a")
    b = build(tmp_path, "b")
    ma = json.loads((a / "manifest.json").read_text())
    mb = json.loads((b / "manifest.json").read_text())
    assert ma == mb
    for split in TINY_CONFIG["splits"]:
        assert (a / f"{split}.jsonl").read_bytes() == (b / f"{split}.jsonl").read_bytes()


def test_one_split_alone_matches_full_build(tmp_path):
    full = build(tmp_path, "full")
    only = build(tmp_path, "only", ["--splits", "val"])
    assert (full / "val.jsonl").read_bytes() == (only / "val.jsonl").read_bytes()


def test_records_are_valid(tmp_path):
    out = build(tmp_path)
    for split, spec in TINY_CONFIG["splits"].items():
        recs = load_split(split, out)
        assert len(recs) == spec["per_task"] * len(TINY_CONFIG["tasks"])
        lo, hi = spec["n_range"]
        for r in recs:
            G = to_graph(r)
            assert lo <= G.number_of_nodes() <= hi
            assert r["meta"]["n_edges"] == G.number_of_edges()
            if r["task"] in ("shortest_path", "diameter"):
                assert nx.is_connected(G)
            if r["task"] == "connected_components":
                assert r["answer"]["count"] == r["meta"]["n_components"]
            # the stored true answer must score as correct after a JSON round trip
            s = score_response(r, answer_text(r))
            assert s["correct"], (r["id"], s["note"])
            assert r["question"] in build_prompt(r)


def test_small_splits_do_not_overlap(tmp_path):
    out = build(tmp_path)
    keys = {}
    for split in ("train", "val"):
        for r in load_jsonl(out / f"{split}.jsonl"):
            k = build_data._key(r)
            assert k not in keys, f"{r['id']} duplicates {keys.get(k)}"
            keys[k] = r["id"]


def test_node_labels_are_not_0_to_n(tmp_path):
    out = build(tmp_path)
    recs = load_split("train", out)
    plain = sum(sorted(r["graph"]["nodes"]) == list(range(len(r["graph"]["nodes"]))) for r in recs)
    assert plain < len(recs) * 0.1


def test_grapharena_adapter(tmp_path):
    # a fake GraphArena pickle with named nodes, in GraphArena's layout
    G = nx.Graph()
    names = ["Ann Lee", "Bo Chen", "Cy Park", "Smith, John"]
    for i, n in enumerate(names):
        G.add_node(i, name=n)
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    problems = [{"id": 0, "graph": G, "source": 0, "target": 3, "exact_answer": 3,
                 "problem_text": "original prompt"}]
    (tmp_path / "Distance_easy.pkl").write_bytes(pickle.dumps(problems))
    (tmp_path / "MCP_easy.pkl").write_bytes(pickle.dumps([{"id": 0, "graph": G, "exact_answer": 2}]))
    recs = convert_dataset(tmp_path, "easy")
    assert [r["task"] for r in recs] == ["shortest_path", "max_clique"]
    sp = recs[0]
    assert sp["params"] == {"source": "Ann Lee", "target": "Smith, John"}
    assert sp["answer"]["length"] == sp["meta"]["grapharena_exact_answer"] == 3
    s = score_response(sp, "[Ann Lee, Bo Chen, Cy Park, Smith, John]")
    assert s["correct"], s["note"]


def test_grapharena_from_zip(tmp_path):
    # dataset.zip unpacks to a "dataset/" folder; --zip should find the pickles inside it
    G = nx.path_graph(4)
    nx.set_node_attributes(G, {i: f"node {i}" for i in G}, "name")
    zip_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("dataset/Connected_easy.pkl", pickle.dumps([{"id": 0, "graph": G, "exact_answer": 1}]))
        z.writestr("dataset/Diameter_hard.pkl", pickle.dumps([{"id": 0, "graph": G, "exact_answer": 3}]))
        z.writestr("__MACOSX/dataset/._Connected_easy.pkl", b"junk")
    out = tmp_path / "out"
    grapharena_adapter.main(["--zip", str(zip_path), "--out", str(out)])
    assert (out / "grapharena" / "dataset" / "Connected_easy.pkl").exists()
    easy, hard = load_jsonl(out / "ga_easy.jsonl"), load_jsonl(out / "ga_hard.jsonl")
    assert [r["task"] for r in easy] == ["connected_components"]
    assert [r["task"] for r in hard] == ["diameter"]
    assert hard[0]["answer"]["diameter"] == 3


def test_grapharena_neighbor_mis_mvc_adapter(tmp_path):
    G = nx.Graph()
    names = ["A", "B", "C", "D", "E"]
    for i, n in enumerate(names):
        G.add_node(i, name=n)
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    (tmp_path / "Neighbor_easy.pkl").write_bytes(pickle.dumps([
        {"id": 0, "graph": G, "node1": 0, "node2": 1, "exact_answer": 1}
    ]))
    (tmp_path / "MIS_easy.pkl").write_bytes(pickle.dumps([{"id": 0, "graph": G}]))
    (tmp_path / "MVC_easy.pkl").write_bytes(pickle.dumps([{"id": 0, "graph": G}]))
    recs = convert_dataset(tmp_path, "easy")
    by_task = {r["task"]: r for r in recs}
    assert {"common_neighbors", "max_independent_set", "min_vertex_cover"} <= set(by_task)
    cn = by_task["common_neighbors"]
    assert cn["params"] == {"node1": "A", "node2": "B"}
    assert score_response(cn, "<answer>[C]</answer>")["correct"]
    mis = by_task["max_independent_set"]
    assert score_response(mis, "<answer>[" + ", ".join(mis["answer"]["nodes"]) + "]</answer>")["correct"]
    mvc = by_task["min_vertex_cover"]
    assert score_response(mvc, "<answer>[" + ", ".join(mvc["answer"]["nodes"]) + "]</answer>")["correct"]


def test_grapharena_ged_mcs_tsp_adapter(tmp_path):
    g1 = nx.Graph()
    g1.add_nodes_from((i, {"label": label}) for i, label in enumerate(["C", "O", "N"]))
    g1.add_edges_from([(0, 1), (1, 2)])
    g2 = nx.Graph()
    g2.add_nodes_from((i, {"label": label}) for i, label in enumerate(["C", "O", "N"]))
    g2.add_edges_from([(0, 1), (1, 2)])
    (tmp_path / "GED_easy.pkl").write_bytes(pickle.dumps([
        {"id": 0, "graph": (g1, g2), "exact_answer": 0, "path": [0, 1, 2]}
    ]))
    (tmp_path / "MCS_easy.pkl").write_bytes(pickle.dumps([
        {"id": 0, "graph": (g1, g2), "exact_answer": 3, "path": ([0, 1, 2], [0, 1, 2])}
    ]))
    tsp = nx.complete_graph(4)
    for u, v in tsp.edges:
        tsp.edges[u, v]["weight"] = 1
    (tmp_path / "TSP_easy.pkl").write_bytes(pickle.dumps([
        {"id": 0, "graph": tsp, "exact_answer": 4, "path": [0, 1, 2, 3, 0]}
    ]))
    recs = convert_dataset(tmp_path, "easy")
    by_task = {r["task"]: r for r in recs}
    assert {"graph_edit_distance", "max_common_subgraph", "traveling_salesman"} <= set(by_task)
    ged = by_task["graph_edit_distance"]
    assert score_response(ged, "<answer>[0, 1, 2]</answer>")["correct"]
    mcs = by_task["max_common_subgraph"]
    assert score_response(mcs, "<answer>[0, 1, 2], [0, 1, 2]</answer>")["correct"]
    t = by_task["traveling_salesman"]
    assert score_response(t, "<answer>[0, 1, 2, 3, 0]</answer>")["correct"]
