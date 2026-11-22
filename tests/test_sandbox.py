import json

import pytest

from graphharness.sandbox import PythonSandbox, SandboxLimits


def record():
    return {
        "id": "shortest_path/test/00000",
        "task": "shortest_path",
        "graph": {"directed": False, "nodes": [1, 2, 3, 4], "edges": [[1, 2], [2, 3], [3, 4]]},
        "params": {"source": 1, "target": 4},
        "question": "Find a shortest path.",
        "answer": {"length": 3, "path": [1, 2, 3, 4]},
    }


def test_graph_and_params_are_injected():
    with PythonSandbox(record()) as box:
        out = box.run("print(G.number_of_nodes(), params['source'], nodes)")
    assert out.ok, out.stderr
    assert out.stdout.strip() == "4 1 [1, 2, 3, 4]"


def test_final_expression_is_echoed():
    with PythonSandbox(record()) as box:
        out = box.run("G.number_of_nodes()")
    assert out.ok, out.stderr
    assert out.stdout.strip() == "4"


def test_printed_output_is_not_duplicated():
    with PythonSandbox(record()) as box:
        out = box.run("print('hello')")
    assert out.ok
    assert out.stdout.strip() == "hello"


def test_answer_is_not_written_to_disk():
    with PythonSandbox(record()) as box:
        payload = json.loads((box.workdir / "graph.json").read_text())
    assert set(payload) == {"graph", "params"}
    assert "path" not in payload["params"]


def test_files_persist_between_runs():
    with PythonSandbox(record()) as box:
        assert box.run("open('scratch.txt', 'w').write('hello')").ok
        out = box.run("print(open('scratch.txt').read())")
    assert out.ok
    assert out.stdout.strip() == "hello"


def test_python_errors_are_reported_not_raised():
    with PythonSandbox(record()) as box:
        out = box.run("raise ValueError('bad plan')")
    assert not out.ok
    assert out.returncode != 0
    assert "bad plan" in out.stderr


def test_timeout_kills_runaway_code():
    limits = SandboxLimits(timeout_s=0.5, cpu_seconds=1, memory_mb=512, max_output_bytes=1000)
    with PythonSandbox(record(), limits=limits) as box:
        out = box.run("while True:\n    pass")
    assert out.timed_out
    assert "timed out" in out.observation()


def test_output_is_truncated():
    limits = SandboxLimits(timeout_s=5, cpu_seconds=5, memory_mb=512, max_output_bytes=1000)
    with PythonSandbox(record(), limits=limits) as box:
        out = box.run("print('x' * 5000)")
    assert out.truncated
    assert len(out.stdout) <= 1000


def test_closed_sandbox_rejects_runs():
    box = PythonSandbox(record())
    box.close()
    with pytest.raises(RuntimeError):
        box.run("print(1)")


def test_two_graphs_and_weights_are_injected():
    from graphharness.generators import graph_to_json
    import networkx as nx
    g1 = nx.Graph([(0, 1)])
    g2 = nx.Graph([(0, 1), (1, 2)])
    g2.add_edge(0, 2, weight=7)
    rec = {
        "id": "x", "task": "max_common_subgraph",
        "graph": graph_to_json(g1),
        "graphs": [graph_to_json(g1), graph_to_json(g2)],
        "params": {}, "question": "",
        "answer": {"size": 1, "nodes1": [0], "nodes2": [0]},
    }
    with PythonSandbox(rec) as box:
        out = box.run("print(G1.number_of_nodes(), G2.number_of_nodes(), G2.edges[0, 2]['weight'])")
    assert out.ok, out.stderr
    assert out.stdout.strip() == "2 3 7"
