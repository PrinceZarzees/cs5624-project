from graphharness.agent import BASELINE_A, run_agent
from graphharness.csandbox import CSandbox
from graphharness.evaluate_answer import score_response
from graphharness.models import ScriptedModel


def record():
    return {
        "id": "shortest_path/test/0", "task": "shortest_path", "split": "test",
        "graph": {"directed": False, "nodes": [0, 1, 2], "edges": [[0, 1], [1, 2]]},
        "params": {"source": 0, "target": 2},
        "question": "Find a shortest path from node 0 to node 2.",
        "answer": {"length": 2, "path": [0, 1, 2]}, "meta": {},
    }


def test_c_sandbox_graph_io():
    code = r'''
#include "graph_io.h"
int main() {
    Graph g = load_graph();
    printf("%d %d %s\n", g.n, g.m, get_param("source"));
    free_graph(&g);
    return 0;
}
'''
    with CSandbox(record()) as box:
        result = box.run(code)
    assert result.ok, result.stderr
    assert result.stdout.strip() == "3 2 0"


def test_c_agent_tool_loop():
    h = BASELINE_A.copy(tool_backend="c", graph_view="summary")
    code = r'''
#include "graph_io.h"
int main() {
    Graph g = load_graph();
    printf("nodes=%d edges=%d\n", g.n, g.m);
    free_graph(&g);
    return 0;
}
'''
    model = ScriptedModel(["```c\n" + code + "\n```", "<answer>[0, 1, 2]</answer>"])
    result = run_agent(h, model, record())
    assert result.tool_calls == 1
    assert "nodes=3" in result.steps[0]["stdout"]
    assert score_response(record(), result.response)["correct"]
