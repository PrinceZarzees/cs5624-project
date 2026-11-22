import json
import networkx as nx
import pytest
from graphharness.sandbox import PythonSandbox, SandboxLimits
from graphharness.csandbox import CSandbox


def record(edges, nodes):
    return {"id": "t", "task": "connected_components", "params": {},
            "graph": {"directed": False, "nodes": nodes,
                      "edges": [list(e) for e in edges]}}


def test_python_no_networkx():
    rec = record([(0, 1), (1, 2)], [0, 1, 2])
    with PythonSandbox(rec, SandboxLimits(timeout_s=20), block_networkx=True) as sb:
        r = sb.run("import networkx")
        assert not r.ok and "NetworkX" in (r.stderr or "")
        r = sb.run("print(len(adj[1]))")
        assert r.stdout.strip() == "2"


def test_c_library_matches_networkx():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)]
    rec = record(edges, list(range(6)))
    G = nx.Graph(edges)
    code = r'''
#include "graph_io.h"
#include "graph_algos.h"
int main() {
    Graph g = load_graph();
    int comp[16]; int c = ga_components(&g, comp);
    int cl[16]; int k = ga_max_clique(&g, cl, 16);
    int is[16]; int a = ga_max_independent_set(&g, is, 16);
    printf("components %d clique %d mis %d cycle %d\\n", c, k, a, ga_has_cycle(&g));
    free_graph(&g);
    return 0;
}
'''
    with CSandbox(rec, SandboxLimits(timeout_s=20), library=True) as sb:
        r = sb.run(code)
    assert r.ok, r.stderr
    out = r.stdout.strip()
    assert "components 2" in out
    assert "clique 2" in out
    assert "mis 3" in out
    assert "cycle 1" in out
