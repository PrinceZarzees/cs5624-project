#!/usr/bin/env python3
"""Create the E10 language x library harness variants."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphharness.agent import BASELINE_B_V2  # noqa: E402

OUT = ROOT / "experiments/final/E10_lang_lib/harnesses"
OUT.mkdir(parents=True, exist_ok=True)

PY_NOLIB_RULES = [
    "Use `adj` and `edge_list` directly; never infer edges or distances from node ids.",
    "Implement BFS, DFS or union-find yourself; NetworkX is not installed and imports fail.",
    "For maximum clique, enumerate only on small graphs; on large graphs use a heuristic and verify pairwise adjacency.",
    "Check feasibility before answering: path endpoints and edges, one representative per component, cycle closure, or clique pairwise adjacency.",
    "Keep sandbox output short: print only the result, its length/size, and a small check.",
    "If code times out or errors, change the algorithm instead of rerunning it unchanged.",
    "Put only the final node list inside <answer></answer>; use [] for a negative answer.",
    "For connected components, run BFS/DFS and return one node per component.",
    "For diameter, run BFS from each node, find a farthest pair, and return a shortest path between them.",
    "Always call print(...) on the value you need.",
    "For maximum clique on small graphs, use branch and bound and verify the result.",
    "Once a successful run prints a feasible result, stop and give that result in <answer> tags.",
    "Never filter, sort, or index nodes by numeric value; node ids are arbitrary.",
    "Do not mutate `adj` while iterating it; copy neighbour lists if needed.",
]

C_LIB_RULES = [
    "Use integer node ids and `load_graph()` from graph_io.h; the graph is in graph.txt.",
    "Prefer graph_algos.h: ga_components, ga_has_cycle, ga_diameter, ga_max_clique, ga_max_independent_set, ga_min_vertex_cover.",
    "For maximum clique, call ga_max_clique and verify that every returned pair is adjacent.",
    "Check feasibility before answering: path endpoints and edges, one representative per component, cycle closure, or clique pairwise adjacency.",
    "Keep program output short: print only the result, its length/size, and a small check.",
    "If compilation or execution fails, change the program instead of rerunning it unchanged.",
    "Put only the final node list inside <answer></answer>; use [] for a negative answer.",
    "For connected components, call ga_components and print one node per component.",
    "For diameter, call ga_diameter and print a witness pair; do not guess.",
    "Always print the value you need with printf.",
    "Include both \"graph_io.h\" and \"graph_algos.h\".",
    "Once a successful run prints a feasible result, stop and give that result in <answer> tags.",
    "Never sort or index nodes by numeric value when the ids are arbitrary; here ids are 0..n-1.",
    "Free the graph with free_graph before exiting.",
]


def main():
    py = BASELINE_B_V2.copy(name="baseline_b_v2_py_nolib", rules=PY_NOLIB_RULES,
                            block_networkx=True, notes="v2 rules rewritten for adjacency only; NetworkX blocked.")
    py.save(OUT / "baseline_b_v2_py_nolib.json")
    c = BASELINE_B_V2.copy(name="baseline_b_v2_c_lib", rules=C_LIB_RULES,
                           tool_backend="c", c_library=True, allow_tools=True,
                           notes="C backend with graph_algos library.")
    c.save(OUT / "baseline_b_v2_c_lib.json")
    for h in (py, c):
        msg = h.system_message()
        print(h.name, "rules", len(h.rules), "block_nx", h.block_networkx,
              "c_lib", h.c_library, "msg_chars", len(msg))
        assert "no NetworkX" in msg or "C graph library" in msg


if __name__ == "__main__":
    main()
