"""How many I/O examples (k) make the hidden task identifiable?

For each true task T, draw k examples from the `examples` pool and count the
candidate tasks (T plus plausible distractors with the same input signature)
whose answer matches all k outputs. Identifiable = T is the only survivor.
This is an upper bound on what an ideal inferrer could do from the examples.

    python -m hda.identifiability --data data/hda --trials 300 --ks 1 2 3 5 10
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path
import networkx as nx
from .tasks import TASKS, nx_answer, nx_graph
from .generator import parse_text
from .evaluate import load_split

yn = lambda b: "YES" if b else "NO"

def _und(n, E):
    return nx_graph(n, E, False, False)

DISTRACTORS = {   # signature (directed, weighted, n_query) -> {name: fn(n, edges, query) -> str}
    (False, False, 0): {
        "node_count": lambda n, E, q: str(n),
        "edge_count": lambda n, E, q: str(len(E)),
        "max_degree": lambda n, E, q: str(max((d for _, d in _und(n, E).degree()), default=0)),
        "isolated_nodes": lambda n, E, q: str(sum(d == 0 for _, d in _und(n, E).degree())),
        "leaf_count": lambda n, E, q: str(sum(d == 1 for _, d in _und(n, E).degree())),
        "largest_component": lambda n, E, q: str(max(len(c) for c in nx.connected_components(_und(n, E)))),
        "cycle_rank": lambda n, E, q: str(len(E) - n + nx.number_connected_components(_und(n, E))),
        "bridge_count": lambda n, E, q: str(sum(1 for _ in nx.bridges(_und(n, E)))),
        "is_connected": lambda n, E, q: yn(nx.is_connected(_und(n, E))),
        "is_tree": lambda n, E, q: yn(nx.is_tree(_und(n, E))),
        "has_triangle": lambda n, E, q: yn(sum(nx.triangles(_und(n, E)).values()) > 0),
        "radius_lcc": lambda n, E, q: str(nx.radius(_und(n, E).subgraph(max(nx.connected_components(_und(n, E)), key=len)))),
        "diameter_lcc": lambda n, E, q: str(nx.diameter(_und(n, E).subgraph(max(nx.connected_components(_und(n, E)), key=len)))),
    },
    (False, False, 2): {
        "edge_exists": lambda n, E, q: yn(_und(n, E).has_edge(*q)),
        "hop_distance": lambda n, E, q: str(nx.shortest_path_length(_und(n, E), *q)) if nx.has_path(_und(n, E), *q) else "-1",
        "degree_u": lambda n, E, q: str(_und(n, E).degree(q[0])),
        "degree_v": lambda n, E, q: str(_und(n, E).degree(q[1])),
        "union_neighbors": lambda n, E, q: str(len(set(_und(n, E)[q[0]]) | set(_und(n, E)[q[1]]))),
        "min_degree_uv": lambda n, E, q: str(min(_und(n, E).degree(q[0]), _und(n, E).degree(q[1]))),
    },
    (False, True, 2): {
        "hop_distance": lambda n, E, q: (lambda G: str(nx.shortest_path_length(G, *q)) if nx.has_path(G, *q) else "-1")(nx_graph(n, E, False, True)),
        "edge_weight": lambda n, E, q: (lambda G: str(G[q[0]][q[1]]["weight"]) if G.has_edge(*q) else "-1")(nx_graph(n, E, False, True)),
        "max_edge_on_path": lambda n, E, q: _minimax(n, E, q),
    },
    (False, True, 0): {
        "total_weight": lambda n, E, q: str(sum(e[2] for e in E)),
        "edge_count": lambda n, E, q: str(len(E)),
        "max_weight": lambda n, E, q: str(max((e[2] for e in E), default=0)),
        "mst_max_edge": lambda n, E, q: str(max((d["weight"] for *_, d in nx.minimum_spanning_tree(nx_graph(n, E, False, True)).edges(data=True)), default=0)),
        "max_spanning_tree": lambda n, E, q: str(int(nx.maximum_spanning_tree(nx_graph(n, E, False, True)).size(weight="weight"))),
    },
    (True, False, 2): {
        "reach_t_to_s": lambda n, E, q: yn(nx.has_path(nx_graph(n, E, True, False), q[1], q[0])),
        "undirected_connected": lambda n, E, q: yn(nx.has_path(_und(n, E), *q)),
        "edge_exists": lambda n, E, q: yn(nx_graph(n, E, True, False).has_edge(*q)),
        "strongly_connected": lambda n, E, q: (lambda G: yn(nx.has_path(G, *q) and nx.has_path(G, q[1], q[0])))(nx_graph(n, E, True, False)),
    },
    (True, True, 2): {
        "out_capacity_s": lambda n, E, q: str(sum(e[2] for e in E if e[0] == q[0])),
        "in_capacity_t": lambda n, E, q: str(sum(e[2] for e in E if e[1] == q[1])),
        "min_out_in": lambda n, E, q: str(min(sum(e[2] for e in E if e[0] == q[0]), sum(e[2] for e in E if e[1] == q[1]))),
        "directed_shortest_path": lambda n, E, q: (lambda G: str(nx.dijkstra_path_length(G, *q)) if nx.has_path(G, *q) else "-1")(nx_graph(n, E, True, True)),
        "edge_disjoint_paths": lambda n, E, q: str(int(nx.maximum_flow_value(nx_graph(n, [(a, b) for a, b, _ in E], True, False), *q, capacity="c")) if q[0] != q[1] else "0"),
        "widest_path": lambda n, E, q: _widest(n, E, q),
    },
}

def _minimax(n, E, q):
    G = nx_graph(n, E, False, True)
    if not nx.has_path(G, *q):
        return "-1"
    T = nx.minimum_spanning_tree(G)
    p = nx.shortest_path(T, *q)
    return str(max((T[a][b]["weight"] for a, b in zip(p, p[1:])), default=0))

def _widest(n, E, q):
    import heapq
    adj = defaultdict(list)
    for a, b, w in E:
        adj[a].append((b, w))
    best = {q[0]: float("inf")}; h = [(-float("inf"), q[0])]
    while h:
        c, u = heapq.heappop(h); c = -c
        if c < best.get(u, -1):
            continue
        for v, w in adj[u]:
            nc = min(c, w)
            if nc > best.get(v, -1):
                best[v] = nc; heapq.heappush(h, (-nc, v))
    return "0" if q[0] == q[1] or q[1] not in best else str(int(best[q[1]]))


def candidates(true_task):
    t = TASKS[true_task]
    sig = (t.directed, t.weighted, t.n_query)
    c = {name: (lambda name: lambda n, E, q: nx_answer(name, n, E, q, t.directed, t.weighted))(name)
         for name, o in TASKS.items() if (o.directed, o.weighted, o.n_query) == sig}
    c.update(DISTRACTORS.get(sig, {}))
    return c


def analyse(data: Path, ks, trials, seed=0):
    rng = random.Random(seed); res = {}
    for task in TASKS:
        pool = load_split(data, "examples", task)
        cands = candidates(task)
        table = []   # per example: set of candidate names that match
        for r in pool:
            n, E, q, d, w = parse_text((data / r["input_path"]).read_text())
            ok = set()
            for name, fn in cands.items():
                try:
                    if fn(n, E, q) == r["answer"]:
                        ok.add(name)
                except Exception:
                    pass
            table.append(ok)
        # A distractor that matches EVERY pool example is equivalent on this task's input
        # domain (e.g. diameter vs diameter of the largest component on connected graphs).
        equiv = sorted(x for x in cands if x != task and all(x in ok for ok in table))
        for ok in table:
            if task in ok:
                ok.update(equiv)
        res[task] = dict(n_candidates=len(cands), equivalent_on_domain=equiv)
        for k in ks:
            uniq, conf = 0, defaultdict(int)
            for _ in range(trials):
                s = set(cands)
                for i in rng.sample(range(len(table)), min(k, len(table))):
                    s &= table[i]
                uniq += s == {task} | set(equiv)
                for x in s - {task} - set(equiv):
                    conf[x] += 1
            res[task][f"k={k}"] = round(uniq / trials, 3)
            res[task][f"confusers_k={k}"] = dict(sorted(conf.items(), key=lambda kv: -kv[1])[:3])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hda")
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 5, 10])
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = analyse(Path(a.data), a.ks, a.trials)
    print(f"{'task':22s} cands " + " ".join(f"k={k:<3d}" for k in a.ks))
    for t, r in res.items():
        print(f"{t:22s} {r['n_candidates']:5d} " + " ".join(f"{r[f'k={k}']:.2f} " for k in a.ks)
              + f"  top confusers at k={a.ks[2] if len(a.ks)>2 else a.ks[-1]}: {r[f'confusers_k={a.ks[2] if len(a.ks)>2 else a.ks[-1]}']}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
