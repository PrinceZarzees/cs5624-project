"""Test generator for the HDA benchmark (phase 1, polynomial-time tasks).

Produces graphs of controlled size, density, family and task-specific
properties, relabels nodes with a random permutation, writes each instance as
a plain-text input file, and computes the answer with the C++ reference
solutions in reference/ (cross-checked against NetworkX by the unit tests).

Usage:
    python -m hda.generator --config configs/splits.json --out data/hda
"""
from __future__ import annotations
import argparse, json, math, os, random, subprocess, time
from pathlib import Path
import networkx as nx
from .tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent
SEEN_FAMILIES = ["er", "ba", "tree", "components"]
UNSEEN_FAMILIES = ["ws", "grid", "path_like", "dense", "regular", "powerlaw"]
ALL_FAMILIES = SEEN_FAMILIES + UNSEEN_FAMILIES
DENSITY_BINS = {"sparse": (1.2, 3.0), "medium": (3.0, 8.0), "dense": (8.0, 20.0)}


# ---------------------------------------------------------------- base graphs
def base_graph(family: str, n: int, avg_deg: float, rng: random.Random) -> nx.Graph:
    s = rng.randrange(2**31)
    avg_deg = min(avg_deg, n - 1)
    if family == "er":
        G = nx.fast_gnp_random_graph(n, avg_deg / max(1, n - 1), seed=s)
    elif family == "dense":
        G = nx.fast_gnp_random_graph(n, rng.uniform(0.3, 0.6), seed=s)
    elif family == "ba":
        G = nx.barabasi_albert_graph(n, max(1, min(n - 1, round(avg_deg / 2))), seed=s)
    elif family == "powerlaw":
        G = nx.powerlaw_cluster_graph(n, max(1, min(n - 1, round(avg_deg / 2))), 0.3, seed=s)
    elif family == "ws":
        k = max(2, min(n - 1 - ((n - 1) % 2), 2 * round(avg_deg / 2)))
        G = nx.watts_strogatz_graph(n, k, rng.uniform(0.05, 0.3), seed=s)
    elif family == "regular":
        d = max(1, min(n - 1, round(avg_deg)))
        if (d * n) % 2:
            d = d - 1 if d > 1 else d + 1
        G = nx.random_regular_graph(d, n, seed=s) if 0 < d < n else nx.empty_graph(n)
    elif family == "tree":
        G = nx.Graph(); G.add_nodes_from(range(n))
        for i in range(1, n):   # mix of bushy (recursive) and long (window) trees
            G.add_edge(i, rng.randrange(i) if rng.random() < 0.5 else rng.randrange(max(0, i - 3), i))
    elif family == "path_like":   # caterpillar: long spine + legs + a few chords
        G = nx.Graph(); G.add_nodes_from(range(n))
        spine = max(2, int(n * rng.uniform(0.5, 0.9)))
        for i in range(1, spine):
            G.add_edge(i - 1, i)
        for i in range(spine, n):
            G.add_edge(i, rng.randrange(spine))
        for _ in range(rng.randint(0, max(1, n // 50))):
            a, b = rng.randrange(n), rng.randrange(n)
            if a != b:
                G.add_edge(a, b)
    elif family == "grid":
        r = max(1, int(math.sqrt(n))); c = max(1, n // r)
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(r, c))
        G.add_nodes_from(range(G.number_of_nodes(), n))       # leftover isolated nodes
        drop = [e for e in G.edges() if rng.random() < 0.1]
        G.remove_edges_from(drop)
    elif family == "components":
        G = nx.Graph(); start = 0
        parts = rng.randint(2, max(2, min(8, n // 3)))
        cuts = sorted(rng.sample(range(1, n), parts - 1)) if n > parts else list(range(1, parts))
        bounds = [0] + cuts + [n]
        for a, b in zip(bounds, bounds[1:]):
            size = b - a
            sub = base_graph(rng.choice(["er", "tree", "ba"]), size, avg_deg, rng) if size > 1 else nx.empty_graph(1)
            G.add_nodes_from(range(a, b))
            G.add_edges_from((a + u, a + v) for u, v in sub.edges())
    else:
        raise KeyError(family)
    G.add_nodes_from(range(n))
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def connect(G: nx.Graph, rng: random.Random) -> None:
    comps = [list(c) for c in nx.connected_components(G)]
    rng.shuffle(comps)
    for a, b in zip(comps, comps[1:]):
        G.add_edge(rng.choice(a), rng.choice(b))


def spanning_forest(G: nx.Graph) -> nx.Graph:
    F = nx.Graph(); F.add_nodes_from(G)
    for comp in nx.connected_components(G):
        r = next(iter(comp))
        F.add_edges_from(nx.bfs_edges(G, r))
    return F


# ------------------------------------------------------------- task shaping
def shape_instance(task: str, G: nx.Graph, rng: random.Random):
    """Return (n, edges, query, meta) with task-specific balancing."""
    t = TASKS[task]
    n = G.number_of_nodes()
    meta = {}
    if t.needs_connected:
        connect(G, rng)
    if task == "cycle_detection":
        if rng.random() < 0.5:
            G = spanning_forest(G); meta["planted"] = "forest"
        else:
            G = spanning_forest(G)
            comps = [list(c) for c in nx.connected_components(G) if len(c) >= 3]
            for _ in range(rng.randint(1, 3)):
                if comps:
                    c = rng.choice(comps); a, b = rng.sample(c, 2)
                    G.add_edge(a, b)
            meta["planted"] = "forest+chords" if comps else "forest"
    if task == "bipartite_check":
        side = {v: rng.random() < 0.5 for v in G}
        B = nx.Graph(); B.add_nodes_from(G)
        B.add_edges_from((u, v) for u, v in G.edges() if side[u] != side[v])
        G = B; meta["planted"] = "bipartite"
        if rng.random() < 0.5:
            for comp in sorted(nx.connected_components(G), key=len, reverse=True):
                same = [v for v in comp if side[v]], [v for v in comp if not side[v]]
                cand = [s for s in same if len(s) >= 2]
                if cand:
                    a, b = rng.sample(rng.choice(cand), 2)
                    G.add_edge(a, b); meta["planted"] = "odd_cycle"; break
    edges = []
    if t.directed:
        for u, v in G.edges():
            if rng.random() < 0.25:
                edges += [(u, v), (v, u)]
            else:
                edges.append((u, v) if rng.random() < 0.5 else (v, u))
    else:
        edges = [(u, v) for u, v in G.edges()]
    if t.weighted:
        hi = 50 if task == "max_flow" else 100
        edges = [(u, v, rng.randint(1, hi)) for u, v in edges]
    query = []
    if t.n_query == 2:
        query = pick_query(task, n, edges, t.directed, rng, meta)
    return n, edges, query, meta


def pick_query(task, n, edges, directed, rng, meta):
    adj = [[] for _ in range(n)]
    for e in edges:
        adj[e[0]].append(e[1])
        if not directed:
            adj[e[1]].append(e[0])
    s = rng.randrange(n)
    if n < 2:
        return [s, s]
    def reach_set(src):
        seen = {src}; st = [src]
        while st:
            u = st.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); st.append(v)
        return seen
    if task == "common_neighbors":
        two_hop = list({w for v in adj[s] for w in adj[v] if w != s})
        if two_hop and rng.random() < 0.7:
            return [s, rng.choice(two_hop)]
        return [s, rng.choice([x for x in range(n) if x != s])]
    want_reach = rng.random() < (0.5 if task == "reachability" else 0.85)
    for _ in range(20):
        R = reach_set(s) - {s}
        if want_reach and R:
            return [s, rng.choice(sorted(R))]
        if not want_reach:
            U = [x for x in range(n) if x not in R and x != s]
            if U:
                return [s, rng.choice(U)]
        s = rng.randrange(n)
    return [s, rng.choice([x for x in range(n) if x != s])]


def relabel(n, edges, query, directed, rng):
    perm = list(range(n)); rng.shuffle(perm)
    out = []
    for e in edges:
        u, v = perm[e[0]], perm[e[1]]
        if not directed and rng.random() < 0.5:
            u, v = v, u
        out.append((u, v) + tuple(e[2:]))
    rng.shuffle(out)
    return out, [perm[q] for q in query]


def to_text(n, edges, query, directed, weighted) -> str:
    lines = [f"{n} {len(edges)} {int(directed)} {int(weighted)}"]
    lines += [" ".join(map(str, e)) for e in edges]
    lines.append(" ".join(map(str, [len(query)] + list(query))))
    return "\n".join(lines) + "\n"


def parse_text(text: str):
    tok = text.split(); i = 0
    n, m, d, w = (int(x) for x in tok[:4]); i = 4
    edges = []
    for _ in range(m):
        if w:
            edges.append((int(tok[i]), int(tok[i + 1]), int(tok[i + 2]))); i += 3
        else:
            edges.append((int(tok[i]), int(tok[i + 1]))); i += 2
    k = int(tok[i]); query = [int(x) for x in tok[i + 1:i + 1 + k]]
    return n, edges, query, bool(d), bool(w)


# ----------------------------------------------------------- ground truth
_BIN = ROOT / "build"

def ensure_reference_binaries() -> None:
    _BIN.mkdir(exist_ok=True)
    for task in TASKS:
        src, exe = ROOT / "reference" / f"{task}.cpp", _BIN / task
        if not src.exists():
            continue
        if not exe.exists() or exe.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["g++", "-std=c++17", "-O2", "-o", str(exe), str(src)], check=True)


def reference_answer(task: str, text: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    r = subprocess.run([str(_BIN / task)], input=text, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"reference {task} failed: {r.stderr[:500]}")
    return r.stdout.strip(), time.perf_counter() - t0


# ------------------------------------------------------------- instances
def make_instance(task, family, n, density, rng):
    lo, hi = DENSITY_BINS[density]
    avg_deg = rng.uniform(lo, hi)
    G = base_graph(family, n, avg_deg, rng)
    n2, edges, query, meta = shape_instance(task, G, rng)
    t = TASKS[task]
    edges, query = relabel(n2, edges, query, t.directed, rng)
    text = to_text(n2, edges, query, t.directed, t.weighted)
    m = len(edges)
    dens = m / (n2 * (n2 - 1) / (1 if t.directed else 2)) if n2 > 1 else 0.0
    meta.update(family=family, n=n2, m=m, density_bin=density, density=round(dens, 6),
                avg_degree=round((1 if t.directed else 2) * m / n2, 3))
    return text, meta


def build_split(name, spec, out_dir: Path, seed: int):
    ensure_reference_binaries()
    rng = random.Random(f"{seed}:{name}")
    families = spec.get("families", SEEN_FAMILIES)
    lo, hi = spec["n_range"]
    records = []
    for task in spec.get("tasks", list(TASKS)):
        tdir = out_dir / name / task
        tdir.mkdir(parents=True, exist_ok=True)
        cap = TASKS[task].max_n
        for i in range(spec["per_task"]):
            fam = families[i % len(families)]
            dens = list(DENSITY_BINS)[(i // len(families)) % 3]
            if fam == "dense" and hi > 1000:
                fam = "er"
            n = rng.randint(min(lo, cap), min(hi, cap))
            text, meta = make_instance(task, fam, n, dens, rng)
            ans, secs = reference_answer(task, text)
            rid = f"{name}-{task}-{i:04d}"
            (tdir / f"{rid}.in").write_text(text)
            rec = dict(id=rid, split=name, task=task, input_path=str((tdir / f"{rid}.in").relative_to(out_dir)),
                       answer=ans, time_limit_s=spec["time_limit_s"], ref_seconds=round(secs, 4),
                       seen_family=fam in SEEN_FAMILIES, **meta)
            records.append(rec)
    with open(out_dir / f"{name}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "splits.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "hda"))
    ap.add_argument("--splits", nargs="*")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, spec in cfg["splits"].items():
        if a.splits and name not in a.splits:
            continue
        fam = spec.get("families", "seen")
        spec = dict(spec, families={"seen": SEEN_FAMILIES, "all": ALL_FAMILIES}.get(fam, fam))
        t0 = time.time()
        recs = build_split(name, spec, out, cfg["seed"] + spec.get("seed_offset", 0))
        print(f"{name}: {len(recs)} records in {time.time()-t0:.1f}s")
    json.dump(cfg, open(out / "config_used.json", "w"), indent=2)


if __name__ == "__main__":
    main()
