"""Build exact small-graph phase-2 splits (max clique, MIS, MVC).

Exact labels come from NetworkX; these are evaluation-only. The HDA sees only
the input and, for training examples, the optimal node list.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import networkx as nx
from .generator import (ALL_FAMILIES, DENSITY_BINS, SEEN_FAMILIES, make_instance)
from .tasks import TASKS

ROOT = Path(__file__).resolve().parent.parent


def exact_solution(task, n, edges):
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from((u, v) for u, v, *_ in edges)
    if task == "max_clique":
        nodes, _ = nx.max_weight_clique(G, weight=None)
    elif task == "max_independent_set":
        nodes, _ = nx.max_weight_clique(nx.complement(G), weight=None)
    elif task == "min_vertex_cover":
        ind, _ = nx.max_weight_clique(nx.complement(G), weight=None)
        nodes = sorted(set(G.nodes) - set(ind))
    else:
        raise KeyError(task)
    nodes = sorted(nodes, key=str)
    return "[" + ", ".join(str(v) for v in nodes) + "]", len(nodes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "phase2_clique.json"))
    ap.add_argument("--out", default="data/hda_phase2")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, spec in cfg["splits"].items():
        rng = random.Random(f"{cfg['seed'] + spec.get('seed_offset', 0)}:{name}")
        fams = SEEN_FAMILIES if spec.get("families") == "seen" else ALL_FAMILIES
        lo, hi = spec["n_range"]
        recs = []
        for task in spec["tasks"]:
            tdir = out / name / task; tdir.mkdir(parents=True, exist_ok=True)
            for i in range(spec["per_task"]):
                fam = fams[i % len(fams)]
                dens = list(DENSITY_BINS)[(i // len(fams)) % 3]
                n = rng.randint(lo, hi)
                text, meta = make_instance(task, fam, n, dens, rng)
                # parse the relabeled text to get exactly what the HDA sees
                tok = text.split()
                nn, mm, d, w = map(int, tok[:4])
                vals = list(map(int, tok[4:]))
                if w:
                    edges = [tuple(vals[j:j + 3]) for j in range(0, 3 * mm, 3)]
                else:
                    edges = [tuple(vals[j:j + 2]) for j in range(0, 2 * mm, 2)]
                ans, ref = exact_solution(task, nn, edges)
                rid = f"{name}-{task}-{i:04d}"
                (tdir / f"{rid}.in").write_text(text)
                recs.append(dict(id=rid, split=name, task=task,
                                 input_path=str((tdir / f"{rid}.in").relative_to(out)),
                                 answer=ans, ref_value=ref,
                                 time_limit_s=spec["time_limit_s"],
                                 seen_family=fam in SEEN_FAMILIES, **meta))
        with open(out / f"{name}.jsonl", "w") as f:
            for r in recs: f.write(json.dumps(r) + "\n")
        print(name, len(recs))
    (out / "manifest.json").write_text(json.dumps({"config": cfg, "tasks": list(TASKS)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
