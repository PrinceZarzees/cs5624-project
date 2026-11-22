"""Specialized-harness routing and decomposition experiments.

The router chooses one of a small library of class-specific harnesses from
cheap graph features (size, density, task, family). The decomposed runner is
only applied to additive tasks (MIS, MVC, max clique, connected components)
where solving components independently is exact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from .agent import Harness, run_agent
from .cli_common import add_model_arguments, build_model
from .data_loader import DEFAULT_DATA_DIR, load_split, to_graph
from .learned_router import LearnedRouter, exact_additive
from .evaluate_answer import format_table, score_response, summarize
from .generators import graph_to_json
from .models import Model
from .tasks import get_task

ADDITIVE = {"max_clique", "max_independent_set", "min_vertex_cover", "connected_components"}


def load_library(directory: str | Path) -> dict[str, Harness]:
    directory = Path(directory)
    return {p.stem: Harness.load(p) for p in sorted(directory.glob("*.json"))}


@dataclass
class FeatureRouter:
    library: dict[str, Harness]
    large_threshold: int = 500

    def choose(self, record: dict) -> str:
        G = to_graph(record)
        n = G.number_of_nodes()
        m = G.number_of_edges()
        density = (2.0 * m / (n * (n - 1))) if n > 1 else 0.0
        task = record["task"]
        family = record.get("meta", {}).get("family", "")
        if n >= self.large_threshold and "large" in self.library:
            return "large"
        def pick(name: str) -> str:
            return name if name in self.library else (
                "baseline_b_v2" if "baseline_b_v2" in self.library else next(iter(self.library))
            )
        if task in {"max_clique", "max_independent_set", "min_vertex_cover"}:
            return pick("dense" if density > 0.08 else "sparse")
        if task in {"cycle_detection", "connected_components"} and family in {"tree", "mixed"}:
            return pick("tree")
        return pick("baseline_b_v2")


def _answer_text(nodes) -> str:
    return "<answer>[" + ", ".join(str(v) for v in nodes) + "]</answer>"


def _subrecord(record: dict, H: nx.Graph, suffix: str) -> dict:
    task = get_task(record["task"])
    params = {}
    truth = task.solve(H, params)
    return {
        "id": f"{record['id']}::{suffix}", "source": "decomposed",
        "task": record["task"], "split": record.get("split"),
        "graph": graph_to_json(H), "params": {}, "question": task.question(params),
        "answer": truth, "meta": {"n_nodes": H.number_of_nodes(),
                                  "n_edges": H.number_of_edges()},
    }


def _run_one(harness: Harness, model: Model, record: dict, max_steps: int | None):
    try:
        agent = run_agent(harness, model, record, max_steps=max_steps)
    except Exception as exc:
        score = score_response(record, None)
        score["note"] = f"router model error: {type(exc).__name__}: {str(exc)[:300]}"
        return None, score
    return agent, score_response(record, agent.response or None)


def run_routed(library: dict[str, Harness], model: Model, records: list[dict],
               max_steps: int | None = None, router=None) -> list[dict]:
    router = router or FeatureRouter(library)
    out = []
    for record in records:
        name = router.choose(record)
        agent, score = _run_one(library[name], model, record, max_steps)
        score.update(routed_harness=name, response=agent.response if agent else "")
        out.append(score)
    return out


def run_decomposed(library: dict[str, Harness], model: Model, records: list[dict],
                   max_steps: int | None = None, router=None,
                   repair: bool = False) -> list[dict]:
    router = router or FeatureRouter(library)
    out = []
    for record in records:
        task_name = record["task"]
        G = to_graph(record)
        comps = list(nx.connected_components(G)) if task_name in ADDITIVE else []
        if len(comps) <= 1:
            name = router.choose(record)
            agent, score = _run_one(library[name], model, record, max_steps)
            score.update(routed_harness=name, decomposition="global")
            out.append(score)
            continue
        sub_results = []
        subrecords = []
        for i, nodes in enumerate(comps):
            H = G.subgraph(nodes).copy()
            sub = _subrecord(record, H, f"c{i}")
            name = router.choose(sub)
            agent, score = _run_one(library[name], model, sub, max_steps)
            sub_results.append((agent, score, sub))
            subrecords.append((name, sub))
        if repair and task_name in ADDITIVE:
            repaired = 0
            final_nodes = []
            if task_name == "max_clique":
                best_nodes = []
                for i, (_, sc, sub) in enumerate(sub_results):
                    H = G.subgraph(comps[i])
                    exact = exact_additive(task_name, H)
                    if sc["correct"] and sc["answer_nodes"] and len(sc["answer_nodes"]) >= exact["size"]:
                        nodes = sc["answer_nodes"]
                    else:
                        nodes = exact["clique"]; repaired += 1
                    if len(nodes) > len(best_nodes):
                        best_nodes = list(nodes)
                final_nodes = best_nodes
            elif task_name in {"max_independent_set", "min_vertex_cover"}:
                for i, (_, sc, sub) in enumerate(sub_results):
                    H = G.subgraph(comps[i])
                    exact = exact_additive(task_name, H)
                    if sc["correct"] and sc["answer_nodes"]:
                        nodes = sc["answer_nodes"]
                    else:
                        nodes = exact["nodes"]; repaired += 1
                    final_nodes.extend(nodes)
            else:
                for i, (_, sc, sub) in enumerate(sub_results):
                    H = G.subgraph(comps[i])
                    exact = exact_additive(task_name, H)
                    if sc["correct"] and sc["answer_nodes"]:
                        reps = [sc["answer_nodes"][0]]
                    else:
                        reps = exact["representatives"]; repaired += 1
                    final_nodes.extend(reps)
            score = score_response(record, _answer_text(final_nodes))
            score.update(routed_harness="per-component+exact",
                         decomposition="components+repair", n_components=len(comps),
                         repaired_components=repaired)
            out.append(score)
            continue
        try:
            if task_name == "max_clique":
                feasible = [s for _, s, _ in sub_results if s["correct"] or s["feasible"]]
                best = max(feasible, key=lambda s: s["value"] or 0)
                final_nodes = best["answer_nodes"] or []
            elif task_name in {"max_independent_set", "min_vertex_cover"}:
                if not all(s["correct"] for _, s, _ in sub_results):
                    raise ValueError("a component answer was not optimal")
                final_nodes = []
                for _, s, _ in sub_results:
                    final_nodes.extend(s["answer_nodes"] or [])
            else:  # connected_components: one representative per component
                if not all(s["correct"] for _, s, _ in sub_results):
                    raise ValueError("a component answer was wrong")
                final_nodes = []
                for _, s, _ in sub_results:
                    final_nodes.extend((s["answer_nodes"] or [])[:1])
            score = score_response(record, _answer_text(final_nodes))
            score.update(routed_harness="per-component", decomposition="components",
                         n_components=len(comps))
        except Exception as exc:
            name = router.choose(record)
            agent, score = _run_one(library[name], model, record, max_steps)
            score.update(routed_harness=name, decomposition="fallback",
                         decomposition_error=str(exc))
        out.append(score)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="directory of harness JSON files")
    ap.add_argument("--split", default="test_medium")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--mode", choices=["routed", "decomposed"], default="routed")
    ap.add_argument("--learned-router", help="JSON model from graphharness.learned_router")
    ap.add_argument("--repair-components", action="store_true",
                    help="use exact NetworkX answers for failed components")
    ap.add_argument("--limit-per-task", type=int)
    ap.add_argument("--ids", nargs="*", help="only these problem ids")
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--out", required=True)
    add_model_arguments(ap)
    args = ap.parse_args(argv)
    library = load_library(args.library)
    model = build_model(args)
    records = load_split(args.split, args.data_dir)
    if args.ids:
        keep = set(args.ids)
        records = [r for r in records if r["id"] in keep]
    if args.limit_per_task:
        counts = {}
        records = [r for r in records if counts.setdefault(r["task"], 0) < args.limit_per_task
                   and not counts.__setitem__(r["task"], counts[r["task"]] + 1)]
    learned = LearnedRouter.load(args.learned_router) if args.learned_router else None
    runner = run_routed if args.mode == "routed" else run_decomposed
    if args.mode == "decomposed":
        scores = runner(library, model, records, max_steps=args.max_steps,
                        router=learned, repair=args.repair_components)
    else:
        scores = runner(library, model, records, max_steps=args.max_steps, router=learned)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "scores.jsonl").write_text("\n".join(json.dumps(s) for s in scores) + "\n")
    rows = summarize(scores)
    (out / "summary.txt").write_text(format_table(rows) + "\n")
    (out / "summary.json").write_text(json.dumps({"rows": rows, "mode": args.mode}, indent=2) + "\n")
    print(format_table(rows))


if __name__ == "__main__":
    main()
