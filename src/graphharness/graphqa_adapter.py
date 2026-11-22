"""Convert GraphQA JSON files into the graphharness JSONL schema.

GraphQA publishes one JSON array per task and split. This adapter supports all
twelve GraphQA tasks. It parses the edge list and query from each question and
recomputes ground truth with the same solvers where the task is deterministic.
GraphQA's node-classification and maximum-flow answers are retained from the
source dataset because their exact generation semantics are not encoded in the
question; ``meta["ground_truth_source"]`` records that distinction.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import networkx as nx

from .data_loader import DEFAULT_DATA_DIR, save_jsonl
from .generators import graph_to_json
from .tasks import as_json_value, get_task

BASE_URL = "https://huggingface.co/datasets/baharef/GraphQA/raw/main"
TASKS = (
    "shortest_path", "cycle_check", "reachability", "connected_nodes",
    "disconnected_nodes", "node_count", "node_degree", "edge_count",
    "edge_existence", "triangle_counting", "node_classification", "maximum_flow",
)
SPLITS = ("train", "validation", "test")
TASK_NAMES = {
    "shortest_path": "shortest_path",
    "cycle_check": "cycle_detection",
    "reachability": "reachability",
    "connected_nodes": "connected_nodes",
    "disconnected_nodes": "disconnected_nodes",
    "node_count": "node_count",
    "node_degree": "node_degree",
    "edge_count": "edge_count",
    "edge_existence": "edge_existence",
    "triangle_counting": "triangle_counting",
    "node_classification": "node_classification",
    "maximum_flow": "maximum_flow",
}
EDGE_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
PAIR_RE = re.compile(r"from node\s+(\d+)\s+to node\s+(\d+)", re.IGNORECASE)
DEGREE_RE = re.compile(r"degree of node\s+(\d+)", re.IGNORECASE)
EDGE_Q_RE = re.compile(r"Is node\s+(\d+)\s+connected to node\s+(\d+)", re.IGNORECASE)
CONNECTED_RE = re.compile(r"nodes connected to\s+(\d+)", re.IGNORECASE)
DISCONNECTED_RE = re.compile(r"nodes that are not connected to\s+(\d+)", re.IGNORECASE)
LABEL_RE = re.compile(r"Node\s+(\d+)\s+likes\s+([A-Za-z0-9_-]+)", re.IGNORECASE)
CLASS_Q_RE = re.compile(r"Does node\s+(\d+)\s+like", re.IGNORECASE)


def _query_pair(question: str) -> tuple[int, int]:
    match = PAIR_RE.search(question)
    if not match:
        raise ValueError(f"could not find source/target nodes in question: {question[:120]!r}")
    return int(match.group(1)), int(match.group(2))


def convert_item(item: dict, source_task: str, split: str, index: int) -> dict:
    question = item["question"]
    edges = [(int(u), int(v)) for u, v in EDGE_RE.findall(question)]
    node_match = re.search(r"among nodes (.+?)\.", question, re.IGNORECASE | re.DOTALL)
    if not node_match:
        raise ValueError("could not find node list in GraphQA question")
    nodes = [int(n) for n in re.findall(r"\d+", node_match.group(1))]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    task_name = TASK_NAMES[source_task]
    task = get_task(task_name)
    source_answer = item.get("answer")
    params = {}

    if task_name in {"shortest_path", "reachability"}:
        source, target = _query_pair(question)
        params = {"source": source, "target": target}
    elif task_name == "node_degree":
        params = {"node": int(DEGREE_RE.search(question).group(1))}
    elif task_name == "edge_existence":
        m = EDGE_Q_RE.search(question)
        params = {"node1": int(m.group(1)), "node2": int(m.group(2))}
    elif task_name == "connected_nodes":
        params = {"node": int(CONNECTED_RE.search(question).group(1))}
    elif task_name == "disconnected_nodes":
        params = {"node": int(DISCONNECTED_RE.search(question).group(1))}
    elif task_name == "node_classification":
        labels = {int(n): label for n, label in LABEL_RE.findall(question)}
        params = {
            "node": int(CLASS_Q_RE.search(question).group(1)),
            "labels": labels,
            "answer": str(source_answer).strip(),
        }
        answer = {"label": str(source_answer).strip()}
    elif task_name == "maximum_flow":
        source, target = _query_pair(question)
        flow = int(re.findall(r"-?\d+", str(source_answer))[0])
        params = {"source": source, "target": target, "answer": flow}
        answer = {"flow": flow}
    else:
        answer = task.solve(graph, params)

    if task_name not in {"node_classification", "maximum_flow"}:
        answer = task.solve(graph, params)
    return {
        "id": f"{task_name}/graphqa_{split}/{index:05d}",
        "source": "graphqa",
        "task": task_name,
        "split": f"graphqa_{split}",
        "graph": graph_to_json(graph),
        "params": as_json_value(params),
        "question": task.question(params),
        "answer": as_json_value(answer),
        "meta": {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "task_kind": task.kind,
            "source_task": source_task,
            "source_answer": source_answer,
            "algorithm": item.get("algorithm"),
            "ground_truth_source": (
                "graphqa_source" if task_name in {"node_classification", "maximum_flow"}
                else "recomputed"
            ),
        },
    }


def load_source(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"GraphQA file must contain a JSON array: {path}")
    return value


def fetch_source(task: str, split: str, raw_dir: Path) -> Path | None:
    path = raw_dir / task / f"{task}_zero_shot_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"{BASE_URL}/{task}/{path.name}"
        try:
            with urllib.request.urlopen(url) as response:
                path.write_bytes(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            raise
    return path


def convert_dataset(raw_dir: Path, out_dir: Path, tasks=TASKS, splits=SPLITS,
                    limit: int | None = None) -> dict[str, int]:
    counts = {}
    for split in splits:
        records = []
        for source_task in tasks:
            path = fetch_source(source_task, split, raw_dir)
            if path is None:
                print(f"  skip {source_task}/{split} (not published)")
                continue
            items = load_source(path)
            if limit is not None:
                items = items[:limit]
            records.extend(convert_item(item, source_task, split, i)
                           for i, item in enumerate(items))
        output = out_dir / f"graphqa_{split}.jsonl"
        save_jsonl(records, output)
        counts[split] = len(records)
        print(f"{split:<12} {len(records):>5} problems -> {output}")
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(DEFAULT_DATA_DIR / "graphqa"))
    parser.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    convert_dataset(Path(args.raw_dir), Path(args.out), limit=args.limit)


if __name__ == "__main__":
    main()
