"""Convert GraphQA JSON files into the graphharness JSONL schema.

GraphQA publishes one JSON array per task and split.  This adapter parses the
edge list and query from each question, then recomputes ground truth with the
same solvers used for synthetic data.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

import networkx as nx

from .data_loader import DEFAULT_DATA_DIR, save_jsonl
from .generators import graph_to_json
from .tasks import as_json_value, get_task

BASE_URL = "https://huggingface.co/datasets/baharef/GraphQA/raw/main"
TASKS = ("shortest_path", "cycle_check", "reachability")
SPLITS = ("train", "validation", "test")
TASK_NAMES = {
    "shortest_path": "shortest_path",
    "cycle_check": "cycle_detection",
    "reachability": "reachability",
}
EDGE_RE = re.compile(r"\((\d+)\s*,\s*(\d+)\)")
PAIR_RE = re.compile(r"from node\s+(\d+)\s+to node\s+(\d+)", re.IGNORECASE)


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
    params = {}
    if task_name in {"shortest_path", "reachability"}:
        source, target = _query_pair(question)
        params = {"source": source, "target": target}
    answer = task.solve(graph, params)
    return {
        "id": f"{task_name}/graphqa_{split}/{index:05d}",
        "source": "graphqa",
        "task": task_name,
        "split": f"graphqa_{split}",
        "graph": graph_to_json(graph),
        "params": as_json_value(params),
        "question": question.strip(),
        "answer": as_json_value(answer),
        "meta": {
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "task_kind": task.kind,
            "source_task": source_task,
            "source_answer": item.get("answer"),
            "algorithm": item.get("algorithm"),
        },
    }


def load_source(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"GraphQA file must contain a JSON array: {path}")
    return value


def fetch_source(task: str, split: str, raw_dir: Path) -> Path:
    path = raw_dir / task / f"{task}_zero_shot_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        url = f"{BASE_URL}/{task}/{path.name}"
        with urllib.request.urlopen(url) as response:
            path.write_bytes(response.read())
    return path


def convert_dataset(raw_dir: Path, out_dir: Path, tasks=TASKS, splits=SPLITS,
                    limit: int | None = None) -> dict[str, int]:
    counts = {}
    for split in splits:
        records = []
        for source_task in tasks:
            path = fetch_source(source_task, split, raw_dir)
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
