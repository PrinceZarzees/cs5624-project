"""Load and save problem records.

Every problem (from our generator or from GraphArena) is stored as one JSON
object per line (JSONL) with the same fields:

    {
      "id":        "shortest_path/test_medium/00007",
      "source":    "synthetic" or "grapharena",
      "task":      "shortest_path",
      "split":     "test_medium",
      "graph":     {"directed": false, "nodes": [...], "edges": [[u, v], ...]},
      "params":    {"source": 12, "target": 40},     # task inputs
      "question":  "Find a shortest path ...",       # does NOT include the graph
      "answer":    {"length": 3, "path": [...]},     # exact ground truth
      "meta":      {"n_nodes": 150, "n_edges": 301, "family": "ba", ...}
    }

The question text does not contain the graph on purpose: how the graph is
written into the prompt is part of the harness, so the harness decides
(see ``encoding.py`` for simple default encodings).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import networkx as nx

from .generators import graph_from_json

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

SPLITS = ("train", "val", "test_small", "test_medium", "test_large", "test_stress")


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            n += 1
    return n


def load_split(split: str, data_dir: str | Path | None = None,
               tasks: Iterable[str] | None = None) -> list[dict]:
    """Load one split, e.g. ``load_split("train")``, optionally only some tasks."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = data_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the data first: python -m graphharness.build_data"
        )
    records = load_jsonl(path)
    if tasks is not None:
        wanted = set(tasks)
        records = [r for r in records if r["task"] in wanted]
    return records


def to_graph(record: dict) -> nx.Graph:
    """The record's graph as a networkx graph."""
    return graph_from_json(record["graph"])


def iter_by_task(records: Iterable[dict]) -> Iterator[tuple[str, list[dict]]]:
    """Group records by task name, in sorted task order."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r["task"], []).append(r)
    for name in sorted(groups):
        yield name, groups[name]
