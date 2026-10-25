"""
ingest_graphqa.py
=================
Downloader and ingestion pipeline for the GraphQA benchmark (Fatemi et al., ICLR 2024).
Source: HuggingFace dataset `baharef/GraphQA`

Extracts:
  - Tasks: shortest_path, cycle_check, reachability
  - Splits: train, validation, test
  - Formats into standard GraphInstance and TaskInstance JSONL records.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .graph_generator import GraphInstance
from .tasks import TaskInstance


GRAPHQA_BASE_URL = "https://huggingface.co/datasets/baharef/GraphQA/raw/main"

# Tasks to ingest
SUPPORTED_TASKS = [
    "shortest_path",
    "cycle_check",
    "reachability",
]

SPLITS = ["train", "validation", "test"]


def _clean_answer(task_name: str, raw_answer: str) -> Any:
    """Standardize GraphQA raw text answer into Python typed answer."""
    raw = raw_answer.strip()
    if task_name == "shortest_path":
        clean = raw.rstrip(".")
        try:
            return int(clean)
        except ValueError:
            return raw
    elif task_name in ("cycle_check", "reachability"):
        lower = raw.lower()
        if "yes" in lower or "true" in lower:
            return True
        elif "no" in lower or "false" in lower:
            return False
        return raw
    return raw


def parse_graphqa_item(
    item: Dict[str, Any],
    task_name: str,
    split_name: str,
    index: int,
) -> Tuple[GraphInstance, TaskInstance]:
    """Parse a single GraphQA JSON entry into GraphInstance + TaskInstance."""
    question_text = item["question"]
    nnodes = int(item["nnodes"])
    nedges = int(item["nedges"])
    algorithm = item.get("algorithm", "unknown")

    # Extract all undirected edges (u, v) from question text
    edge_pairs = re.findall(r"\((\d+),\s*(\d+)\)", question_text)
    edges = [[int(u), int(v)] for u, v in edge_pairs]
    nodes = list(range(nnodes))

    # Graph ID
    graph_id = f"graphqa_{algorithm}_{split_name}_{index:05d}"
    graph_inst = GraphInstance(
        graph_id=graph_id,
        model=f"graphqa_{algorithm}",
        num_nodes=nnodes,
        num_edges=len(edges),
        nodes=nodes,
        edges=edges,
        directed=False,
        scale="small" if nnodes <= 50 else ("medium" if nnodes <= 300 else "large"),
    )

    # Task ID & typed answer
    ans = _clean_answer(task_name, item["answer"])
    # Map GraphQA task names to canonical task types
    task_type_map = {
        "shortest_path": "shortest_path",
        "cycle_check": "cycle_detection",
        "reachability": "reachability",
    }
    canonical_task_type = task_type_map.get(task_name, task_name)
    task_id = f"{graph_id}_{canonical_task_type}"

    task_inst = TaskInstance(
        task_id=task_id,
        graph_id=graph_id,
        task_type=canonical_task_type,
        question=question_text,
        answer=ans,
        metadata={
            "source_benchmark": "GraphQA",
            "original_task": task_name,
            "split": split_name,
            "algorithm": algorithm,
            "raw_answer": item["answer"],
        },
    )
    return graph_inst, task_inst


def download_and_ingest_graphqa(
    output_dir: str = "data",
    tasks: Optional[List[str]] = None,
    splits: Optional[List[str]] = None,
    max_per_split: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Download GraphQA dataset files from HuggingFace and save to JSONL.

    Parameters
    ----------
    output_dir : str
        Target output directory (default "data").
    tasks : list of str, optional
        Tasks to download. Default: ["shortest_path", "cycle_check", "reachability"].
    splits : list of str, optional
        Splits to download. Default: ["train", "validation", "test"].
    max_per_split : int, optional
        Limit number of examples per split (for fast subset download).

    Returns
    -------
    summary dict
    """
    if tasks is None:
        tasks = SUPPORTED_TASKS
    if splits is None:
        splits = SPLITS

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {}
    all_graphs: Dict[str, GraphInstance] = {}

    for split in splits:
        split_tasks: List[TaskInstance] = []

        for task_name in tasks:
            fname = f"{task_name}_zero_shot_{split}.json"
            url = f"{GRAPHQA_BASE_URL}/{task_name}/{fname}"
            print(f"Downloading GraphQA: {task_name}/{fname} ...")

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                print(f"  Warning: failed to download {url}: {e}")
                continue

            if max_per_split is not None:
                data = data[:max_per_split]

            for idx, item in enumerate(data):
                g_inst, t_inst = parse_graphqa_item(item, task_name, split, idx)
                all_graphs[g_inst.graph_id] = g_inst
                split_tasks.append(t_inst)

            print(f"  Parsed {len(data)} items for {task_name} ({split})")

        # Save tasks for this split
        split_task_file = out / f"tasks_graphqa_{split}.jsonl"
        with open(split_task_file, "w") as f:
            for t in split_tasks:
                f.write(json.dumps(t.to_dict()) + "\n")
        print(f"Saved {len(split_tasks)} tasks to {split_task_file}")

        summary[f"graphqa_{split}"] = {
            "n_tasks": len(split_tasks),
            "file": str(split_task_file),
        }

    # Save all unique graphs
    graph_file = out / "graphs_graphqa.jsonl"
    with open(graph_file, "w") as f:
        for g in all_graphs.values():
            f.write(json.dumps(g.to_dict()) + "\n")
    print(f"Saved {len(all_graphs)} graphs to {graph_file}")
    summary["graphqa_graphs"] = {
        "n_graphs": len(all_graphs),
        "file": str(graph_file),
    }

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest GraphQA benchmark dataset.")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--max-per-split", type=int, default=None,
                        help="Limit items per task/split (default: all).")
    args = parser.parse_args()

    print("=" * 60)
    print("CS 5624 — Ingesting GraphQA Benchmark Dataset")
    print("=" * 60)
    summary = download_and_ingest_graphqa(
        output_dir=args.output_dir,
        max_per_split=args.max_per_split,
    )
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
