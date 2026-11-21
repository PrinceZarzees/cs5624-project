"""Verify dataset integrity and that every stored ground truth scores correctly.

    python -m graphharness.verify_data
    python -m graphharness.verify_data --data-dir data --json

This catches a corrupted split, a stale manifest, or a solver/checker
regression without calling any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .data_loader import DEFAULT_DATA_DIR, load_split
from .evaluate_answer import score_response
from .tasks import get_task


def truth_text(record: dict) -> str:
    task = get_task(record["task"])
    answer = record["answer"]
    if task.answer_kind == "nodes":
        nodes = {
            "shortest_path": answer.get("path"),
            "reachability": answer.get("path"),
            "diameter": answer.get("path"),
            "connected_components": answer.get("representatives"),
            "cycle_detection": answer.get("cycle"),
            "max_clique": answer.get("clique"),
            "common_neighbors": answer.get("nodes"),
            "max_independent_set": answer.get("nodes"),
            "min_vertex_cover": answer.get("nodes"),
            "hamiltonian_path": answer.get("path"),
            "connected_nodes": answer.get("nodes"),
            "disconnected_nodes": answer.get("nodes"),
            "traveling_salesman": answer.get("route"),
        }.get(record["task"])
    elif task.answer_kind == "number":
        nodes = None
        for key in ("count", "degree", "flow", "cost"):
            if key in answer:
                return f"<answer>{answer[key]}</answer>"
    elif task.answer_kind == "boolean":
        return "<answer>" + ("yes" if answer.get("exists") else "no") + "</answer>"
    elif task.answer_kind == "string":
        return f"<answer>{answer.get('label', '')}</answer>"
    elif task.answer_kind == "mapping":
        mapping = answer.get("mapping") or []
        return "<answer>[" + ", ".join(str(v) for v in mapping) + "]</answer>"
    elif task.answer_kind == "two_lists":
        n1, n2 = answer.get("nodes1") or [], answer.get("nodes2") or []
        return "<answer>[" + ", ".join(map(str, n1)) + "], [" + ", ".join(map(str, n2)) + "]</answer>"
    else:
        nodes = None
    if nodes is None:
        raise KeyError(f"no truth extractor for task {record['task']!r}")
    return "<answer>[" + ", ".join(str(v) for v in nodes) + "]</answer>"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_split(split: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> dict:
    path = Path(data_dir) / f"{split}.jsonl"
    records = load_split(split, data_dir)
    bad = []
    tasks = {}
    min_nodes = None
    max_nodes = None
    for record in records:
        tasks[record["task"]] = tasks.get(record["task"], 0) + 1
        n = len(record["graph"]["nodes"])
        min_nodes = n if min_nodes is None else min(min_nodes, n)
        max_nodes = n if max_nodes is None else max(max_nodes, n)
        missing = [key for key in ("id", "source", "task", "split", "graph", "params",
                                   "question", "answer", "meta") if key not in record]
        if missing:
            bad.append({"id": record.get("id"), "error": f"missing fields {missing}"})
            continue
        score = score_response(record, truth_text(record))
        if not score["correct"]:
            bad.append({"id": record["id"], "error": score.get("note") or "truth not correct"})
    return {
        "split": split,
        "path": str(path),
        "count": len(records),
        "tasks": tasks,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "sha256": sha256(path) if path.exists() else None,
        "errors": bad,
    }


def verify_all(data_dir: str | Path = DEFAULT_DATA_DIR,
               manifest: str | Path | None = None) -> dict:
    data_dir = Path(data_dir)
    manifest_path = Path(manifest) if manifest else data_dir / "manifest.json"
    if manifest_path.exists():
        config = json.loads(manifest_path.read_text(encoding="utf-8"))
        splits = [
            name[:-6] if name.endswith(".jsonl") else name
            for name in config.get("files", {})
        ]
        expected = config.get("files", {})
    else:
        splits = sorted(p.stem for p in data_dir.glob("*.jsonl"))
        expected = {}

    results = [verify_split(split, data_dir) for split in splits]
    manifest_errors = []
    for result in results:
        want = expected.get(f"{result['split']}.jsonl")
        if not want:
            continue
        if want.get("count") != result["count"]:
            manifest_errors.append(
                f"{result['split']}: count {result['count']} != manifest {want.get('count')}"
            )
        if want.get("sha256") and want["sha256"] != result["sha256"]:
            manifest_errors.append(f"{result['split']}: sha256 differs from manifest")
        if want.get("per_task") and want["per_task"] != result["tasks"]:
            manifest_errors.append(f"{result['split']}: per-task counts differ from manifest")
    return {
        "data_dir": str(data_dir),
        "manifest": str(manifest_path) if manifest_path.exists() else None,
        "splits": results,
        "manifest_errors": manifest_errors,
        "ok": not manifest_errors and all(not r["errors"] for r in results),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--manifest", help="default: <data-dir>/manifest.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    payload = verify_all(args.data_dir, args.manifest)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for row in payload["splits"]:
            status = "OK" if not row["errors"] else f"{len(row['errors'])} ERRORS"
            print(f"{row['split']:<12} {row['count']:>5} problems  "
                  f"nodes {row['min_nodes']}-{row['max_nodes']}  {status}")
        for error in payload["manifest_errors"]:
            print("manifest:", error)
        print("OK" if payload["ok"] else "FAILED")
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
