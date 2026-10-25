"""Build the synthetic data splits (reproducible from one seed).

    python -m graphharness.build_data                      # all splits, default config
    python -m graphharness.build_data --splits train val   # only some splits
    python -m graphharness.build_data --config configs/data_splits.json --out data

Writes ``<out>/<split>.jsonl`` for each split and ``<out>/manifest.json``
with counts and a checksum per file, so teammates can confirm they built
exactly the same data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import zlib
from pathlib import Path

from .data_loader import DEFAULT_DATA_DIR, save_jsonl
from .generators import graph_to_json, make_graph_for_task
from .tasks import TASKS, as_json_value, get_task

# Only tiny random graphs can realistically come out identical twice, so only
# those are checked for duplicates across splits.
DEDUP_MAX_NODES = 12

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "data_splits.json"


def split_rng(seed: int, split: str, task: str) -> random.Random:
    """A separate random stream per (split, task), so changing one split or
    adding a task does not change the others."""
    return random.Random(seed * 1_000_003 + zlib.crc32(f"{split}/{task}".encode()))


def make_record(task_name: str, split: str, index: int, n: int, rng: random.Random) -> dict:
    task = get_task(task_name)
    G, info = make_graph_for_task(task_name, n, rng)
    params = task.make_params(G, rng)
    answer = task.solve(G, params)
    return {
        "id": f"{task_name}/{split}/{index:05d}",
        "source": "synthetic",
        "task": task_name,
        "split": split,
        "graph": graph_to_json(G, rng),
        "params": as_json_value(params),
        "question": task.question(params),
        "answer": as_json_value(answer),
        "meta": {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "task_kind": task.kind,
            **info,
        },
    }


def _key(rec: dict) -> str:
    """Identity of a problem, used to keep train/val/test free of duplicates."""
    edges = sorted(tuple(sorted(map(str, e))) for e in rec["graph"]["edges"])
    return json.dumps([rec["task"], rec["graph"]["nodes"], edges, rec["params"]], default=str)


def build_split(split: str, spec: dict, seed: int, tasks: list[str], seen: set) -> list[dict]:
    lo, hi = spec["n_range"]
    out = []
    for task_name in tasks:
        rng = split_rng(seed, split, task_name)
        made, tries = 0, 0
        while made < spec["per_task"]:
            tries += 1
            if tries > spec["per_task"] * 50:
                raise RuntimeError(f"could not make enough unique problems for {split}/{task_name}")
            n = rng.randint(lo, hi)
            rec = make_record(task_name, split, made, n, rng)
            if n <= DEDUP_MAX_NODES:  # tiny graphs can repeat; keep splits disjoint
                k = _key(rec)
                if k in seen:
                    continue
                seen.add(k)
            out.append(rec)
            made += 1
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--out", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--splits", nargs="*", help="only build these splits")
    ap.add_argument("--tasks", nargs="*", help="only build these tasks")
    args = ap.parse_args(argv)

    config = json.loads(Path(args.config).read_text())
    seed = config["seed"]
    tasks = args.tasks or config.get("tasks") or sorted(TASKS)
    split_names = args.splits or list(config["splits"])
    out_dir = Path(args.out)

    # Build splits in config order so duplicate removal is the same every time.
    seen: set = set()
    manifest = {"seed": seed, "tasks": tasks, "files": {}}
    for split in config["splits"]:
        spec = config["splits"][split]
        if split not in split_names and spec["n_range"][0] > DEDUP_MAX_NODES:
            continue  # cannot affect duplicate removal in other splits
        t0 = time.perf_counter()
        records = build_split(split, spec, seed, tasks, seen)
        if split not in split_names:
            continue  # built only to fill `seen`, keeps other splits identical
        path = out_dir / f"{split}.jsonl"
        save_jsonl(records, path)
        counts = {t: sum(r["task"] == t for r in records) for t in tasks}
        sizes = [r["meta"]["n_nodes"] for r in records]
        manifest["files"][path.name] = {
            "n_range": spec["n_range"],
            "count": len(records),
            "per_task": counts,
            "min_nodes": min(sizes),
            "max_nodes": max(sizes),
            "sha256": sha256(path),
        }
        print(f"{split:<12} {len(records):>5} problems, nodes {min(sizes)}-{max(sizes)}, "
              f"{time.perf_counter() - t0:.1f}s -> {path}")

    if not args.splits and not args.tasks:
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
