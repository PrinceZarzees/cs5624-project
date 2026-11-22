"""O3: score the empty prompt on the acceptance set A (20 instances x 3 seeds)."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from .rollout import rollout_many

TASKS = ["shortest_path","reachability","connected_components","cycle_detection","diameter",
         "common_neighbors","triangle_count","bipartite_check","mst_weight","max_flow"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[200,201,202])
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    inst = [(t,c,s) for t in a.tasks for c in ("given","hidden") for s in a.seeds]
    cfg = {"model":a.model, "base_url":a.base_url}
    rows = rollout_many({}, inst, out/"rollouts", workers=a.workers, config=cfg)
    (out/"rollouts.json").write_text(json.dumps(rows, indent=2) + "\n")
    by = {}
    for r in rows:
        by.setdefault((r["task"], r["condition"]), []).append(r["val_opt_acc"])
    means = {f"{t}|{c}": statistics.mean(v) for (t,c),v in by.items()}
    overall = statistics.mean(r["val_opt_acc"] for r in rows)
    std = statistics.pstdev([means[k] for k in means])
    payload = dict(overall=overall, per_instance=means,
                   seed_means={str(s): statistics.mean(r["val_opt_acc"] for r in rows if r["seed"]==s)
                               for s in a.seeds},
                   std_across_instances=std, n=len(rows),
                   tokens_prompt=sum(r["tokens_prompt"] for r in rows),
                   tokens_completion=sum(r["tokens_completion"] for r in rows),
                   llm_calls=sum(r["llm_calls"] for r in rows))
    (out/"baseline_A.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
