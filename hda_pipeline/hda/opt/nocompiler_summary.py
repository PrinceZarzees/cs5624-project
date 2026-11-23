"""Aggregate the full compiler-on vs no-compiler grid."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path

TASKS = ["shortest_path", "reachability", "connected_components", "cycle_detection",
         "diameter", "common_neighbors", "triangle_count", "bipartite_check",
         "mst_weight", "max_flow"]
SPLITS = ["test_small", "test_medium", "test_large", "test_xl", "test_stress"]


def load(root: Path, arm: str, task: str, cond: str, seed: int):
    d = root / f"{task}__{cond}__k5__s{seed}"
    f = d / "final.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    actions, fails, cfails, tools = {}, 0, 0, 0
    log = d / "log.jsonl"
    if log.exists():
        for line in log.open():
            e = json.loads(line)
            if e.get("kind") == "reply":
                actions[e.get("action")] = actions.get(e.get("action"), 0) + 1
            elif e.get("kind") == "tool":
                tools += 1
            elif e.get("kind") == "submission" and not e.get("ok"):
                fails += 1
                if e.get("compile_log"):
                    cfails += 1
    row = dict(arm=arm, task=task, condition=cond, seed=seed,
               val_acc=data.get("best_val_acc"), judge=data.get("statement_judge"),
               no_compiler=bool(data.get("no_compiler")),
               tokens_prompt=data.get("tokens", {}).get("prompt", 0),
               tokens_completion=data.get("tokens", {}).get("completion", 0),
               failed_submissions=fails, compile_fail_submissions=cfails, tool_events=tools,
               actions=" ".join(f"{k}:{v}" for k, v in sorted(actions.items(), key=lambda kv: str(kv[0]))))
    for s in SPLITS:
        # all splits were requested; a missing split means the final program did
        # not compile, which is a zero, not a missing observation
        acc = (data.get("test", {}).get(s) or {}).get("accuracy")
        row[s] = 0.0 if acc is None else acc
    return row


def mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return statistics.mean(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", default="../experiments/hda/optimizers/final/unoptimized")
    ap.add_argument("--off", default="../experiments/hda/nocompiler_grid_all10")
    ap.add_argument("--out", default="../experiments/hda/nocompiler_grid_all10/compare_all10.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    rows = []
    for task in TASKS:
        for cond in ("given", "hidden"):
            for seed in a.seeds:
                for arm, root in (("on", Path(a.on)), ("off", Path(a.off))):
                    r = load(root, arm, task, cond, seed)
                    if r:
                        rows.append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} runs -> {a.out}")
    for cond in ("given", "hidden", "all"):
        print(f"\n== {cond}")
        for arm in ("on", "off"):
            sub = [r for r in rows if r["arm"] == arm and (cond == "all" or r["condition"] == cond)]
            print(f"  {arm:3s} n={len(sub):3d} val={mean(sub,'val_acc'):.3f} "
                  + " ".join(f"{s.replace('test_','')}={mean(sub,s):.3f}" for s in SPLITS)
                  + f" prompt_tok={mean(sub,'tokens_prompt'):.0f} "
                  + f"tool_events={mean(sub,'tool_events'):.1f} "
                  + f"failed={sum(r['failed_submissions'] for r in sub)} "
                  + f"cfail={sum(r['compile_fail_submissions'] for r in sub)}")

    # paired per-task deltas, averaged over seeds
    print("\n== per-task delta (off - on), mean over seeds")
    for cond in ("given", "hidden"):
        for task in TASKS:
            on = [r for r in rows if r["arm"] == "on" and r["condition"] == cond and r["task"] == task]
            off = [r for r in rows if r["arm"] == "off" and r["condition"] == cond and r["task"] == task]
            if not on or not off:
                continue
            a, b = mean(off, "val_acc"), mean(on, "val_acc")
            c, d = mean(off, "test_stress"), mean(on, "test_stress")
            dval = f"{a-b:+.3f}" if a is not None and b is not None else "n/a"
            dstr = f"{c-d:+.3f}" if c is not None and d is not None else "n/a"
            print(f"  {cond:6s} {task:22s} dval={dval} dstress={dstr}")


if __name__ == "__main__":
    main()
