"""Compare matched compiler-on and no-compiler runs for one task/condition."""
from __future__ import annotations
import argparse, csv, json, statistics
from pathlib import Path

SPLITS = ["test_small", "test_medium", "test_large", "test_xl", "test_stress"]


def load_run(root: Path, task: str, cond: str, seed: int):
    d = root / f"{task}__{cond}__k5__s{seed}"
    f = d / "final.json"
    if not f.exists():
        return None
    data = json.loads(f.read_text())
    log = d / "log.jsonl"
    actions, fails, tools = {}, 0, 0
    if log.exists():
        for line in log.open():
            e = json.loads(line)
            if e.get("kind") == "reply":
                actions[e.get("action")] = actions.get(e.get("action"), 0) + 1
            elif e.get("kind") == "tool":
                tools += 1
            elif e.get("kind") == "submission" and not e.get("ok"):
                fails += 1
    row = dict(seed=seed, task=task, condition=cond,
               val_acc=data.get("best_val_acc"),
               no_compiler=bool(data.get("no_compiler")),
               tokens_prompt=data.get("tokens", {}).get("prompt", 0),
               tokens_completion=data.get("tokens", {}).get("completion", 0),
               compile_fail_submissions=fails, tool_events=tools,
               actions=" ".join(f"{k}:{v}" for k, v in sorted(actions.items(), key=lambda kv: str(kv[0]))))
    for s in SPLITS:
        row[s] = (data.get("test", {}).get(s) or {}).get("accuracy")
    return row


def mean(rows, key):
    v = [r[key] for r in rows if r[key] is not None]
    return statistics.mean(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", default="../experiments/hda/optimizers/final/unoptimized")
    ap.add_argument("--off", required=True)
    ap.add_argument("--task", default="max_flow")
    ap.add_argument("--condition", default="given")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    on_root, off_root = Path(a.on), Path(a.off)
    rows = []
    for s in a.seeds:
        for arm, root in (("on", on_root), ("off", off_root)):
            r = load_run(root, a.task, a.condition, s)
            if r:
                r["arm"] = arm
                rows.append(r)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{'arm':4s} {'seed':4s} {'val':>6s} {'small':>6s} {'med':>6s} {'large':>6s} {'xl':>6s} {'stress':>6s} "
          f"{'cfail':>5s} {'tools':>5s}")
    for r in rows:
        print(f"{r['arm']:4s} {r['seed']:<4d} {r['val_acc']:6.3f} "
              + " ".join(f"{(r[s] if r[s] is not None else float('nan')):6.3f}" for s in SPLITS)
              + f" {r['compile_fail_submissions']:5d} {r['tool_events']:5d}")
    print("\nmeans")
    for arm in ("on", "off"):
        sub = [r for r in rows if r["arm"] == arm]
        print(arm, "val", round(mean(sub, "val_acc"), 4),
              {s: round(mean(sub, s), 4) for s in SPLITS})


if __name__ == "__main__":
    main()
