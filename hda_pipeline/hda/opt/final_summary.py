"""Collect final-grid runs, score each H* on val_opt, write final_summary.csv."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from hda.evaluate import evaluate_exe, load_split
from hda.sandbox import compile_cpp

SPLITS = ["test_small", "test_medium", "test_large", "test_xl", "test_stress"]


def score_val_opt(run: Path, task: str, seed: int, data: Path, split: str) -> float:
    cache = run / "val_opt.json"
    if cache.exists():
        return float(json.loads(cache.read_text())["accuracy"])
    code = run / "best_solution.cpp"
    if not code.exists():
        return 0.0
    ok, _log, exe = compile_cpp(code.read_text(), run / "score")
    acc = 0.0
    if ok:
        _, s = evaluate_exe(exe, load_split(data, split, task), data, permutation_check=5, seed=seed)
        acc = s.get("accuracy") or 0.0
    cache.write_text(json.dumps({"accuracy": acc}) + "\n")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default="../experiments/hda/optimizers/final")
    ap.add_argument("--data", default="data/hda")
    ap.add_argument("--val-opt", default="val_opt")
    ap.add_argument("--out", default="../experiments/hda/optimizers/final_summary.csv")
    a = ap.parse_args()
    rows = []
    for run in sorted(Path(a.final).glob("*/*")):
        f = run / "final.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        prompt = run.parent.name
        row = {"prompt": prompt, "task": d["task"], "condition": d["condition"],
               "seed": d["seed"], "best_it": d.get("best_it"),
               "val_acc": d.get("best_val_acc"),
               "val_opt": score_val_opt(run, d["task"], d["seed"], Path(a.data), a.val_opt),
               "judge": d.get("statement_judge"),
               "hda_prompt_tokens": d.get("tokens", {}).get("prompt", 0),
               "hda_completion_tokens": d.get("tokens", {}).get("completion", 0)}
        for s in SPLITS:
            row[s] = (d.get("test", {}).get(s) or {}).get("accuracy")
            row[f"{s}_verdicts"] = json.dumps((d.get("test", {}).get(s) or {}).get("verdicts") or {})
        for key in ("by_bucket", "by_density", "by_seen_family"):
            row[key] = json.dumps((d.get("test", {}).get("test_small") or {}).get(key) or {})
        rows.append(row)
    cols = list(rows[0]) if rows else []
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} runs -> {a.out}")


if __name__ == "__main__":
    main()
