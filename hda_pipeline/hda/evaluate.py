"""Score a C++ solution on a split (validation during training, test at the end)."""
from __future__ import annotations
import argparse, json, random, statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .sandbox import compile_cpp, run_binary
from .generator import parse_text, relabel, to_text

SIZE_BUCKETS = [(15, "tiny"), (50, "small"), (300, "medium"), (1000, "large"), (2000, "xl"), (10**9, "stress")]

def bucket(n):
    return next(name for hi, name in SIZE_BUCKETS if n <= hi)


def load_split(data_dir: Path, split: str, task: str | None = None, limit: int | None = None):
    recs = [json.loads(l) for l in open(data_dir / f"{split}.jsonl")]
    if task:
        recs = [r for r in recs if r["task"] == task]
    return recs[:limit] if limit else recs


def evaluate_exe(exe: Path, recs, data_dir: Path, workers: int = 4, permutation_check: int = 0, seed: int = 0):
    def one(r):
        text = (data_dir / r["input_path"]).read_text()
        res = run_binary(exe, text, r["time_limit_s"], r["answer"])
        row = dict(id=r["id"], verdict=res.verdict, output=res.output.strip()[:200], expected=r["answer"],
                   cpu_s=res.cpu_s, n=r["n"], m=r["m"], family=r["family"], density_bin=r["density_bin"],
                   seen_family=r.get("seen_family", True), bucket=bucket(r["n"]))
        return row
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, recs))
    # Relabeling invariance: the same graph with fresh node IDs must give the same answer.
    inv = None
    if permutation_check:
        rng = random.Random(seed); bad = 0; tried = 0
        for r in rng.sample(recs, min(permutation_check, len(recs))):
            n, edges, q, d, w = parse_text((data_dir / r["input_path"]).read_text())
            e2, q2 = relabel(n, edges, q, d, rng)
            a = run_binary(exe, to_text(n, e2, q2, d, w), r["time_limit_s"], r["answer"])
            tried += 1; bad += a.verdict != "AC"
        inv = dict(tried=tried, failed=bad)
    return rows, summarize(rows, inv)


def summarize(rows, invariance=None):
    def acc(rs):
        return round(sum(r["verdict"] == "AC" for r in rs) / len(rs), 4) if rs else None
    by = lambda key: {k: dict(n=len(v), acc=acc(v)) for k, v in sorted(group(rows, key).items())}
    return dict(n=len(rows), accuracy=acc(rows), verdicts=dict(Counter(r["verdict"] for r in rows)),
                by_bucket=by("bucket"), by_family=by("family"), by_density=by("density_bin"),
                by_seen_family=by("seen_family"),
                max_cpu_s=max((r["cpu_s"] for r in rows), default=0.0),
                median_cpu_s=statistics.median([r["cpu_s"] for r in rows]) if rows else 0.0,
                invariance=invariance)


def group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r[key]].append(r)
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--data", default="data/hda")
    ap.add_argument("--splits", nargs="+", default=["test_small", "test_medium", "test_large", "test_xl", "test_stress"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ok, log, exe = compile_cpp(Path(a.solution).read_text(), out / "build")
    if not ok:
        json.dump(dict(compile_error=log), open(out / "summary.json", "w"), indent=2); return
    allsum = {}
    for sp in a.splits:
        rows, s = evaluate_exe(exe, load_split(Path(a.data), sp, a.task), Path(a.data), a.workers,
                               permutation_check=5)
        allsum[sp] = s
        with open(out / f"{sp}.rows.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(sp, s["accuracy"], s["verdicts"])
    json.dump(allsum, open(out / "summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
