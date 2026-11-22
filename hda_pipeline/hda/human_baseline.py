"""Score the human-written reference solutions (reference/*.cpp) on every split
through the same sandbox: an upper bound and a check that time limits are fair."""
import argparse, json
from pathlib import Path
from .tasks import TASKS
from .sandbox import compile_cpp
from .evaluate import load_split, evaluate_exe

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/hda"); ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test_small", "test_medium", "test_large", "test_xl", "test_stress"])
    a = ap.parse_args(); out = Path(a.out); res = {}
    root = Path(__file__).resolve().parent.parent
    for t in TASKS:
        ok, log, exe = compile_cpp((root / "reference" / f"{t}.cpp").read_text(), out / "build" / t)
        assert ok, log
        res[t] = {}
        for sp in a.splits:
            if (Path(a.data) / f"{sp}.jsonl").exists():
                _, s = evaluate_exe(exe, load_split(Path(a.data), sp, t), Path(a.data), permutation_check=3)
                res[t][sp] = dict(accuracy=s["accuracy"], max_cpu_s=s["max_cpu_s"], verdicts=s["verdicts"])
        print(t, {k: (v["accuracy"], v["max_cpu_s"]) for k, v in res[t].items()})
    json.dump(res, open(out / "human_reference.json", "w"), indent=2)

if __name__ == "__main__":
    main()
