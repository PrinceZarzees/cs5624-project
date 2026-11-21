"""Run the phase-1 experiment grid in parallel against one vLLM server.

    python -m hda.grid --conditions given hidden --seeds 0 1 2 --out runs/hda/main \
        --model openai/gpt-oss-120b --base-url http://$HOST:$PORT/v1 --parallel 16
Existing runs (final.json present) are skipped, so the command is restartable.
"""
from __future__ import annotations
import argparse, json, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .agent import OpenAIClient, RunConfig, run
from .tasks import TASKS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--conditions", nargs="+", default=["given", "hidden"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--data", default="data/hda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--feedback", default="aggregate", choices=["aggregate", "none"])
    ap.add_argument("--parallel", type=int, default=8)
    a = ap.parse_args()
    jobs = [(t, c, s) for t in a.tasks for c in a.conditions for s in a.seeds]
    def one(job):
        t, c, s = job
        out = Path(a.out) / f"{t}__{c}__k{a.k}__s{s}"
        if (out / "final.json").exists():
            return json.load(open(out / "final.json"))
        try:
            cl = OpenAIClient(a.model, a.base_url, a.temperature, a.max_tokens, a.reasoning_effort)
            judge = OpenAIClient(a.model, a.base_url, 0.0, 2000, "low")
            return run(RunConfig(task=t, condition=c, k=a.k, n_iter=a.n_iter, steps=a.steps, seed=s,
                                 data=a.data, out=str(out), feedback=a.feedback), cl, judge)
        except Exception:
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(traceback.format_exc())
            return None
    with ThreadPoolExecutor(a.parallel) as ex:
        for job, r in zip(jobs, ex.map(one, jobs)):
            print(job, None if r is None else (r["best_val_acc"], {k: v["accuracy"] for k, v in r["test"].items()}), flush=True)

if __name__ == "__main__":
    main()
