"""Run one HDA rollout for a prompt candidate and score its H* on val_opt."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hda.evaluate import evaluate_exe, load_split
from hda.sandbox import compile_cpp

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs" / "opt_hda.json").read_text())


def prompt_hash(candidate: dict) -> str:
    blob = json.dumps({k: candidate.get(k, "") for k in
                       ("strategy", "hidden_procedure", "iteration_note")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _trace(log_path: Path, code_path: Path) -> str:
    lines = []
    for raw in log_path.read_text().splitlines() if log_path.exists() else []:
        e = json.loads(raw)
        k = e.get("kind")
        if k == "iteration":
            lines.append(f"ITER {e.get('it')}: val={e.get('val_acc')} statement={str(e.get('statement'))[:200]}")
        elif k == "reply":
            lines.append(f"  step {e.get('step')}: action={e.get('action')} code={bool(e.get('code'))} "
                         f"completion={e.get('completion_tokens')}")
        elif k == "tool":
            lines.append(f"    tool {e.get('action')} compile_ok={e.get('compile_ok')}")
            if e.get("compile_log"):
                lines.append("    compiler: " + str(e["compile_log"])[:800].replace("\n", " | "))
            for o in (e.get("example_outputs") or [])[:5]:
                lines.append(f"    example {o.get('verdict')} got={o.get('output')} expected={o.get('expected')}")
        elif k == "submission":
            lines.append(f"    submission ok={e.get('ok')} policy={e.get('policy')}")
            if e.get("compile_log"):
                lines.append("    submit-compile: " + str(e["compile_log"])[:500].replace("\n", " | "))
    if code_path.exists():
        code = code_path.read_text().splitlines()
        lines.append("FINAL CODE (truncated):")
        lines.extend(code[:150])
    return "\n".join(lines)[:30000]


def _check_no_test(run_dir: Path):
    log = run_dir / "log.jsonl"
    for raw in log.read_text().splitlines() if log.exists() else []:
        e = json.loads(raw)
        if e.get("kind") == "test":
            raise RuntimeError("test evaluation ran during optimization")
    text = log.read_text() if log.exists() else ""
    for split in ("test_small", "test_medium", "test_large", "test_xl", "test_stress"):
        if split in text:
            raise RuntimeError(f"test split {split} appears in optimization log")


def rollout(candidate: dict, task: str, condition: str, seed: int, out_dir: str | Path,
            config: dict | None = None, timeout_s: int = 1800) -> dict:
    cfg = dict(CONFIG, **(config or {}))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ph = prompt_hash(candidate)
    key = f"{ph}_{task}_{condition}_{seed}"
    cache = out / "cache" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    run = out / key
    if run.exists():
        import shutil; shutil.rmtree(run)
    run.mkdir(parents=True)
    prompt_file = run / "prompt.json"
    prompt_file.write_text(json.dumps(candidate, indent=2) + "\n")
    cmd = [sys.executable, "-m", "hda.agent", "--task", task, "--condition", condition,
           "--k", str(cfg["k"]), "--n-iter", str(cfg["n_iter"]), "--steps", str(cfg["steps"]),
           "--seed", str(seed), "--data", cfg["data"], "--out", str(run),
           "--model", cfg["model"], "--base-url", cfg["base_url"],
           "--max-tokens", str(cfg["max_tokens"]), "--temperature", str(cfg["temperature"]),
           "--reasoning-effort", "", "--test-splits"]
    env = dict(os.environ, HDA_PROMPT_FILE=str(prompt_file), PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    with open(run / "stdout.log", "w") as so, open(run / "stderr.log", "w") as se:
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=so, stderr=se, timeout=timeout_s)
    wall = time.perf_counter() - t0
    _check_no_test(run)
    final = json.loads((run / "final.json").read_text()) if (run / "final.json").exists() else {}
    tokens = final.get("tokens") or {}
    llm_calls = sum(1 for raw in open(run / "log.jsonl") if '"kind": "reply"' in raw)
    val_opt_acc = 0.0
    code = run / "best_solution.cpp"
    if code.exists():
        ok, clog, exe = compile_cpp(code.read_text(), run / "score")
        if ok:
            _, s = evaluate_exe(exe, load_split(Path(cfg["data"]), cfg["val_opt"], task),
                                Path(cfg["data"]), permutation_check=5, seed=seed)
            val_opt_acc = s.get("accuracy") or 0.0
    result = dict(task=task, condition=condition, seed=seed, val_opt_acc=val_opt_acc,
                  val_acc=final.get("best_val_acc"), tokens_prompt=tokens.get("prompt", 0),
                  tokens_completion=tokens.get("completion", 0), llm_calls=llm_calls,
                  wall_s=round(wall, 1), returncode=p.returncode,
                  trace=_trace(run / "log.jsonl", code))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, indent=2) + "\n")
    return result


def rollout_many(candidate: dict, instances, out_dir, workers: int = 8, config=None):
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(rollout, candidate, t, c, s, out_dir, config) for t, c, s in instances]
        return [f.result() for f in futs]
