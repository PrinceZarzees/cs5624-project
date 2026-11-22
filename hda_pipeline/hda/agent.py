"""HDA training loop.

One run = (task, condition, k, seed, model). For it = 1..n_iter:
  * the HDA receives k NEW I/O examples (cumulative list is shown), the current
    best program and the aggregate validation feedback of its last submission;
  * it works for up to `steps` tool calls (test / run / stress / submit) and may
    revise its <statement> at any step;
  * the submitted program passes the policy check, is scored on the hidden
    validation split, and re-run on relabeled graphs.
The program with the best validation accuracy (ties -> later iteration) is the
run's final harness H*; it is then evaluated once on the test splits.

Everything is logged to <out>/log.jsonl (one event per line): prompts, raw
replies, the model's reasoning trace when the server returns it, statements,
code, tool results, program outputs on every example and validation graph.
"""
from __future__ import annotations
import argparse, json, os, random, re, time
from dataclasses import dataclass, field
from pathlib import Path
from .tasks import TASKS
from .prompts import SYSTEM, STATEMENT_GIVEN, STATEMENT_HIDDEN, EXAMPLES, ITERATION, FEEDBACK, FEEDBACK_Q, JUDGE
from .sandbox import compile_cpp, run_binary, run_capture
from .policy import check_source
from .evaluate import load_split, evaluate_exe, evaluate_exe_quality

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = (ROOT / "templates" / "graph_template.cpp").read_text()
MAX_SHOWN_INPUT_CHARS = 6000


# ------------------------------------------------------------------ clients
class OpenAIClient:
    """Any OpenAI-compatible server (vLLM on ARC). Reasoning traces are taken from
    `reasoning_content` / `reasoning` when the server returns them (gpt-oss does)."""
    def __init__(self, model, base_url, temperature=1.0, max_tokens=16000, reasoning_effort=None):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model, self.temperature, self.max_tokens, self.effort = model, temperature, max_tokens, reasoning_effort

    def chat(self, messages):
        kw = dict(model=self.model, messages=messages, temperature=self.temperature,
                  stop=["</action>"])
        if self.max_tokens:
            kw["max_tokens"] = self.max_tokens
        if self.effort:
            kw["extra_body"] = {"reasoning_effort": self.effort}
        r = self.c.chat.completions.create(**kw)
        m = r.choices[0].message
        reasoning = getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None)
        u = r.usage
        return dict(content=m.content or "", reasoning=reasoning,
                    prompt_tokens=getattr(u, "prompt_tokens", 0), completion_tokens=getattr(u, "completion_tokens", 0))


class ScriptedClient:
    """Offline client for tests: replays a list of replies (cycling the last one)."""
    def __init__(self, replies):
        self.replies, self.i = replies, 0
    def chat(self, messages):
        r = self.replies[min(self.i, len(self.replies) - 1)]; self.i += 1
        return dict(content=r, reasoning="(scripted)", prompt_tokens=sum(len(m["content"]) // 4 for m in messages),
                    completion_tokens=len(r) // 4)


# ------------------------------------------------------------------ parsing
def last_block(text, lang):
    blocks = re.findall(rf"```{lang}\s*\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else None

def parse_action(text):
    acts = re.findall(r"<action>\s*(\w+)\s*</action>", text)
    return acts[-1].lower() if acts else None

def parse_statement(text):
    s = re.findall(r"<statement>(.*?)</statement>", text, flags=re.S)
    return s[-1].strip() if s else None

def show_input(text):
    return text if len(text) <= MAX_SHOWN_INPUT_CHARS else text[:MAX_SHOWN_INPUT_CHARS] + "\n... (truncated)\n"


# ------------------------------------------------------------------ run
@dataclass
class RunConfig:
    task: str
    condition: str = "given"         # given | hidden
    k: int = 10
    n_iter: int = 5
    steps: int = 8
    seed: int = 0
    data: str = "data/hda"
    out: str = "runs/hda/debug"
    val_limit: int | None = None
    feedback: str = "aggregate"      # aggregate | none
    test_splits: list = field(default_factory=lambda: ["test_small", "test_medium", "test_large", "test_xl", "test_stress"])


class Logger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True); self.f = open(path, "a")
    def __call__(self, kind, **kw):
        self.f.write(json.dumps(dict(t=round(time.time(), 3), kind=kind, **kw)) + "\n"); self.f.flush()


def run(cfg: RunConfig, client, judge_client=None):
    task = TASKS[cfg.task]
    data = Path(cfg.data); out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "log.jsonl")
    log("config", **cfg.__dict__)
    rng = random.Random(f"{cfg.seed}:{cfg.task}:{cfg.condition}:{cfg.k}")
    pool = load_split(data, "examples", cfg.task); rng.shuffle(pool)
    val = load_split(data, "val", cfg.task, cfg.val_limit)
    examples, best, history = [], dict(acc=-1.0, code=TEMPLATE, it=0), []
    statement = task.statement if cfg.condition == "given" else "(not yet stated)"
    tokens = dict(prompt=0, completion=0)
    for it in range(1, cfg.n_iter + 1):
        if len(pool) < cfg.k:
            pool = load_split(data, "examples", cfg.task)
            rng.shuffle(pool)
        new = [pool.pop() for _ in range(min(cfg.k, len(pool)))]
        examples += [dict(id=r["id"], input=(data / r["input_path"]).read_text(),
                          output=r["answer"], ref_value=r.get("ref_value")) for r in new]
        log("examples", it=it, ids=[r["id"] for r in new], answers=[r["answer"] for r in new])
        cur = examples[-len(new):] if new else []
        blocks = "\n".join(f"Example {i+1}:\n```input\n{show_input(e['input'])}```\nExpected output: {e['output']}\n"
                           for i, e in enumerate(cur))
        sys_msg = SYSTEM.replace("{template}", TEMPLATE)
        user = (STATEMENT_GIVEN.format(statement=task.statement) if cfg.condition == "given" else STATEMENT_HIDDEN)
        user += EXAMPLES.format(k=len(new), total=len(examples), blocks=blocks)
        fb = history[-1]["feedback"] if history and cfg.feedback == "aggregate" else "This is the first iteration." if not history else "(no feedback)"
        if best["it"] > 0:
            best_block = (f"Current best program (validation accuracy {max(best['acc'], 0.0):.3f}):\n"
                          f"```cpp\n{best['code']}\n```")
        else:
            best_block = "No valid program yet."
        user += ITERATION.format(it=it, n_iter=cfg.n_iter, feedback=fb,
                                 best_block=best_block, statement=statement, steps=cfg.steps)
        if cfg.condition == "hidden" and task.statement in user:
            raise RuntimeError("hidden prompt leaked the problem statement")
        msgs = [dict(role="system", content=sys_msg), dict(role="user", content=user)]
        log("prompt", it=it, system=sys_msg, user=user)
        code, submitted = None, None
        for step in range(1, cfg.steps + 1):
            r = client.chat(msgs)
            tokens["prompt"] += r["prompt_tokens"]; tokens["completion"] += r["completion_tokens"]
            text = r["content"]
            # some servers strip the stop sequence; restore it so the action parses
            if "<action>" in text and "</action>" not in text:
                text = text + "</action>"
            act = parse_action(text)
            code = last_block(text, "cpp") or code
            st = parse_statement(text)
            if st:
                statement = st
            log("reply", it=it, step=step, content=text, reasoning=r["reasoning"], action=act,
                statement=statement, code=code, prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"])
            msgs.append(dict(role="assistant", content=text))
            if act == "submit" and not code:
                obs = ("Invalid submission: no complete ```cpp block found. The evaluator scores only the code, "
                       "not the statement or prose. Write the full program in a ```cpp block and submit again.")
                msgs = [dict(role="system", content=sys_msg), dict(role="user", content=user),
                        dict(role="assistant", content=text),
                        dict(role="user", content=obs + f"\n\nCurrent statement: {statement}\n"
                             f"Current program:\n```cpp\n(no code yet)\n```\n({cfg.steps - step} steps left.)")]
                continue
            if act == "submit" and step < cfg.steps:
                ok, clog, _ = compile_cpp(code, out / f"it{it}" / f"pre_submit")
                if not ok:
                    obs = ("Your submission does not compile and was not submitted. Fix these errors and "
                           "call <action>test</action> or <action>submit</action> again:\n" + clog[-1800:])
                    msgs = [dict(role="system", content=sys_msg), dict(role="user", content=user),
                            dict(role="assistant", content=text),
                            dict(role="user", content=obs + f"\n\nCurrent statement: {statement}\n"
                                 f"Current program:\n```cpp\n{code}\n```\n({cfg.steps - step} steps left.)")]
                    continue
            if act == "submit" or step == cfg.steps:
                submitted = code; break
            obs = tool(act, text, code, examples, out / f"it{it}", log, it, step, task=cfg.task)
            # Keep the prompt bounded: drop older replies, keep only the current
            # program, statement and the latest tool result.
            state = (f"\n\nCurrent statement: {statement}\nCurrent program:\n```cpp\n"
                     f"{code or '(no code yet)'}\n```")
            msgs = [dict(role="system", content=sys_msg),
                    dict(role="user", content=user),
                    dict(role="assistant", content=text),
                    dict(role="user", content=obs + state + f"\n({cfg.steps - step} steps left.)")]
        # ---- score the submission on hidden validation
        res = score_submission(submitted, examples, val, data, out / f"it{it}", cfg, log, it)
        res["statement"] = statement
        history.append(res)
        if res["acc"] >= best["acc"] and submitted:
            best = dict(acc=res["acc"], code=submitted, it=it, statement=statement)
        log("iteration", it=it, val_acc=res["acc"], best_acc=best["acc"], best_it=best["it"], statement=statement,
            policy=res["policy"], verdicts=res.get("summary", {}).get("verdicts"))
    # ---- final harness and test evaluation
    (out / "best_solution.cpp").write_text(best["code"])
    final = dict(task=cfg.task, condition=cfg.condition, k=cfg.k, seed=cfg.seed, best_it=best["it"],
                 best_val_acc=best["acc"], final_statement=best.get("statement", statement),
                 val_curve=[h["acc"] for h in history], statements=[h["statement"] for h in history], tokens=tokens)
    if judge_client and cfg.condition == "hidden":
        j = judge_client.chat([dict(role="user", content=JUDGE.format(truth=task.statement, inferred=final["final_statement"]))])
        final["statement_judge"] = (j["content"].split() or ["?"])[0].strip(".:").upper()
        log("judge", content=j["content"], reasoning=j["reasoning"])
    ok, clog, exe = compile_cpp(best["code"], out / "final")
    final["test"] = {}
    if ok and best["it"] > 0:
        for sp in cfg.test_splits:
            if not (data / f"{sp}.jsonl").exists():
                continue
            if task.objective in ("max", "min"):
                rows, s = evaluate_exe_quality(exe, load_split(data, sp, cfg.task), data, cfg.task,
                                               permutation_check=5, seed=cfg.seed)
            else:
                rows, s = evaluate_exe(exe, load_split(data, sp, cfg.task), data, permutation_check=5, seed=cfg.seed)
            final["test"][sp] = s
            log("test", split=sp, summary=s, rows=rows)
    json.dump(final, open(out / "final.json", "w"), indent=2)
    return final


def tool(act, text, code, examples, wd, log, it, step, task=None):
    current_code = last_block(text, "cpp")
    if act in ("compile", "test", "run", "stress") and not current_code:
        return ("No ```cpp block in this reply. You must include the FULL program in a ```cpp block in the same "
                "reply as the action; a bare action is invalid and older code is not reused.")
    if act == "read":
        if step > 2:
            return ("You have read enough. Write the full ```cpp program now and call <action>test</action> "
                    "or <action>compile</action>.")
        rel = (last_block(text, "input") or "").strip().splitlines()[0:1]
        rel = rel[0].strip() if rel else ""
        if not rel:
            return "No ```input block with a file path found."
        path = (wd / rel).resolve()
        try:
            path.relative_to(wd.resolve())
        except ValueError:
            return "That path is outside the working directory."
        if not path.exists():
            return f"No such file: {rel}"
        return path.read_text()[:8000]
    if act not in ("compile", "test", "run", "stress"):
        return "No valid action found. Use <action>compile</action>, <action>test</action>, <action>run</action>, <action>read</action>, <action>stress</action> or <action>submit</action>."
    if not code:
        return "No ```cpp block found yet. Write the full program in a ```cpp block."
    ok, clog, exe = compile_cpp(code, wd / f"s{step}")
    if not ok:
        log("tool", it=it, step=step, action=act, compile_ok=False, compile_log=clog)
        return f"Compilation failed:\n{clog}"
    if act == "compile":
        log("tool", it=it, step=step, action=act, compile_ok=True)
        return "Compilation succeeded."
    if act == "test":
        from .generator import parse_text
        from .quality import VALIDATORS, normalized_quality, parse_nodes
        quality_task = task and TASKS[task].objective in ("max", "min")
        lines, outs = [], []
        for i, e in enumerate(examples):
            if quality_task:
                r = run_binary(exe, e["input"], 2.0, None)
                n, edges, q, d, w = parse_text(e["input"])
                nodes = parse_nodes(r.output) if r.verdict == "OK" else None
                feasible = bool(nodes is not None and VALIDATORS[task](nodes, n, edges))
                size = len(nodes) if feasible else 0
                ref = e.get("ref_value") or 0
                qual = normalized_quality(task, size, ref) if feasible else 0.0
                v = "AC" if qual >= 0.999999 else ("FEAS" if feasible else (
                    r.verdict if r.verdict in {"TLE", "RE", "MLE", "OLE"} else "WA"))
                outs.append(dict(id=e["id"], verdict=v,
                                 output=r.output.strip()[:200], value=size, ref_value=ref, quality=qual))
                lines.append(f"Example {i+1}: {'FEAS' if feasible else r.verdict}  size={size}  "
                             f"quality={qual:.3f}  best_known={ref}")
            else:
                r = run_binary(exe, e["input"], 2.0, e["output"])
                outs.append(dict(id=e["id"], verdict=r.verdict, output=r.output.strip()[:200], expected=e["output"]))
                lines.append(f"Example {i+1}: {r.verdict}  got={r.output.strip()[:80]!r} expected={e['output']!r}"
                             + (f" stderr={r.stderr[-200:]!r}" if r.stderr else ""))
        log("tool", it=it, step=step, action=act, compile_ok=True, example_outputs=outs)
        return "\n".join(lines)
    if act == "run":
        inp = last_block(text, "input")
        if not inp:
            return "No ```input block found."
        r = run_binary(exe, inp, 2.0)
        log("tool", it=it, step=step, action=act, compile_ok=True, input=inp[:2000], result=r.to_dict())
        return f"verdict={r.verdict} cpu={r.cpu_s}s output={r.output.strip()[:500]!r} stderr={r.stderr[-300:]!r}"
    gen = last_block(text, "gen")
    if not gen:
        return "No ```gen block found."
    gok, glog, gexe = compile_cpp(gen, wd / f"s{step}_gen")
    if not gok:
        return f"Generator compilation failed:\n{glog}"
    gen_viol = check_source(gen)
    if gen_viol:
        return f"Generator rejected by policy: {gen_viol}"
    gen_out = run_capture(gexe, 10.0)
    if gen_out is None:
        return "Generator failed or exceeded 10 s / 64 MiB."
    r = run_binary(exe, gen_out, 5.0)
    log("tool", it=it, step=step, action=act, compile_ok=True, gen=gen[:4000], gen_bytes=len(gen_out), result=r.to_dict())
    return f"verdict={r.verdict} cpu={r.cpu_s}s output={r.output.strip()[:200]!r} stderr={r.stderr[-300:]!r}"


def score_submission(code, examples, val, data, wd, cfg, log, it):
    task = TASKS[cfg.task]
    if not code:
        log("submission", it=it, ok=False, reason="no code")
        return dict(acc=0.0, policy=["no code submitted"], feedback="No program was submitted.")
    pol = check_source(code, [e["output"] for e in examples])
    ok, clog, exe = compile_cpp(code, wd / "submit")
    if pol or not ok:
        reason = pol if pol else ["compile error"]
        log("submission", it=it, ok=False, policy=pol, compile_log=None if ok else clog)
        fbk = (f"Your submission was REJECTED (score 0). Policy: {pol or 'ok'}. "
               + ("" if ok else f"Compiler output:\n{clog[-1500:]}"))
        return dict(acc=0.0, policy=reason, feedback=fbk)
    if task.objective in ("max", "min"):
        rows, s = evaluate_exe_quality(exe, val, data, cfg.task, permutation_check=5, seed=cfg.seed + it)
    else:
        rows, s = evaluate_exe(exe, val, data, permutation_check=5, seed=cfg.seed + it)
    ex_outs = [dict(id=e["id"], **{k: v for k, v in run_binary(exe, e["input"], 2.0, e["output"]).to_dict().items()
                                   if k in ("verdict", "output")}) for e in examples]
    log("submission", it=it, ok=True, code=code, val_summary=s, val_rows=rows, example_outputs=ex_outs)
    inv = s["invariance"]
    if task.objective in ("max", "min"):
        acc = s["quality"] if not inv or inv["failed"] == 0 else s["quality"] * (1 - inv["failed"] / inv["tried"])
        fbk = FEEDBACK_Q.format(q=s["quality"], n=s["n"], feas=s["feasible"], opt=s["optimal"],
                                verdicts=s["verdicts"],
                                by_bucket={k: v["quality"] for k, v in s["by_bucket"].items()},
                                by_family={k: v["quality"] for k, v in s["by_family"].items()},
                                max_cpu=s["max_cpu_s"],
                                inv=f"{inv['failed']} of {inv['tried']} relabeled graphs changed the score" if inv else "n/a",
                                policy="passed")
        return dict(acc=round(acc, 4), policy=[], feedback=fbk, summary=s)
    acc = s["accuracy"] if not inv or inv["failed"] == 0 else s["accuracy"] * (1 - inv["failed"] / inv["tried"])
    fbk = FEEDBACK.format(acc=s["accuracy"], n=s["n"], verdicts=s["verdicts"],
                          by_bucket={k: v["acc"] for k, v in s["by_bucket"].items()},
                          by_family={k: v["acc"] for k, v in s["by_family"].items()},
                          max_cpu=s["max_cpu_s"], inv=f"{inv['failed']} of {inv['tried']} relabeled graphs changed the answer" if inv else "n/a",
                          policy="passed")
    return dict(acc=round(acc, 4), policy=[], feedback=fbk, summary=s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--condition", default="given", choices=["given", "hidden"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="data/hda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--base-url", default=os.environ.get("HDA_BASE_URL", "http://localhost:8000/v1"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="0 = no output cap (stop at </action> or context limit)")
    ap.add_argument("--reasoning-effort", default="medium")
    ap.add_argument("--judge-model", default=None, help="defaults to --model")
    ap.add_argument("--feedback", default="aggregate", choices=["aggregate", "none"])
    ap.add_argument("--test-splits", nargs="*", default=None)
    a = ap.parse_args()
    client = OpenAIClient(a.model, a.base_url, a.temperature, a.max_tokens, a.reasoning_effort)
    judge = OpenAIClient(a.judge_model or a.model, a.base_url, 0.0, 2000, "low")
    cfg = RunConfig(task=a.task, condition=a.condition, k=a.k, n_iter=a.n_iter, steps=a.steps, seed=a.seed,
                    data=a.data, out=a.out, feedback=a.feedback)
    if a.test_splits is not None:
        cfg.test_splits = a.test_splits
    print(json.dumps(run(cfg, client, judge), indent=2)[:3000])


if __name__ == "__main__":
    main()
