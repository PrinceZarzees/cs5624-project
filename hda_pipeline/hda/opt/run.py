"""Run one of the four HDA prompt optimizers (GEPA, SkillOpt, Meta-Harness, EvoSkill)."""
from __future__ import annotations
import argparse, json, random, re, statistics
from pathlib import Path
from .common import Chat, INSTANCES, TASKS, candidate_text, evaluate, save_candidate, traces
from .validate import validate_candidate

FIELDS = ("strategy","hidden_procedure","iteration_note")
CONTRACT = """Editable text only. Do not name a benchmark task or a textbook algorithm
(shortest path, reachability, components, cycle, diameter, common neighbors,
triangle, bipartite, spanning tree, max flow, clique, independent set, vertex
cover, Dijkstra, Bellman, Floyd, Warshall, Kruskal, Prim, Dinic, Edmonds, Karp,
Ford, Fulkerson, Tarjan, Kosaraju, Johnson, Hopcroft). Do not include 3+ digit
numbers, pasted solvers, or instructions that weaken the fixed contract (stdlib
only, no hardcoding, stdin/stdout, limits). Generic techniques (BFS, DFS,
union-find, priority queue, adjacency list, fast I/O, testing) are allowed."""


def extract_json(text):
    for m in re.finditer(r"\{.*\}", text, re.S):
        try: return json.loads(m.group(0))
        except Exception: pass
    for m in re.finditer(r"\[.*\]", text, re.S):
        try: return json.loads(m.group(0))
        except Exception: pass
    return None


def extract_tag(text, tag):
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.S)
    return m.group(1).strip() if m else None


def better(child, parent, child_text, parent_text):
    return child >= parent + 0.02 or (abs(child-parent) <= 0.02 and len(child_text) < len(parent_text))


def baseline_data(out):
    p = Path("../experiments/hda/optimizers/baseline")
    if (p/"baseline_A.json").exists():
        d = json.loads((p/"baseline_A.json").read_text())
        rows = json.loads((p/"rollouts.json").read_text()) if (p/"rollouts.json").exists() else []
        return d, rows
    rows, mean, _ = evaluate({}, {"model":"qwen3-4b","base_url":"http://127.0.0.1:8002/v1"},
                             out/"parent0", instances=INSTANCES, workers=8, tag="parent")
    return {"overall":mean,"per_instance":{}}, rows


def pick_minibatch(baseline, rng, k=4):
    keys = list(baseline.get("per_instance") or {})
    if not keys: return rng.sample(INSTANCES, k)
    weights = [(1 - baseline["per_instance"][x]) + 0.05 for x in keys]
    out = []
    for _ in range(k):
        x = rng.choices(keys, weights=weights)[0]
        t, c = x.split("|")
        out.append((t, c, rng.randint(100,199)))
    return out


def prompt_for(opt, candidate, rows, instruction):
    return f"""You optimize the prompt of an HDA that writes C++17 graph solvers.
The fixed contract is read-only.
{CONTRACT}

Current editable text:
<<<
{json.dumps({k:candidate.get(k,'') for k in FIELDS}, indent=1)}
>>>

Recent rollout traces (validation aggregate only):
{traces(rows)}

{instruction}
Return ONLY the requested JSON or tagged text.
"""


def run_gepa(args, chat, out, parent, parent_rows, parent_mean, base, rng):
    pool = [({}, parent_mean, parent_rows)]
    best = ({}, parent_mean)
    hist = open(out/"history.jsonl","a")
    for step in range(1, args.rounds+1):
        # Pareto-ish parent: candidate best on at least one instance, else best mean
        front = pool
        cand, mean, rows = rng.choice(front)
        field = FIELDS[(step-1) % 3]
        mb = pick_minibatch(base, rng)
        mb_rows = [r for r in rows if (r["task"],r["condition"],r["seed"]) in mb]
        text = chat.ask(prompt_for("gepa", cand, mb_rows or rows,
            f"Rewrite ONLY the `{field}` field. Keep it general. Return <new_text>...</new_text>."))
        new = extract_tag(text, "new_text") or (extract_json(text) or {}).get(field, "")
        child = dict(cand); child[field] = new
        ok = validate_candidate(child)
        row = dict(step=step, field=field, accepted=False, validator=ok, mean=None, best=mean, raw=text[:2000])
        if ok["ok"]:
            _, cm, _ = evaluate(child, args.cfg, out, instances=INSTANCES, workers=args.workers, tag=f"step{step}")
            row["mean"] = cm
            if better(cm, mean, candidate_text(child), candidate_text(cand)):
                pool.append((child, cm, rows)); best = (child, cm); row["accepted"] = True
                save_candidate(child, out/"candidates"/f"{step:02d}.json")
        row["best"] = best[1]
        hist.write(json.dumps(row)+"\n"); hist.flush()
        print(opt_step("GEPA", step, row, best[1]), flush=True)
    return best


def opt_step(name, step, row, best):
    return f"{name} {step}: mean={row.get('mean')} accepted={row['accepted']} best={best} {row['validator'].get('reasons','')[:80]}"


def run_skillopt(args, chat, out, parent, parent_rows, parent_mean, base, rng):
    inc = dict(parent); inc_mean = parent_mean; best = (dict(inc), inc_mean)
    hist = open(out/"history.jsonl","a")
    for step in range(1, args.rounds+1):
        rows = parent_rows if step == 1 else evaluate(inc, args.cfg, out, instances=INSTANCES, workers=args.workers, tag=f"inc{step}")[0]
        text = chat.ask(prompt_for("skillopt", inc, rows,
            'Return JSON edits: [{"op":"add|modify|delete","field":"strategy|hidden_procedure|iteration_note","index":N,"text":"..."}]. '
            'Rules are newline-separated. 1 to 3 edits.'))
        edits = extract_json(text) or []
        child = dict(inc); reasons = []
        for e in edits if isinstance(edits,list) else []:
            f = e.get("field"); idx = e.get("index", 0)
            lines = [x for x in str(child.get(f,"")).splitlines() if x.strip()]
            if e.get("op") == "add": lines.append(str(e.get("text","")))
            elif e.get("op") == "delete" and 0 <= idx < len(lines): lines.pop(idx)
            elif e.get("op") == "modify" and 0 <= idx < len(lines): lines[idx] = str(e.get("text",""))
            child[f] = "\n".join(lines)
        ok = validate_candidate(child)
        row = dict(step=step, edits=edits, accepted=False, validator=ok, mean=None, best=inc_mean, raw=text[:2000])
        if ok["ok"]:
            _, cm, _ = evaluate(child, args.cfg, out, instances=INSTANCES, workers=args.workers, tag=f"step{step}")
            row["mean"] = cm
            if better(cm, inc_mean, candidate_text(child), candidate_text(inc)):
                inc, inc_mean = child, cm; best = (dict(child), cm); row["accepted"] = True
                save_candidate(child, out/"candidates"/f"{step:02d}.json")
        row["best"] = inc_mean
        hist.write(json.dumps(row)+"\n"); hist.flush(); print(opt_step("SkillOpt", step, row, inc_mean), flush=True)
    return best


def run_meta(args, chat, out, parent, parent_rows, parent_mean, base, rng):
    hist_rows = []
    best = (dict(parent), parent_mean)
    h = open(out/"history.jsonl","a")
    for step in range(1, args.rounds+1):
        rows = parent_rows if step == 1 else evaluate(best[0], args.cfg, out, instances=INSTANCES, workers=args.workers, tag=f"meta{step}")[0]
        text = chat.ask(prompt_for("meta", best[0], rows,
            'Write a full candidate. Return <candidate>{"strategy":"...","hidden_procedure":"...","iteration_note":"..."}</candidate>.'))
        obj = extract_json(extract_tag(text,"candidate") or text) or {}
        child = {k:str(obj.get(k,"") or "") for k in FIELDS}
        ok = validate_candidate(child)
        row = dict(step=step, accepted=False, validator=ok, mean=None, best=best[1], raw=text[:2000])
        if ok["ok"]:
            _, cm, _ = evaluate(child, args.cfg, out, instances=INSTANCES, workers=args.workers, tag=f"step{step}")
            row["mean"] = cm
            if better(cm, best[1], candidate_text(child), candidate_text(best[0])):
                best = (child, cm); row["accepted"] = True; save_candidate(child, out/"candidates"/f"{step:02d}.json")
        row["best"] = best[1]; h.write(json.dumps(row)+"\n"); h.flush(); print(opt_step("Meta", step, row, best[1]), flush=True)
    return best


def run_evoskill(args, chat, out, parent, parent_rows, parent_mean, base, rng):
    frontier = [([], parent_mean)]
    h = open(out/"history.jsonl","a")
    best = (dict(parent), parent_mean)
    for step in range(1, args.rounds+1):
        skills, smean = rng.choice(frontier)
        rendered = "\n".join(f"Skill {s['name']}: use when {s['when_to_use']}. {s['instructions']}" for s in skills)
        cand = {"strategy": rendered}
        rows = evaluate(cand, args.cfg, out, instances=INSTANCES, workers=args.workers, tag=f"evo{step}")[0]
        text = chat.ask(prompt_for("evoskill", cand, rows,
            'Diagnose the failures and add or refine one skill. Return JSON {"name":...,"when_to_use":...,"instructions":...}.'))
        obj = extract_json(text) or {}
        new_skills = list(skills)
        if obj.get("name"):
            s = {k:str(obj.get(k,""))[:600] for k in ("name","when_to_use","instructions")}
            if len(new_skills) < 6: new_skills.append(s)
            else: new_skills[rng.randrange(len(new_skills))] = s
        text2 = "\n".join(f"Skill {s['name']}: use when {s['when_to_use']}. {s['instructions']}" for s in new_skills)
        child = {"strategy": text2}
        ok = validate_candidate(child)
        row = dict(step=step, accepted=False, validator=ok, mean=None, best=smean, raw=text[:2000], skills=new_skills)
        if ok["ok"]:
            _, cm, _ = evaluate(child, args.cfg, out, instances=INSTANCES, workers=args.workers, tag=f"step{step}")
            row["mean"] = cm
            if cm >= min(x[1] for x in frontier) + 0.02 or len(frontier) < 3:
                frontier.append((new_skills, cm)); frontier = sorted(frontier,key=lambda x:-x[1])[:3]
                row["accepted"] = True
                if cm > best[1]: best = ({"strategy":text2}, cm); save_candidate(best[0], out/"candidates"/f"{step:02d}.json")
        row["best"] = best[1]; h.write(json.dumps(row)+"\n"); h.flush(); print(opt_step("Evo", step, row, best[1]), flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", required=True, choices=["gepa","skillopt","meta_harness","evoskill"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="one round, two instances, one HDA iteration")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out/"candidates").mkdir(exist_ok=True)
    a.cfg = {"model":a.model, "base_url":a.base_url}
    if a.dry_run:
        global INSTANCES
        INSTANCES = [("shortest_path","given",200), ("bipartite_check","hidden",200)]
        a.rounds = 1
        a.cfg["n_iter"] = 1
    base, parent_rows = baseline_data(out)
    parent_mean = float(base.get("overall") or 0.0)
    chat = Chat(a.model, a.base_url)
    rng = random.Random(a.seed)
    fn = {"gepa":run_gepa,"skillopt":run_skillopt,"meta_harness":run_meta,"evoskill":run_evoskill}[a.optimizer]
    best, best_mean = fn(a, chat, out, {}, parent_rows, parent_mean, base, rng)
    save_candidate(best, out/"best.json")
    # 3-seed re-check (seeds 201,202; seed 200 comes from the best A score)
    inst = ([("shortest_path","given",201)]
            if a.dry_run else [(t,c,s) for t in TASKS for c in ("given","hidden") for s in (201,202)])
    rows, m, _ = evaluate(best, a.cfg, out, instances=inst, workers=a.workers, tag="recheck")
    payload = dict(best_fields=best, best_A=best_mean, recheck_mean=m, n=len(rows),
                   proposer_calls=chat.calls, proposer_prompt_tokens=chat.prompt_tokens,
                   proposer_completion_tokens=chat.completion_tokens)
    (out/"final_optimizer.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2))


if __name__ == "__main__":
    main()
