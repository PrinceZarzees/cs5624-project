"""EvoSkill-style skill-library evolution baseline.

A skill is a named trigger, instruction, and optional Python helper preamble.
Candidates append selected skills to a base harness; the optimizer maintains a
small population by validation accuracy. This is a faithful lightweight
baseline, not a reproduction of the original EvoSkill implementation.
"""
from __future__ import annotations
import argparse, json, random, re, time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from .agent import Harness
from .cli_common import add_model_arguments, build_model, harness_from_arg
from .data_loader import DEFAULT_DATA_DIR, load_split
from .models import Model
from .optimizer import (CRITIC_SYSTEM, OptimizerConfig, _append_history, _evaluate,
                        accuracy, critic_prompt, extract_proposal, sample_records)

@dataclass
class Skill:
    name: str
    trigger: str
    instruction: str
    preamble: str = ""

@dataclass
class EvoSkillConfig:
    rounds: int = 10
    train_per_task: int = 4
    val_per_task: int = 10
    max_steps: int | None = None
    seed: int = 5624
    population: int = 4
    max_skills: int = 8
    critic_temperature: float = 0.7

def apply_skills(base: Harness, skills: list[Skill]) -> Harness:
    rules = list(base.rules)
    for s in skills:
        rules.append(f"{s.trigger}: {s.instruction}")
    preamble = "\n\n".join(s.preamble for s in skills if s.preamble)
    if base.tool_preamble:
        preamble = base.tool_preamble + ("\n\n" + preamble if preamble else "")
    return base.copy(rules=rules, tool_preamble=preamble,
                     name=f"{base.name}-evoskill-{len(skills)}")

def validate_skill(proposal: dict, current: list[Skill], config: EvoSkillConfig):
    skill = proposal.get("skill")
    if not isinstance(skill, dict):
        return None, "missing skill object"
    name=str(skill.get("name","")).strip()
    trigger=str(skill.get("trigger","")).strip()
    instruction=str(skill.get("instruction","")).strip()
    preamble=str(skill.get("preamble","") or "")
    if not name or not trigger or not instruction:
        return None, "skill needs name, trigger, instruction"
    if len(current) >= config.max_skills:
        return None, "skill budget exhausted"
    text=(trigger+" "+instruction).lower()
    if re.search(r"\bnode\s+\d+\b", text):
        return None, "skill mentions a specific node id"
    if any(w in preamble for w in ["subprocess","os.system","socket","requests.","__import__"]):
        return None, "skill preamble uses a forbidden API"
    return Skill(name,trigger,instruction,preamble), str(proposal.get("reason",""))

def optimize_evoskill(start: Harness, actor: Model, critic: Model, train: list[dict],
                      val: list[dict], out_dir: str | Path, config: EvoSkillConfig):
    out=Path(out_dir); (out/"candidates").mkdir(parents=True,exist_ok=True)
    (out/"checkpoints").mkdir(parents=True,exist_ok=True)
    hp=out/"history.jsonl"
    if hp.exists(): hp.unlink()
    opt=OptimizerConfig(max_steps=config.max_steps,seed=config.seed)
    val_records=sample_records(val,config.val_per_task,random.Random(config.seed+1))
    population=[([],start,_evaluate(start,actor,val_records,opt))]
    best=start; best_val=accuracy(population[0][2]); total=sum(r.agent.total_tokens for r in population[0][2] if r.agent)
    history=[{"round":0,"role":"initial","val_accuracy":best_val,"best_val_accuracy":best_val,"tokens":total}]
    _append_history(out,history[0]); best.save(out/"best_harness.json")
    started=time.perf_counter()
    for rnd in range(1,config.rounds+1):
        skills,current,current_results=max(population,key=lambda x:accuracy(x[2]))
        train_records=sample_records(train,config.train_per_task,random.Random(config.seed*1000+rnd))
        train_results=_evaluate(current,actor,train_records,opt); total+=sum(r.agent.total_tokens for r in train_results if r.agent)
        critique=critic.chat([
            {"role":"system","content":CRITIC_SYSTEM+'\n\nReturn a skill object under "skill" with name, trigger, instruction, optional preamble.'},
            {"role":"user","content":critic_prompt(current,train_results,opt)+"\nPropose one reusable skill."},
        ],temperature=config.critic_temperature)
        total+=critique.prompt_tokens+critique.completion_tokens
        proposal=None
        for chunk in reversed(re.findall(r"\{.*\}", critique.text, re.DOTALL)):
            try:
                obj=json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(obj,dict) and "skill" in obj:
                proposal=obj; break
        if proposal is None:
            proposal=extract_proposal(critique.text)
        row={"round":rnd,"train_accuracy":accuracy(train_results)}
        if proposal is None:
            row.update(accepted=False,reason="could not parse proposal",best_val_accuracy=best_val)
        else:
            skill,reason=validate_skill(proposal,skills,config)
            if skill is None:
                row.update(accepted=False,reason=reason,proposal=proposal,best_val_accuracy=best_val)
            else:
                cand_skills=skills+[skill]; cand=apply_skills(start,cand_skills)
                cand.save(out/"candidates"/f"round_{rnd:02d}.json")
                res=_evaluate(cand,actor,val_records,opt); total+=sum(r.agent.total_tokens for r in res if r.agent)
                va=accuracy(res); row.update(accepted=va>best_val,reason=reason,proposal=proposal,val_accuracy=va,
                                              best_val_accuracy=max(best_val,va),skill=asdict(skill))
                population.append((cand_skills,cand,res))
                population=sorted(population,key=lambda x:accuracy(x[2]),reverse=True)[:config.population]
                if va>best_val:
                    best,best_val=cand,va; best.save(out/"best_harness.json"); best.save(out/"checkpoints"/f"round_{rnd:02d}.json")
        row["tokens"]=total; history.append(row); _append_history(out,row)
        print(f"EvoSkill {rnd}/{config.rounds}: best {best_val:.3f} {'ACCEPT' if row.get('accepted') else 'reject'} {row.get('reason','')[:70]}",flush=True)
    summary={"start_harness":start.to_dict(),"best_harness":best.to_dict(),"config":config.__dict__,
             "initial_val_accuracy":accuracy(population[0][2]) if False else history[0]["val_accuracy"],
             "best_val_accuracy":best_val,"total_tokens":total,"duration_s":time.perf_counter()-started,
             "started_at":datetime.now(timezone.utc).isoformat(),"history":history,"optimizer":"evoskill"}
    (out/"evoskill_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start",default="baseline_a"); ap.add_argument("--train-split",default="train")
    ap.add_argument("--val-split",default="val"); ap.add_argument("--data-dir",default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--rounds",type=int,default=10); ap.add_argument("--train-per-task",type=int,default=4)
    ap.add_argument("--val-per-task",type=int,default=10); ap.add_argument("--max-steps",type=int)
    ap.add_argument("--seed",type=int,default=5624); ap.add_argument("--out",default="experiments/final/E4_evoskill")
    ap.add_argument("--critic-model"); ap.add_argument("--critic-base-url")
    add_model_arguments(ap); args=ap.parse_args(argv)
    actor=build_model(args); ca=argparse.Namespace(**vars(args)); ca.model=args.critic_model or args.model
    ca.base_url=args.critic_base_url or args.base_url; ca.temperature=0.7; critic=build_model(ca)
    summary=optimize_evoskill(harness_from_arg(args.start),actor,critic,
        load_split(args.train_split,args.data_dir),load_split(args.val_split,args.data_dir),args.out,
        EvoSkillConfig(rounds=args.rounds,train_per_task=args.train_per_task,val_per_task=args.val_per_task,
                       max_steps=args.max_steps,seed=args.seed))
    print(json.dumps({k:summary[k] for k in ("initial_val_accuracy","best_val_accuracy","total_tokens")},indent=2))

if __name__=="__main__": main()
