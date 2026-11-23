"""Shared optimizer utilities: proposer client, candidate I/O, rollout scoring."""
from __future__ import annotations
import json, os, time
from pathlib import Path
from .rollout import rollout_many

TASKS = ["shortest_path","reachability","connected_components","cycle_detection","diameter",
         "common_neighbors","triangle_count","bipartite_check","mst_weight","max_flow"]
INSTANCES = [(t,c,200) for t in TASKS for c in ("given","hidden")]
FIELDS = ("strategy","hidden_procedure","iteration_note")


class Chat:
    def __init__(self, model, base_url, temperature=0.7, max_tokens=8192):
        from openai import OpenAI
        self.c = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY","EMPTY"))
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.prompt_tokens = self.completion_tokens = self.calls = 0
    def ask(self, prompt):
        r = self.c.chat.completions.create(model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=self.temperature, max_tokens=self.max_tokens)
        self.calls += 1
        u = r.usage
        self.prompt_tokens += getattr(u,"prompt_tokens",0)
        self.completion_tokens += getattr(u,"completion_tokens",0)
        return r.choices[0].message.content or ""


def candidate_text(c):
    return "\n".join(str(c.get(k,"") or "") for k in FIELDS)


def save_candidate(c, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k:str(c.get(k,"") or "") for k in FIELDS}, indent=2)+"\n")


def evaluate(candidate, cfg, out, instances=INSTANCES, workers=16, tag="eval"):
    rows = rollout_many(candidate, instances, Path(out)/"rollouts", workers=workers, config=cfg)
    vals = [r["val_opt_acc"] for r in rows]
    mean = sum(vals)/len(vals) if vals else 0.0
    ledger = dict(tag=tag, n=len(rows), mean=round(mean,4),
                  hda_llm_calls=sum(r["llm_calls"] for r in rows),
                  hda_prompt_tokens=sum(r["tokens_prompt"] for r in rows),
                  hda_completion_tokens=sum(r["tokens_completion"] for r in rows),
                  wall_s=round(sum(r["wall_s"] for r in rows),1))
    with open(Path(out)/"ledger.jsonl","a") as f: f.write(json.dumps(ledger)+"\n")
    return rows, mean, vals


def traces(rows, max_chars=18000):
    parts = []
    for r in rows:
        parts.append(f"### {r['task']} / {r['condition']} seed {r['seed']} val_opt={r['val_opt_acc']:.3f}\n{r['trace']}")
    return "\n\n".join(parts)[:max_chars]


def accept(child, parent, cfg):
    if child >= parent + 0.02:
        return True
    if abs(child-parent) <= 0.02 and len(candidate_text(cfg)) < len(candidate_text(parent)):
        return True
    return False
