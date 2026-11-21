#!/usr/bin/env python3
"""Aggregate E1/E2 runs once the job finishes."""
import csv, json, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from graphharness.stats import wilson_ci
R=Path('experiments/final')

def read_summary(p):
    d=json.loads(Path(p).read_text()); o=d['overall']
    n=d['n']; err=d.get('errors') or 0
    return {'harness':d['harness'],'split':d['split'],'n':n,'accuracy':o['accuracy'],
            'parsed':o['parsed'],'feasible':o['feasible'],'errors':err,
            'tokens':(d.get('tokens') or {}).get('total'),
            'agent_error_rate': err / n if n else 0.0}

def read_scores(path):
    return {r['id']: bool(r.get('correct')) for r in map(json.loads, open(path)) if r}

def write(path,rows,fields):
    rows=[{k: r.get(k) for k in fields} for r in rows]
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

# E1
rows=[]
for h in ['baseline_a','baseline_b','baseline_b_v2']:
    for rep in [1,2,3]:
        p=R/f'E1_heldout/{h}/rep{rep}/summary.json'
        if p.exists():
            r=read_summary(p); r['repeat']=rep; rows.append(r)
if rows:
    write(R/'E1_heldout/summary_long.csv',rows,['harness','repeat','split','n','accuracy','parsed','feasible','errors','tokens'])
    agg=[]
    for h in ['baseline_a','baseline_b','baseline_b_v2']:
        vals=[r['accuracy'] for r in rows if r['harness']==h]; n=sum(r['n'] for r in rows if r['harness']==h)
        if vals:
            lo,hi=wilson_ci(round(sum(vals)/len(vals)*n),n)
            agg.append({'harness':h,'repeats':len(vals),'mean':statistics.mean(vals),
                        'std':statistics.pstdev(vals),'min':min(vals),'max':max(vals),
                        'wilson_lo':lo,'wilson_hi':hi})
    write(R/'E1_heldout/summary.csv',agg,['harness','repeats','mean','std','min','max','wilson_lo','wilson_hi'])

    # per-task rows across repeats
    task_rows=[]
    for h in ['baseline_a','baseline_b','baseline_b_v2']:
        tasks={}
        for rep in [1,2,3]:
            p=R/f'E1_heldout/{h}/rep{rep}/summary.json'
            if not p.exists(): continue
            for t in json.loads(p.read_text())['by_task']:
                tasks.setdefault(t['task'],[]).append(t['accuracy'])
        for task,vals in sorted(tasks.items()):
            task_rows.append({'harness':h,'task':task,'repeats':len(vals),
                              'accuracy_mean':statistics.mean(vals),
                              'accuracy_std':statistics.pstdev(vals)})
    write(R/'E1_heldout/summary_by_task.csv',task_rows,
          ['harness','task','repeats','accuracy_mean','accuracy_std'])

    # paired comparisons on identical problem ids, per repeat
    from graphharness.stats import mcnemar
    comp=[]
    for rep in [1,2,3]:
        a=read_scores(R/f'E1_heldout/baseline_b/rep{rep}/scores.jsonl')
        b=read_scores(R/f'E1_heldout/baseline_b_v2/rep{rep}/scores.jsonl')
        ids=sorted(set(a)&set(b))
        if ids:
            r=mcnemar([a[i] for i in ids],[b[i] for i in ids])
            r.update(repeat=rep,n=len(ids),acc_b=sum(a[i] for i in ids)/len(ids),
                     acc_v2=sum(b[i] for i in ids)/len(ids),comparison='baseline_b_vs_v2')
            comp.append(r)
    if comp:
        write(R/'E1_heldout/mcnemar.csv',comp,['comparison','repeat','n','acc_b','acc_v2','a_only','b_only','p_value'])

# E2 scaling: existing = rep1, final = rep2/3
rows=[]
for h in ['baseline_a','baseline_b','baseline_b_v2']:
    for split in ['test_small','test_medium','test_large','test_stress']:
        p=Path(f'experiments/qwen3_4b_scaling_full/scaling/{h}/{split}/summary.json')
        if p.exists():
            r=read_summary(p); r['repeat']=1; rows.append(r)
        for rep in [2,3,4]:
            p=R/f'E2_repeats/{h}/rep{rep}/{split}/summary.json'
            if p.exists():
                r=read_summary(p); r['repeat']=rep; rows.append(r)
if rows:
    write(R/'E2_repeats/summary_long.csv',rows,
          ['harness','repeat','split','n','accuracy','parsed','feasible','errors','agent_error_rate','tokens'])
    agg=[]
    for h in ['baseline_a','baseline_b','baseline_b_v2']:
        for split in ['test_small','test_medium','test_large','test_stress']:
            allr=[r for r in rows if r['harness']==h and r['split']==split]
            vals=[r['accuracy'] for r in allr]
            if vals:
                agg.append({'harness':h,'split':split,'repeats':len(vals),
                            'mean':statistics.mean(vals),'std':statistics.pstdev(vals),
                            'min':min(vals),'max':max(vals),
                            'n':allr[0]['n'],
                            'flagged_runs':';'.join(str(r['repeat']) for r in allr if r['agent_error_rate']>0.05)})
    from graphharness.stats import wilson_ci
    for a in agg:
        lo,hi=wilson_ci(round(a['mean']*a['n']),a['n']); a['wilson_lo']=lo; a['wilson_hi']=hi
    write(R/'E2_repeats/summary.csv',agg,
          ['harness','split','repeats','mean','std','min','max','n','wilson_lo','wilson_hi','flagged_runs'])

    # paired McNemar between harnesses on identical problems, per split/repeat
    from graphharness.stats import mcnemar
    comp=[]
    pairs=[('baseline_b_v2','baseline_b'),('baseline_b_v2','baseline_a'),('baseline_b','baseline_a')]
    for x,y in pairs:
        for split in ['test_small','test_medium','test_large','test_stress']:
            for rep in [1,2,3]:
                px=(Path(f'experiments/qwen3_4b_scaling_full/scaling/{x}/{split}/scores.jsonl') if rep==1
                    else R/f'E2_repeats/{x}/rep{rep}/{split}/scores.jsonl')
                py=(Path(f'experiments/qwen3_4b_scaling_full/scaling/{y}/{split}/scores.jsonl') if rep==1
                    else R/f'E2_repeats/{y}/rep{rep}/{split}/scores.jsonl')
                if not (px.exists() and py.exists()): continue
                a,b=read_scores(px),read_scores(py); ids=sorted(set(a)&set(b))
                if not ids: continue
                r=mcnemar([a[i] for i in ids],[b[i] for i in ids])
                r.update(harness_x=x,harness_y=y,split=split,repeat=rep,n=len(ids),
                         acc_x=sum(a[i] for i in ids)/len(ids),
                         acc_y=sum(b[i] for i in ids)/len(ids))
                comp.append(r)
    if comp:
        write(R/'E2_repeats/mcnemar.csv',comp,
              ['harness_x','harness_y','split','repeat','n','acc_x','acc_y','a_only','b_only','p_value'])
    gaps=[]
    for h in ['baseline_a','baseline_b','baseline_b_v2']:
        srows=[a for a in agg if a['harness']==h and a['split']=='test_small']
        lrows=[a for a in agg if a['harness']==h and a['split']=='test_large']
        if srows and lrows:
            sm, lg=srows[0], lrows[0]
            se=(sm['std']**2/sm['repeats']+lg['std']**2/lg['repeats'])**0.5
            gaps.append({'harness':h,'repeat':'mean',
                         'delta_gen':sm['mean']-lg['mean'],'std':se,
                         'small_mean':sm['mean'],'large_mean':lg['mean'],
                         'n_small_runs':sm['repeats'],'n_large_runs':lg['repeats']})
    write(R/'E2_repeats/gen_gap.csv',gaps,
          ['harness','repeat','delta_gen','std','small_mean','large_mean','n_small_runs','n_large_runs'])

# E2 tool access
rows=[]
for c in ['baseline_a_tools','baseline_a_text','baseline_b_v2_tools','baseline_b_v2_text']:
    for split in ['test_small','test_medium','test_large']:
        p=Path(f'experiments/review_tool_access/runs/{c}/{split}/summary.json')
        if p.exists(): r=read_summary(p); r.update(condition=c,repeat=1); rows.append(r)
        for rep in [2,3]:
            p=R/f'E2_tool_access/rep{rep}/{c}/{split}/summary.json'
            if p.exists(): r=read_summary(p); r.update(condition=c,repeat=rep); rows.append(r)
if rows:
    write(R/'E2_tool_access/summary.csv',rows,
          ['condition','split','repeat','n','accuracy','parsed','feasible','errors','agent_error_rate','tokens'])
print('aggregated E1/E2')
