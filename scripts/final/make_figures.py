#!/usr/bin/env python3
"""Regenerate main report figures from stored summaries."""
import csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; F=ROOT/'report/figures'; F.mkdir(exist_ok=True)
colors={'baseline_a':'#4C78A8','baseline_b':'#F58518','baseline_b_v2':'#54A24B'}

def save(fig,name):
    for ext in ['pdf','png']: fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',dpi=200)
    plt.close(fig)

# scaling (mean over repeats with std error bars when E2 is available)
s=json.load(open(ROOT/'experiments/qwen3_4b_scaling_full/scaling/scaling_summary.json'))
e2=ROOT/'experiments/final/E2_repeats/summary.csv'
order=['test_small','test_medium','test_large','test_stress']
mean_nodes={r['split']:r['mean_nodes'] for h in ['baseline_a'] for r in s['runs'] if r['harness']==h}
fig,ax=plt.subplots(figsize=(6,3.6))
if e2.exists():
    agg=list(csv.DictReader(open(e2)))
    for h in ['baseline_a','baseline_b','baseline_b_v2']:
        rows=sorted([r for r in agg if r['harness']==h], key=lambda r: mean_nodes[r['split']])
        ax.errorbar([mean_nodes[r['split']] for r in rows],[float(r['mean']) for r in rows],
                    yerr=[float(r['std']) for r in rows],marker='o',capsize=3,label=h,color=colors[h])
else:
    for h in ['baseline_a','baseline_b','baseline_b_v2']:
        rows=[r for r in s['runs'] if r['harness']==h]
        ax.plot([r['mean_nodes'] for r in rows],[r['accuracy'] for r in rows],marker='o',label=h,color=colors[h])
ax.set_xscale('log'); ax.set_xlabel('mean nodes'); ax.set_ylabel('accuracy'); ax.set_ylim(0.55,1.0); ax.grid(alpha=.3); ax.legend(); save(fig,'fig_scaling')

# gen gap (mean over repeats when available)
gg=ROOT/'experiments/final/E2_repeats/gen_gap.csv'
if gg.exists():
    rows=list(csv.DictReader(open(gg))); g=[]
    for h in ['baseline_a','baseline_b','baseline_b_v2']:
        vals=[float(r['delta_gen']) for r in rows if r['harness']==h]
        g.append({'harness':h,'delta_gen':sum(vals)/len(vals),
                  'std':(sum((v-sum(vals)/len(vals))**2 for v in vals)/len(vals))**0.5})
else:
    g=s['generalization_gaps']; g=[dict(x,std=0) for x in g]
fig,ax=plt.subplots(figsize=(5,3.2))
ax.bar([x['harness'] for x in g],[x['delta_gen'] for x in g],yerr=[x.get('std',0) for x in g],
       capsize=3,color=[colors[x['harness']] for x in g])
ax.axhline(0,color='k',lw=.8); ax.set_ylabel('small - large accuracy'); save(fig,'fig_gen_gap')

# tool access: mean over 3 repeats unless only the frozen run exists
conds=['baseline_a_tools','baseline_a_text','baseline_b_v2_tools','baseline_b_v2_text']
labels=['A tools','A text','Bv2 tools','Bv2 text']; splits=['test_small','test_medium','test_large']
ta=ROOT/'experiments/final/E2_tool_access/summary.csv'
runs=list(csv.DictReader(open(ta))) if ta.exists() else []
fig,ax=plt.subplots(figsize=(7,3.4)); w=.2
for j,sp in enumerate(splits):
    vals=[]; errs=[]
    for c in conds:
        rs=[r for r in runs if r['condition']==c and r['split']==sp]
        if rs:
            v=[float(r['accuracy']) for r in rs]
            m=sum(v)/len(v); s=(sum((x-m)**2 for x in v)/len(v))**0.5
        else:
            d=json.load(open(ROOT/f'experiments/review_tool_access/runs/{c}/{sp}/summary.json'))
            m=d['overall']['accuracy']; s=0
        vals.append(m); errs.append(s)
    ax.bar([i+j*w for i in range(len(conds))],vals,yerr=errs,capsize=2,width=w,label=sp)
ax.set_xticks([i+1.5*w for i in range(len(conds))]); ax.set_xticklabels(labels); ax.set_ylabel('accuracy'); ax.set_ylim(0,1); ax.legend(); save(fig,'fig_tool_access')

# optimizer cost-vs-accuracy curves (E9 curves.csv; zero accepted edits)
cur=ROOT/'experiments/final/E9_costs/curves.csv'
fig,ax=plt.subplots(figsize=(6,3.4))
if cur.exists():
    cr=list(csv.DictReader(open(cur)))
    seen=set()
    for r in cr:
        label=f"{r['optimizer']} ({r['start'].replace('baseline_','')})"
        x=[float(c['cumulative_tokens'])/1e6 for c in cr if c['optimizer']==r['optimizer'] and c['start']==r['start']]
        y=[float(c['best_val']) for c in cr if c['optimizer']==r['optimizer'] and c['start']==r['start']]
        if label in seen or not x: continue
        seen.add(label); ax.plot(x,y,marker='o',ms=3,label=label)
ax.set_xlabel('cumulative optimizer tokens (M)'); ax.set_ylabel('best validation accuracy')
ax.grid(alpha=.3); ax.legend(fontsize=7); save(fig,'fig_optimizer_curves')

# rule ablation large
d=json.load(open(ROOT/'experiments/qwen3_4b_ablation_full/ablation/ablation_summary.json'))
rows=d['rows'][1:]; fig,ax=plt.subplots(figsize=(7,3.2))
ax.bar([r['rule_index'] for r in rows],[-r.get('delta',0) for r in rows])
ax.set_xlabel('removed rule index'); ax.set_ylabel('accuracy drop'); ax.grid(axis='y',alpha=.3); save(fig,'fig_rule_ablation')

# real benchmark per-task overall bars
fig,ax=plt.subplots(figsize=(5.5,3.2))
vals=[]; names=[]
for n,p in [('GA easy','experiments/qwen3_4b_real_bench/bench/ga_easy/summary.json'),('GA hard','experiments/qwen3_4b_real_bench/bench/ga_hard/summary.json'),('GraphQA','experiments/qwen3_4b_real_bench/bench/graphqa_test/summary.json')]:
    d=json.load(open(ROOT/p)); names.append(n); vals.append(d['overall']['accuracy'])
ax.bar(names,vals); ax.set_ylim(0,1); ax.set_ylabel('accuracy'); save(fig,'fig_real_bench')

# proxy: same 15 candidates, val vs test_large
rows=list(csv.DictReader(open(ROOT/'experiments/final/E5_proxy/labels_same_candidates.csv')))
fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),sharex=True)
for ax,target,label in [(axes[0],'val_accuracy','val'),(axes[1],'large_accuracy','test_large')]:
    ax.scatter([float(r['proxy']) for r in rows],[float(r[target]) for r in rows],s=18)
    ax.set_xlabel('learned static proxy (LOO)'); ax.set_ylabel(f'{label} accuracy')
    ax.grid(alpha=.3)
save(fig,'fig_proxy')

# router
rows=[]
for m in ['global','routed','decomposed']:
    text=open(ROOT/f'experiments/review_router/{m}/summary.txt').read(); line=[x for x in text.splitlines() if 'disconnected ALL' in x][0]
    rows.append((m,float(line.split()[-1].strip('%'))/100))
fig,ax=plt.subplots(figsize=(4.5,3.2)); ax.bar([x[0] for x in rows],[x[1] for x in rows]); ax.set_ylim(0,.6); ax.set_ylabel('accuracy'); save(fig,'fig_router')

# language/library available cells
fig,ax=plt.subplots(figsize=(5,3.2))
ax.bar(['Python+NetworkX','C/gcc no library'],[.96,.12]); ax.set_ylim(0,1); ax.set_ylabel('accuracy'); save(fig,'fig_lang_lib')
print('figures written')

# characteristics: family and density bars from the E8 CSV
rows=list(csv.DictReader(open(ROOT/'experiments/final/E8_characteristics/characteristics.csv')))
fig,ax=plt.subplots(figsize=(7,3.4))
fams=sorted({r['value'] for r in rows if r['section']=='by_family'})
for j,h in enumerate(['Baseline A','Baseline B','Baseline B v2']):
    vals=[]
    for f in fams:
        x=[r for r in rows if r['section']=='by_family' and r['run']==h and r['value']==f]
        vals.append(float(x[0]['accuracy']) if x else 0)
    ax.bar([i+j*.25 for i in range(len(fams))],vals,width=.25,label=h)
ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams,rotation=20); ax.set_ylabel('accuracy'); ax.legend(); save(fig,'fig_characteristics')
fig,ax=plt.subplots(figsize=(6,3.2))
for h in ['Baseline A','Baseline B','Baseline B v2']:
    vals=[float(r['accuracy']) for r in rows if r['section']=='by_density' and r['run']==h]
    labels=[r['value'] for r in rows if r['section']=='by_density' and r['run']==h]
    ax.plot(range(len(vals)),vals,marker='o',label=h)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,rotation=20,fontsize=7); ax.set_ylabel('accuracy'); ax.legend(); save(fig,'fig_density')
print('extra figures written')
