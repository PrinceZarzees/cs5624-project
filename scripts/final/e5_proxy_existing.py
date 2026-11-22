#!/usr/bin/env python3
"""Reduced E5: proxy ranking against existing test_large labels.

Candidates: A, B, v2, and the 14 one-rule-removed v2 variants. Labels are the
already-measured test_large accuracies from the frozen scaling/ablation runs.
No new model calls.
"""
import csv, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import numpy as np
from graphharness.agent import BASELINE_B_V2, get_harness
from graphharness.rubric_proxy import spearman, pairwise_accuracy, static_features

out=Path('experiments/final/E5_proxy'); out.mkdir(parents=True, exist_ok=True)
cands=[]
for h in ['baseline_a','baseline_b','baseline_b_v2']:
    p=Path(f'experiments/qwen3_4b_scaling_full/scaling/{h}/test_large/summary.json')
    d=json.load(open(p)); cands.append({'id':h,'harness':get_harness(h),'large_accuracy':d['overall']['accuracy']})
ab=json.load(open('experiments/qwen3_4b_ablation_full/ablation/ablation_summary.json'))
for row in ab['rows']:
    if row['rule_index'] is None: continue
    h=BASELINE_B_V2.without_rule(row['rule_index'])
    cands.append({'id':row['variant'],'harness':h,'large_accuracy':row['accuracy'],'rule_index':row['rule_index']})
# features
keywords=['verify','check','print','networkx','nx.','diameter','clique','component','timeout','node id','answer','exact','heuristic']
for c in cands:
    h=c['harness']; text=(h.system_prompt+'\n'+'\n'.join(h.rules)).lower()
    c['features']=[len(h.system_message()),len(h.rules),sum(k in text for k in keywords),
                   int('print(' in text),int('diameter' in text),int('clique' in text),
                   int('component' in text),int('verify' in text or 'check' in text),
                   int(h.allow_tools),int(h.graph_view=='edge_list'),
                   int(bool(re.search(r'\bnode\s+\d+\b',text)))]
X=np.array([c['features'] for c in cands],float); y=np.array([c['large_accuracy'] for c in cands])
mu=X.mean(0); sd=X.std(0); sd[sd==0]=1; Xs=(X-mu)/sd
pred=[]
for i in range(len(y)):
    mask=np.arange(len(y))!=i
    w=np.linalg.solve(Xs[mask].T@Xs[mask]+np.eye(Xs.shape[1]), Xs[mask].T@y[mask])
    pred.append(float(Xs[i]@w))
for c,p in zip(cands,pred): c['learned_proxy']=p
ranked=sorted(cands,key=lambda c:c['learned_proxy'],reverse=True)
best=max(cands,key=lambda c:c['large_accuracy'])
with open(out/'labels.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['id','large_accuracy','learned_proxy','rule_index']); w.writeheader()
    for c in cands: w.writerow({'id':c['id'],'large_accuracy':c['large_accuracy'],'learned_proxy':c['learned_proxy'],'rule_index':c.get('rule_index','')})
summary={'n':len(cands),'spearman':spearman([c['learned_proxy'] for c in cands],y.tolist()),
         'pairwise':pairwise_accuracy([c['learned_proxy'] for c in cands],y.tolist()),
         'top3_recall':best['id'] in [c['id'] for c in ranked[:3]],
         'best_actual':best['id'],'top3':[c['id'] for c in ranked[:3]]}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
