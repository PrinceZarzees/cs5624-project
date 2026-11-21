#!/usr/bin/env python3
"""Regenerate report/tables/tab_*.{csv,tex} from stored summaries."""
import csv, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
T=ROOT/'report/tables'; T.mkdir(exist_ok=True)

def esc(x):
    s=str(x)
    for ch,rep in [('%','\\%'),('&','\\&'),('_','\\_'),('#','\\#')]:
        s=s.replace(ch,rep)
    return s

def tex(name, header, rows, align=None):
    align=align or ('l'+'r'*(len(header)-1))
    lines=[f'\\begin{{tabular}}{{{align}}}','\\toprule',' & '.join(esc(h) for h in header)+r' \\','\\midrule']
    for r in rows: lines.append(' & '.join(esc(x) for x in r)+r' \\')
    lines += ['\\bottomrule','\\end{tabular}']
    (T/f'{name}.tex').write_text('\n'.join(lines)+'\n')

def csvw(name, header, rows):
    with open(T/f'{name}.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

# main scaling: mean +/- std over repeats (E2) with n per cell
s=json.load(open(ROOT/'experiments/qwen3_4b_scaling_full/scaling/scaling_summary.json'))
e2=ROOT/'experiments/final/E2_repeats/summary.csv'
splits=['test_small','test_medium','test_large','test_stress']
if e2.exists():
    agg=list(csv.DictReader(open(e2)))
    rows=[]
    for h in ['baseline_a','baseline_b','baseline_b_v2']:
        cells=[]
        for sp in splits:
            r=[x for x in agg if x['harness']==h and x['split']==sp]
            cells.append(f"{float(r[0]['mean']):.1%} $\\pm$ {float(r[0]['std'])*100:.1f}" if r else '--')
        n={r['split']:r['n'] for r in agg if r['harness']==h}
        rows.append([h]+cells+[n.get(splits[0],'')])
    header=['Harness','small','medium','large','stress','n']
    csvw('tab_main_scaling',header,rows)
    tex('tab_main_scaling',header,rows,align='lrrrrr')
    (ROOT/'experiments/final/E2_repeats/NOTES.md').write_text(
        '# E2 repeats\n\n'
        '- ' + ', '.join(f"{r['harness']}/{r['split']}: {float(r['mean']):.1%}±{float(r['std'])*100:.1f} (k={r['repeats']})"
                          for r in agg) + '\n'
        '- std is the population std over repeats; see `mcnemar.csv` for paired tests.\n'
        '- Cells flagged for >5% agent no-prediction errors: ' +
        ', '.join(f"{r['harness']}/{r['split']} rep{r['flagged_runs']}" for r in agg if r['flagged_runs']) + '\n')
else:
    rows=[]
    for h in ['baseline_a','baseline_b','baseline_b_v2']:
        d={r['split']:r for r in s['runs'] if r['harness']==h}
        rows.append([h]+[f"{d[x]['accuracy']:.1%}" for x in splits])
    header=['Harness','small','medium','large','stress']
    csvw('tab_main_scaling',header,rows); tex('tab_main_scaling',header,rows)

# tool access: mean +/- std over 3 repeats (rep 1 is the frozen review run)
ta=ROOT/'experiments/final/E2_tool_access/summary.csv'
rows=[]
if ta.exists():
    runs=list(csv.DictReader(open(ta)))
    for c in ['baseline_a_tools','baseline_a_text','baseline_b_v2_tools','baseline_b_v2_text']:
        cells=[]; ns=set()
        for sp in ['test_small','test_medium','test_large']:
            rs=[r for r in runs if r['condition']==c and r['split']==sp]
            vals=[float(r['accuracy']) for r in rs]
            if vals:
                import statistics
                cells.append(f"{statistics.mean(vals):.1%} $\\pm$ {statistics.pstdev(vals)*100:.1f}")
                ns.add(int(rs[0]['n']))
            else:
                cells.append('--')
        rows.append([c.replace('baseline_','')]+cells)
    header=['condition','small (n=150)','medium (n=150)','large (n=25)']
    csvw('tab_tool_access',header,rows); tex('tab_tool_access',header,rows,align='lrrr')
else:
    for c in ['baseline_a_tools','baseline_a_text','baseline_b_v2_tools','baseline_b_v2_text']:
        vals=[]
        for sp in ['test_small','test_medium','test_large']:
            d=json.load(open(ROOT/f'experiments/review_tool_access/runs/{c}/{sp}/summary.json'))
            vals.append(f"{d['overall']['accuracy']:.1%}")
        rows.append([c]+vals)
    header=['Condition','small','medium','large']
    csvw('tab_tool_access',header,rows); tex('tab_tool_access',header,rows)

# ablation large/stress
for name,path in [('tab_rule_ablation','experiments/qwen3_4b_ablation_full/ablation/ablation_summary.json'),
                  ('tab_rule_ablation_stress','experiments/qwen3_4b_ablation_stress/ablation/ablation_summary.json')]:
    d=json.load(open(ROOT/path)); rows=[]
    for r in d['rows'][1:]:
        rows.append([r['rule_index'],f"{r['accuracy']:.1%}",f"{r.get('delta',0):+.1%}",r['rule'][:70]])
    header=['rule','accuracy','delta','text']
    csvw(name,header,rows); tex(name,header,rows,align='rrrp{7cm}')

# costs (per optimizer run) + large-graph counterfactual
rows=[]
for r in csv.DictReader(open(ROOT/'experiments/final/E9_costs/per_run.csv')):
    rows.append([r['optimizer'],r['start'],f"{float(r['tokens'])/1e6:.2f}",
                 f"{float(r['wall_s']):.0f}",f"{float(r['gpu_hours']):.2f}",
                 f"USD {float(r['usd_tokens']):.2f}",f"USD {float(r['usd_gpu_hours']):.2f}"])
header=['optimizer','start','tokens (M)','wall s','GPU-h','USD tokens','USD GPU-h']
csvw('tab_costs',header,rows); tex('tab_costs',header,rows,align='llrrrrr')

cf=list(csv.DictReader(open(ROOT/'experiments/final/E9_costs/counterfactual.csv')))
rows=[[r['scenario'].replace('optimize on ',''),f"{float(r['tokens_per_problem']):.0f}",
       f"{float(r['seconds_per_problem']):.1f}",f"{int(r['est_tokens'])/1e6:.2f}",
       f"{int(r['est_wall_s'])/3600:.2f}",r['ratio_vs_actual_small_run']] for r in cf]
header=['training graphs','tok/prob','s/prob','10-round tokens (M)','GPU-h','ratio vs measured run']
csvw('tab_cost_counterfactual',header,rows); tex('tab_cost_counterfactual',header,rows,align='lrrrrr')

# proxy E5: same-candidate transfer plus earlier pilots
rows=[]
for r in csv.DictReader(open(ROOT/'experiments/final/E5_proxy/summary.csv')):
    def f3(k):
        return f"{float(r[k]):.2f}" if r.get(k) not in ('', None) else '--'
    ci=f"[{f3('rho_lo')}, {f3('rho_hi')}]" if r.get('rho_lo') else '--'
    name=r['proxy'].replace('learned_static_LOO','static-LOO').replace('cheap_rollout_k1','rollout-k1')
    rows.append([name,r['target'],r['n'],f3('spearman'),ci,
                 f"{float(r['pairwise']):.1%}" if r.get('pairwise') else '--',
                 r.get('top3_hit') or '--',r['source']])
header=['proxy','target','n','Spearman','95% CI','pairwise','top-3','pool']
csvw('tab_proxy',header,rows); tex('tab_proxy',header,rows,align='llrlrrlp{2.2cm}')

# real benchmarks overall
rows=[]
for name,path in [('GraphArena easy','experiments/qwen3_4b_real_bench/bench/ga_easy/summary.json'),
                  ('GraphArena hard','experiments/qwen3_4b_real_bench/bench/ga_hard/summary.json'),
                  ('GraphQA test','experiments/qwen3_4b_real_bench/bench/graphqa_test/summary.json')]:
    d=json.load(open(ROOT/path)); o=d['overall']; rows.append([name,d['n'],f"{o['accuracy']:.1%}",f"{o['parsed']:.1%}",f"{o['feasible']:.1%}"])
header=['benchmark','n','accuracy','parsed','feasible']
csvw('tab_real_bench_overall',header,rows); tex('tab_real_bench_overall',header,rows)

# real per task
rows=list(csv.reader(open(ROOT/'report/tables/real_benchmarks.csv')))
csvw('tab_real_bench_per_task',rows[0],rows[1:]); tex('tab_real_bench_per_task',rows[0],rows[1:],align='llrrrr')

# characteristics: family rows for all harnesses
rows=[]
for r in csv.DictReader(open(ROOT/'report/tables/characteristics.csv')):
    if r['section']=='by_family': rows.append([r['run'],r['value'],r['n'],f"{float(r['accuracy']):.1%}"])
header=['harness','family','n','accuracy']
csvw('tab_characteristics',header,rows); tex('tab_characteristics',header,rows)

# lang/lib proxy: only Python/NetworkX and C data available
rows=[['Python+NetworkX','yes','test_small','96.0%'],['Python, no NetworkX','no','test_small','not run'],
      ['C/gcc, no library','no','test_small','12.0%'],['C/gcc + library','yes','test_small','not run']]
header=['language','graph library','split','accuracy']
csvw('tab_lang_lib',header,rows); tex('tab_lang_lib',header,rows)

# router
rows=[]
for m,p in [('global','experiments/review_router/global/summary.txt'),('routed','experiments/review_router/routed/summary.txt'),('decomposed','experiments/review_router/decomposed/summary.txt')]:
    text=open(ROOT/p).read(); line=[x for x in text.splitlines() if 'disconnected ALL' in x][0]
    acc=line.split()[-1]; rows.append([m,acc])
header=['mode','accuracy']
csvw('tab_router',header,rows); tex('tab_router',header,rows)

# cross model
rows=list(csv.reader(open(ROOT/'report/tables/cross_model.csv')))
body=[]
for r in rows[1:]:
    r=list(r)
    try:
        r[-1]=f"{float(r[-1]):.1%}"
    except ValueError:
        pass
    body.append(r)
csvw('tab_cross_model',rows[0],body); tex('tab_cross_model',rows[0],body,align='lllllrr')

# E1 held-out small split: mean +/- std over 3 repeats + 95% Wilson CI
if (ROOT/'experiments/final/E1_heldout/summary.csv').exists():
    rows=[]; logs=[]
    for r in csv.DictReader(open(ROOT/'experiments/final/E1_heldout/summary.csv')):
        rows.append([r['harness'],r['repeats'],f"{float(r['mean']):.1%}",
                     f"{float(r['std'])*100:.1f}",
                     f"[{float(r['wilson_lo']):.1%}, {float(r['wilson_hi']):.1%}]"])
    header=['harness','repeats','accuracy','std (pp)','95% Wilson CI']
    csvw('tab_heldout_small',header,rows); tex('tab_heldout_small',header,rows,align='lrrcr')
    mc=ROOT/'experiments/final/E1_heldout/mcnemar.csv'
    if mc.exists():
        for r in csv.DictReader(open(mc)):
            logs.append(f"McNemar B vs v2 repeat {r['repeat']}: B-only {r['a_only']}, v2-only {r['b_only']}, p={float(r['p_value']):.3f}")
        (ROOT/'experiments/final/E1_heldout/NOTES.md').write_text(
            '# E1 fresh held-out split\n\n'
            '- Split: `test_small_fresh`, seed 56240, 30/task x 5 tasks = 150, |V| 4-50, 0 overlap with the six frozen splits.\n'
            '- 3 repeats per harness, Qwen3-4B BF16, 8k context, temperature 0.\n'
            '- ' + '\n- '.join(logs) + '\n')
else:
    print('E1 heldout not available yet; not writing tab_heldout_small')
# optimizer table from per_run
rows=[]
for r in csv.DictReader(open(ROOT/'experiments/final/E9_costs/per_run.csv')):
    rows.append([r['optimizer'],r['start'],r['initial_val'],r['best_val'],
                 f"{float(r['tokens'])/1e6:.2f}",f"USD {float(r['usd_tokens']):.2f}"])
header=['optimizer','start','initial val','best val','tokens (M)','USD tokens']
csvw('tab_optimizers',header,rows); tex('tab_optimizers',header,rows,align='llrrrr')
print('tables written')
