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

# real benchmarks: expanded E6 if present, else the frozen 20/task run
e6=ROOT/'experiments/final/E6_real_bench/summary.csv'
if e6.exists():
    rs=list(csv.DictReader(open(e6)))
    rows=[]
    for r in rs:
        rows.append([r['benchmark'],r['harness'].replace('baseline_',''),r['n'],
                     f"{float(r['accuracy']):.1%}",
                     f"[{float(r['wilson_lo']):.1%}, {float(r['wilson_hi']):.1%}]",
                     f"{float(r['parsed']):.1%}"])
    header=['benchmark','harness','n','accuracy','95% CI','parsed']
    csvw('tab_real_bench_overall',header,rows); tex('tab_real_bench_overall',header,rows,align='llrrcr')

    pt=list(csv.DictReader(open(ROOT/'experiments/final/E6_real_bench/per_task.csv')))
    keys=sorted({(r['benchmark'],r['task']) for r in pt})
    rows=[]
    for bench,task in keys:
        cells=[]
        for h in ['baseline_a','baseline_b','baseline_b_v2']:
            x=[r for r in pt if r['benchmark']==bench and r['task']==task and r['harness']==h]
            cells.append(f"{float(x[0]['accuracy']):.1%}" if x else '--')
        n=[r for r in pt if r['benchmark']==bench and r['task']==task][0]['n']
        rows.append([bench.replace('GraphArena ','GA ').replace(' test',''),task,n]+cells)
    header=['benchmark','task','n','A','B','B v2']
    csvw('tab_real_bench_per_task',header,rows); tex('tab_real_bench_per_task',header,rows,align='llrrrr')

    # E11 no-tools long context
    e11=ROOT/'experiments/final/E11_notools_longctx/summary.csv'
    if e11.exists():
        rows=[]
        for r in csv.DictReader(open(e11)):
            fmt=lambda k: f"{float(r[k]):.1%}" if r.get(k) not in ('',None) else '--'
            rows.append([r['harness'].replace('_text','').replace('baseline_',''),r['split'],r['n'],
                         fmt('accuracy'),fmt('parsed'),fmt('feasible'),
                         f"{r['n_le8k']}"+(f" ({fmt('acc_le8k')})" if r.get('acc_le8k') else ''),
                         f"{r['n_8k_to_32k']}",r['n_gt32k'],r['max_prompt_tokens']])
        header=['harness','split','n','acc 32k','parsed','feasible','<=8k','8k-32k','>32k','max tok']
        csvw('tab_notools_longctx',header,rows); tex('tab_notools_longctx',header,rows,align='llrrrrrrrrr')
else:
    rows=[]
    for name,path in [('GraphArena easy','experiments/qwen3_4b_real_bench/bench/ga_easy/summary.json'),
                      ('GraphArena hard','experiments/qwen3_4b_real_bench/bench/ga_hard/summary.json'),
                      ('GraphQA test','experiments/qwen3_4b_real_bench/bench/graphqa_test/summary.json')]:
        d=json.load(open(ROOT/path)); o=d['overall']; rows.append([name,d['n'],f"{o['accuracy']:.1%}",f"{o['parsed']:.1%}",f"{o['feasible']:.1%}"])
    header=['benchmark','n','accuracy','parsed','feasible']
    csvw('tab_real_bench_overall',header,rows); tex('tab_real_bench_overall',header,rows)
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

# E7 120-problem router modes
e7=ROOT/'experiments/final/E7_router/summary.csv'
if e7.exists():
    rs=list(csv.DictReader(open(e7)))
    all_modes=['global_v2','rule_router','learned_router','decomposed_rule','decomposed_learned']
    modes=[m for m in all_modes if any(r['mode']==m for r in rs)]
    tasks=['ALL','connected_components','max_clique','max_independent_set','min_vertex_cover']
    rows=[]
    for m in modes:
        cells=[]
        for t in tasks:
            x=[r for r in rs if r['set']=='disconnected120' and r['mode']==m and r['task']==t]
            cells.append(f"{float(x[0]['accuracy']):.1%}" if x else '--')
        rows.append([m.replace('_',' ')]+cells)
    header=['mode','ALL','components','max clique','MIS','MVC']
    csvw('tab_router120',header,rows); tex('tab_router120',header,rows,align='lrrrrr')

# router (8-problem pilot)
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

# E3c overfitting probe
probe=ROOT/'experiments/final/E3_probe/summary.csv'
if probe.exists():
    rows=[]
    for r in csv.DictReader(open(probe)):
        rows.append([r['run'].replace('_probe',''),r['checkpoint'],
                     f"{float(r['val_accuracy']):.1%}",f"{float(r['large_accuracy']):.1%}",
                     f"{float(r['delta_gen'])*100:+.1f}"])
    header=['run','checkpoint','val','test_large','small-large (pp)']
    csvw('tab_overfit',header,rows); tex('tab_overfit',header,rows,align='llrrr')

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
# optimizer table: E3/E4 summary when available, otherwise cost-only
e3s=ROOT/'experiments/final/E3_optimizers/summary.csv'
rows=[]
if e3s.exists():
    for r in csv.DictReader(open(e3s)):
        init=f"{float(r['init_val']):.1%}" if r['init_val'] else '--'
        best=f"{float(r['best_val']):.1%}" if r['best_val'] else '--'
        acc=f"{r['accepted']}/{r['rounds']}"
        rows.append([r['optimizer'],r['start'].replace('baseline_',''),
                     acc,f"{init} $\\rightarrow$ {best}",
                     f"{float(r['total_tokens'])/1e6:.2f}",f"{float(r['est_usd']):.2f}",
                     r['parsed_proposals']])
    header=['optimizer','start','accepted/rounds','init $\\rightarrow$ best val','tokens (M)','USD','parsed']
    csvw('tab_optimizers',header,rows); tex('tab_optimizers',header,rows,align='llrcrrr')

    # changed-harness evaluation on the four scaling splits
    out_rows=[]; b=[r for r in csv.DictReader(open(ROOT/'experiments/final/E2_repeats/summary.csv'))
                    if r['harness']=='baseline_b']
    getb={r['split']:r for r in b}
    if getb:
        out_rows.append(['Baseline B (3 repeats)',
                         f"{float(getb['test_small']['mean']):.1%}",
                         f"{float(getb['test_medium']['mean']):.1%}",
                         f"{float(getb['test_large']['mean']):.1%}",
                         f"{float(getb['test_stress']['mean']):.1%}",
                         f"{(float(getb['test_small']['mean'])-float(getb['test_large']['mean']))*100:+.1f}"])
    for r in csv.DictReader(open(e3s)):
        if not r['small_fresh']:
            continue
        name=f"{r['optimizer']} output"
        out_rows.append([name,
                         f"{float(r['small_fresh']):.1%}", f"{float(r['medium']):.1%}",
                         f"{float(r['large']):.1%}", f"{float(r['stress']):.1%}",
                         f"{float(r['gen_gap'])*100:+.1f}"])
    if out_rows:
        header=['harness','fresh small','medium','large','stress','small-large (pp)']
        csvw('tab_optimizer_outputs',header,out_rows)
        tex('tab_optimizer_outputs',header,out_rows,align='lrrrrr')
else:
    for r in csv.DictReader(open(ROOT/'experiments/final/E9_costs/per_run.csv')):
        rows.append([r['optimizer'],r['start'],r['initial_val'],r['best_val'],
                     f"{float(r['tokens'])/1e6:.2f}",f"USD {float(r['usd_tokens']):.2f}"])
    header=['optimizer','start','initial val','best val','tokens (M)','USD tokens']
    csvw('tab_optimizers',header,rows); tex('tab_optimizers',header,rows,align='llrrrr')
print('tables written')
