# Final results index (feedback → result)

Updated 2026-11-21. Status: **DONE**, **PARTIAL**, **PENDING**, **NOT RUN**.

| Feedback | Experiment | Status | Result path | Report location |
|---|---|---|---|---|
| F1 3-page minimum | E13/E15 | **PARTIAL** | `experiments/final/E13_lint.md`; `report/AI_DRAFTS.md` | AI-DRAFT section starts p.3; Sections 1-5 still end on p.2, so the authors must write Intro/Related Work (see E13_lint.md) |
| F2 tool access | E0/E2/E10/E11 | E2 **DONE**, E11 **PENDING**, E10 **NOT RUN** | `E0_smoke_cpu/`, `E2_tool_access/summary.csv` | `tab_tool_access`, `fig_tool_access`, "Tool access" paragraph |
| F3 characteristics | E8 | **DONE** (CPU) | `E8_characteristics/{family_x_size,regression,characteristics}.csv` | `tab_characteristics`, `fig_characteristics`, `fig_density`, "Characteristics" paragraph |
| F4 GEPA | E3 | **DONE** (7B critic: 0 accepted) | `E3_optimizers/gepa_baseline_b/` | `tab_optimizers`, final section |
| F5 SkillOpt / EvoSkill | E3/E4 | **DONE**: SkillOpt 1 accepted; EvoSkill 0 accepted (val already 100%) | `E3_optimizers/skillopt_baseline_b/`, `E4_evoskill/baseline_b/` | `tab_optimizers`, `tab_optimizer_outputs`, final section |
| F6 optimization cost | E9 | **DONE** (CPU) | `E9_costs/{per_run,curves,counterfactual}.csv` | `tab_costs`, `tab_cost_counterfactual`, `fig_optimizer_curves` |
| F7 large-graph proxies | E5 | **PARTIAL** (reduced: 17/15 candidates, no evolved rubric) | `E5_proxy/{summary.csv,labels_same_candidates.csv}` | `tab_proxy`, `fig_proxy`, "Proxies" paragraph |
| F8 router/decomposition | E7 | **PARTIAL** (existing 8-problem pilot only; no learned router) | `experiments/review_router/` | `tab_router`, `fig_router`, "Router" paragraph |
| F9 RRSI overfitting | E3 | **PARTIAL** (two 7B-critic runs accepted 1 edit each; overfitting probe not run) | `E3_optimizers/rrsi_baseline_b{,_run1}/` | `tab_optimizers`, final section |
| F10 citation spacing | E13 | **DONE** (0 violations) | `E13_lint.md` | – |
| held-out v2 | E1 | **DONE** | `E1_heldout/{summary,summary_by_task,mcnemar}.csv` | `tab_heldout_small`, "Held-out" paragraph |
| repeats/error bars | E2 | **DONE** | `E2_repeats/{summary,gen_gap,mcnemar}.csv` | `tab_main_scaling`, `fig_scaling`, `fig_gen_gap` |
| larger real bench | E6 | **DONE** | `E6_real_bench/{summary,per_task}.csv` | `tab_real_bench_overall`, `tab_real_bench_per_task`, `fig_real_bench` |
| C + library 2×2 | E10 | **NOT RUN** | – | `tab_lang_lib` has only the existing C-no-library pilot |
| 32k no-tools | E11 | **DONE** | `E11_notools_longctx/{summary.csv,fit_rows.csv}` | `tab_notools_longctx`, `fig_notools_longctx`, "No-tools at 32k" |
| cross-model | E14 | **NOT RUN** (optional) | – | – |

## Headline numbers now in the report

- Held-out `test_small_fresh` (150, 3 repeats): A 88.4±1.3, B 93.8±0.8,
  v2 95.8±0.8. v2's contaminated `test_small` value was 94.0.
- Scaling (3 repeats): v2 leads on small, B leads on large (90.7±3.7 vs
  85.3±0.5); stress is 74.5±8.4 (B) vs 73.3±0.9 (v2).
- Tool access (3 repeats): v2 96.4±0.3 vs 35.8±0.6 on small; 0.0% without
  tools on the 25-problem large subset.
- Optimizers: with the 7B critic, SkillOpt and RRSI each accepted one edit;
  GEPA and Meta-Harness accepted none; EvoSkill accepted none (val 100%).
  Changed SkillOpt harness: 98.7/94.0/94.0/76.0% on fresh/medium/large/stress;
  changed RRSI: 96.0/96.0/94.0/74.0% (one run each).
- Real benchmarks (100/task, GraphQA 50/task): B is best on GraphArena
  (75.6 easy, 62.6 hard), v2 ties B on GraphQA (89.2 vs 89.0), A is below both.
- No-tools at 32k: 18.0/16.7% on medium, 11.0/10.0% on large; tools are 84-92%.
- Proxy transfer on the same 15 candidates: Spearman −0.25 (val) and
  +0.29 (`test_large`), both CIs include zero.

## Not run, and why

- E10 (language × library) needs new C/Python sandboxes; no GPU window.
- E7 learned router needs 5 library-harness val runs plus new 120-problem data;
  no GPU window.
- E14 optional cross-model check; no larger actor on disk.
- Full E5 (60 candidates, evolved rubric, `proxy_large` split): no GPU window.
