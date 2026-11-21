# Final results index (feedback → result)

Updated 2026-11-21. Status: **DONE**, **PARTIAL**, **PENDING**, **NOT RUN**.

| Feedback | Experiment | Status | Result path | Report location |
|---|---|---|---|---|
| F1 3-page minimum | E13/E15 | **PARTIAL** | `experiments/final/E13_lint.md`; `report/AI_DRAFTS.md` | AI-DRAFT section starts p.3; Sections 1-5 still end on p.2, so the authors must write Intro/Related Work (see E13_lint.md) |
| F2 tool access | E0/E2/E10/E11 | E2 **DONE**, E11 **PENDING**, E10 **NOT RUN** | `E0_smoke_cpu/`, `E2_tool_access/summary.csv` | `tab_tool_access`, `fig_tool_access`, "Tool access" paragraph |
| F3 characteristics | E8 | **DONE** (CPU) | `E8_characteristics/{family_x_size,regression,characteristics}.csv` | `tab_characteristics`, `fig_characteristics`, `fig_density`, "Characteristics" paragraph |
| F4 GEPA | E3 | **PENDING** (running in E3-lite) | `E3_optimizers/gepa_baseline_b/` | `tab_optimizers`, final section |
| F5 SkillOpt / EvoSkill | E3/E4 | **PENDING** (running in E3-lite) | `E3_optimizers/skillopt_baseline_b/`, `E4_evoskill/baseline_b/` | `tab_optimizers`, final section |
| F6 optimization cost | E9 | **DONE** (CPU) | `E9_costs/{per_run,curves,counterfactual}.csv` | `tab_costs`, `tab_cost_counterfactual`, `fig_optimizer_curves` |
| F7 large-graph proxies | E5 | **PARTIAL** (reduced: 17/15 candidates, no evolved rubric) | `E5_proxy/{summary.csv,labels_same_candidates.csv}` | `tab_proxy`, `fig_proxy`, "Proxies" paragraph |
| F8 router/decomposition | E7 | **PARTIAL** (existing 8-problem pilot only; no learned router) | `experiments/review_router/` | `tab_router`, `fig_router`, "Router" paragraph |
| F9 RRSI overfitting | E3 | **PARTIAL** (RRSI pilot done; strong-critic probe not run) | `review_optimizers2/`, `E3_optimizers/rrsi_baseline_b/` | `tab_optimizers` |
| F10 citation spacing | E13 | **DONE** (0 violations) | `E13_lint.md` | – |
| held-out v2 | E1 | **DONE** | `E1_heldout/{summary,summary_by_task,mcnemar}.csv` | `tab_heldout_small`, "Held-out" paragraph |
| repeats/error bars | E2 | **DONE** | `E2_repeats/{summary,gen_gap,mcnemar}.csv` | `tab_main_scaling`, `fig_scaling`, `fig_gen_gap` |
| larger real bench | E6 | **RUNNING** | `E6_real_bench/summary.csv` (written when the job finishes) | `tab_real_bench_overall`, `tab_real_bench_per_task`, `fig_real_bench` |
| C + library 2×2 | E10 | **NOT RUN** | – | `tab_lang_lib` has only the existing C-no-library pilot |
| 32k no-tools | E11 | **PENDING** (job 1547590) | `E11_notools_longctx/` | `tab_tool_access` note + new paragraph when done |
| cross-model | E14 | **NOT RUN** (optional) | – | – |

## Headline numbers now in the report

- Held-out `test_small_fresh` (150, 3 repeats): A 88.4±1.3, B 93.8±0.8,
  v2 95.8±0.8. v2's contaminated `test_small` value was 94.0.
- Scaling (3 repeats): v2 leads on small, B leads on large (90.7±3.7 vs
  85.3±0.5); stress is 74.5±8.4 (B) vs 73.3±0.9 (v2).
- Tool access (3 repeats): v2 96.4±0.3 vs 35.8±0.6 on small; 0.0% without
  tools on the 25-problem large subset.
- Optimizer pilots: zero accepted edits with a 4B critic; strong-critic grid
  is job 1547527 (SkillOpt/GEPA/RRSI/Meta/EvoSkill from B).
- Proxy transfer on the same 15 candidates: Spearman −0.25 (val) and
  +0.29 (`test_large`), both CIs include zero.

## Not run, and why

- E10 (language × library) needs new C/Python sandboxes; no GPU window.
- E7 learned router needs 5 library-harness val runs plus new 120-problem data;
  no GPU window.
- E14 optional cross-model check; no larger actor on disk.
- Full E5 (60 candidates, evolved rubric, `proxy_large` split): no GPU window.
