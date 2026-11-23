# Report edit log

All changes are on `saminrk-final-experiments`.

| Date | File | Change | Reason | Source |
|---|---|---|---|---|
| 2026-11-21 | report/main.tex | Added final-results section with figures/tables inside AI-DRAFT markers | integrate final results | experiments/final |
| 2026-11-21 | report/README.md | added link to EXPERIMENTAL_REPORT.md | report index | report/EXPERIMENTAL_REPORT.md |

No existing human-written prose was deleted or reworded.

## 2026-11-21 E1/E2 integration (inside the existing AI-DRAFT block)

| Location | Change | Source |
|---|---|---|
| main.tex "Scaling and ablation" | Replaced "The v2 harness is best on the 2,000-node stress split" with the 3-repeat mean ± std numbers; added McNemar outcomes and Δ_gen | `experiments/final/E2_repeats/summary.csv`, `mcnemar.csv`, `gen_gap.csv` |
| main.tex new "Held-out small-graph estimate" | Added paragraph + `tab_heldout_small` figure-free table | `experiments/final/E1_heldout/summary.csv`, `summary_by_task.csv`, `mcnemar.csv` |

Both edits are inside the `% >>> AI-DRAFT` block (lines ~103+).

## 2026-11-21 E8/E9 integration (inside the existing AI-DRAFT block)

| Location | Change | Source |
|---|---|---|
| main.tex new "Characteristics" | Added paragraph, `tab_characteristics`, `fig_characteristics`, `fig_density`, and the logistic-regression odds ratios | `experiments/final/E8_characteristics/{family_x_size.csv,regression.csv}` |
| main.tex "Optimization and proxies" | Added per-problem cost and the 10-round large-graph counterfactual (1.94x) + `tab_cost_counterfactual` | `experiments/final/E9_costs/{counterfactual.csv,curves.csv}` |

## 2026-11-21 E5 integration (inside the existing AI-DRAFT block)

| Location | Change | Source |
|---|---|---|
| main.tex "Proxies for large-graph accuracy" | Replaced the old sentence mixing two candidate pools with same-candidate Spearman values and CIs; added `fig_proxy` and `tab_proxy` | `experiments/final/E5_proxy/{summary.csv,labels_same_candidates.csv}` |

## 2026-11-21 mechanical report changes

| File | Change | Reason |
|---|---|---|
| main.tex | moved `\end{document}` from line 101 to after the AI-DRAFT section | the appended final-results section was outside the document and did not compile into the PDF |
| main.tex | wrapped `tab_proxy`, `tab_costs`, `tab_cost_counterfactual`, `tab_rule_ablation` in `\small`; escaped `%`/`_`/`&` in generated tables | fit the 1-inch margin; avoid LaTeX errors |
| SLIDES.md | Baselines: 87.3/90.0/93.3/94.0 → 89.3±1.6 / 94.4±0.6 / 95.8±1.4 (test_small, 3 repeats) + held-out 88.4/93.8/95.8 | `experiments/final/E1_heldout/summary.csv`, `E2_repeats/summary.csv` |
| SLIDES.md | Scaling: stress 62/60/74 and "v2 best at 2000 nodes" → 66.0/72.7/73.3 and B higher on large | `E2_repeats/summary.csv` |
| SLIDES.md | appended AI-DRAFT slide stubs 15-21 | new results |
| main.tex "Tool access" | single-run values (96.0/35.3) → 3-repeat means ± std, added medium/large and Baseline A | `experiments/final/E2_tool_access/summary.csv` |

## 2026-11-21 superseded numbers in existing (human) text

| Location | Old → new | Source |
|---|---|---|
| Sec. 4 table | A 87.3 → 89.3±1.6, B 90.0 → 94.4±0.6, v2(14) 94.0 → 95.8±1.4 on test_small; caption states k=3 | `experiments/final/E2_repeats/summary.csv` |
| Sec. 6 table | A 87.3/80.7/81.0/62.0 → 89.3±1.6/86.2±3.9/84.3±2.4/66.0±2.8; B 94.0/90.7/86.0/60.0 → 94.4±0.6/92.0±1.1/90.7±3.7/72.7±9.0; v2 94.0/90.0/85.0/74.0 → 95.8±1.4/90.7±0.9/85.3±0.5/73.3±0.9 | `experiments/final/E2_repeats/summary.csv` |
| Sec. 3 | added AI-DRAFT "Setup facts" block (model, splits, seeds, harnesses, sandbox, protocol, repeats) | `experiments/final/INTERFACES.md`, `data/manifest_final.json` |
| Sec. 6 table / final section | Baseline B stress 72.7±9.0 → 74.5±8.4 after adding replacement repeat 4 (4 repeats) | `experiments/final/E2_repeats/summary.csv` |

## 2026-11-21 E6/E11 integration

| Location | Change | Source |
|---|---|---|
| Sec. 8 | v2-only 20/task numbers (74.5/61.0/88.3) → A/B/v2 at 100/task and 50/task; added the 20/task subset check | `experiments/final/E6_real_bench/summary.csv` |
| final section "Real benchmarks" | added A/B/v2 text, `fig_real_bench`, `tab_real_bench_overall`, `tab_real_bench_per_task` | same |
| final section "No-tools at 32k" | new paragraph + `tab_notools_longctx`, `fig_notools_longctx` | `experiments/final/E11_notools_longctx/summary.csv` |

## 2026-11-21 E3/E4 integration

| Location | Change | Source |
|---|---|---|
| final section "Optimization and proxies" | replaced "all optimizers accepted zero edits" with 7B-critic outcomes (SkillOpt 1, RRSI 1, GEPA 0, Meta 0) and changed-harness test numbers; added `tab_optimizer_outputs` and `tab_optimizers` | `experiments/final/E3_optimizers/{summary.csv,eval_*}` |
| final section | counterfactual ratio 1.90 -> 1.94 (fixed denominator = six 4B-critic pilot runs) | `experiments/final/E9_costs/counterfactual.csv` |

## 2026-11-21 remaining-experiment integration

| Location | Change | Source |
|---|---|---|
| final section | added "Overfitting probe" paragraph + `tab_overfit`, `fig_overfit_probe` | `experiments/final/E3_probe/summary.csv` |
| final section | added "Router at scale" paragraph + `tab_router120`, `fig_router120` | `experiments/final/E7_router/summary.csv` |
| final section | added "Not completed" paragraph (E10/E5/E14, learned-router bug, decomposition timeout) | `experiments/final/{RESULTS_INDEX,JOBS}.md` |

## 2026-11-21 HDA section

| Location | Change | Source |
|---|---|---|
| main.tex new Section 5 | HDA methodology: motivation, input/output, tasks, generator, loop, conditions, anti-memorization, evaluation, logging; AI-DRAFT markers | `HDA_RUNBOOK.md`, `hda_pipeline/` |
| main.tex HDA results | human-reference/identifiability summary and reduced Qwen3-4B pilot table | `experiments/hda/{human_reference,identifiability.json,qwen_cpu_pilot2}` |
| tables/hda/* | generated task, split, identifiability, human-reference and pilot tables | `hda_pipeline/scripts/make_hda_{tables,results}.py` |
| main.tex protocol table | HDA input/tools/output summary | runbook 1.1--1.3 |

## 2026-11-21 standalone HDA report

| File | Change |
|---|---|
| `report/hda_report.tex` / `.pdf` | new 8-page report covering only the HDA runs: methods in detail, H0 verification, human reference, identifiability, Qwen3-4B Q4 pilot, hidden smoke, limitations and reproduction |

## 2026-11-22 HDA course-report integration

| Location | Change | Source |
|---|---|---|
| main.tex HDA section | added combined loop/message diagram, all-task given results table and heatmap; replaced pilot text with corrected 10-task results | experiments/hda/all_given_fixed_3it, report/hda_report.tex |

## 2026-11-22 compiler-tool ablation (inside new AI-DRAFT block)

| Location | Change | Source |
|---|---|---|
| `hda_paper.tex` abstract | Added the compiler-off/on validation and failure counts | `experiments/hda/nocompiler_grid_all10/summary.csv` |
| `hda_paper.tex` new "Compiler-tool ablation" | Added method, results, paired bootstrap, per-task exceptions, cost, and caveats | `experiments/hda/nocompiler_grid_all10/{compare_all10.csv,summary.csv,bootstrap.json,per_task_delta.csv}` |
| `tables/hda_nocompiler/tab_nocompiler.tex` | Accuracy by condition and split, compiler on/off | `summary.csv` |
| `tables/hda_nocompiler/tab_nocompiler_ci.tex` | Paired bootstrap differences and 95% CIs | `bootstrap.json` |
| `figures/hda_nocompiler/fig_nocompiler_accuracy.pdf` | Accuracy by split, two panels | `compare_all10.csv` |
| `figures/hda_nocompiler/fig_nocompiler_effects.pdf` | Per-task effect and win/tie/loss counts | `compare_all10.csv` |
| `figures/hda_nocompiler/fig_nocompiler_cost.pdf` | Prompt tokens and failed submissions | `compare_all10.csv`, `summary.csv` |
| `hda_paper.pdf` | Rebuilt with Tectonic 0.17 (no system LaTeX); 12 pages | `hda_paper.tex` |
