# Final report

See [`EXPERIMENTAL_REPORT.md`](EXPERIMENTAL_REPORT.md) for the full methods-and-results report.

This directory is for the final ACL-style paper and presentation material. The
course requires human-written prose, so the repository only provides structure
and generated tables/figures; it deliberately does not contain drafted paper
text.

Suggested sections and source artifacts:

1. **Introduction** - research question and contributions.
2. **Related Work** - harness optimization, graph reasoning benchmarks.
3. **Problem Setup** - tasks, splits, harness definition, metrics.
4. **Baselines** - `experiments/qwen3_4b/baseline/`.
5. **Small-Graph Optimization** - `experiments/qwen3_4b/optimizer/` and
   `experiments/qwen3_4b_opt_a/optimizer/` (`history.jsonl`,
   `optimization_summary.json`).
6. **Scaling Results** - `experiments/qwen3_4b_scaling_full/scaling/`
   (`scaling_summary.{json,txt}`, `scaling_by_task.md`, `failure_analysis_v2.txt`).
7. **Rule Ablations** - `experiments/qwen3_4b_ablation_full/ablation/` and
   `experiments/qwen3_4b_ablation_stress/ablation/`.
8. **Error Analysis** - trajectories under the run directories and
   `failure_analysis*.txt`.
9. **Limitations & Discussion** - synthetic data, model/API choices, sandbox
   assumptions.
10. **Artifact Appendix** - repository layout, exact commands, seeds.

Before submission, archive the commands, model identifiers, seeds, and the
`scaling_summary.json` used for each table.

## Current headline numbers (Qwen3-4B-Instruct-2507, BF16, B200)

Full `test_small` (150 problems):

| Harness | Accuracy |
|---|---:|
| Baseline A | 87.3% |
| Baseline B | 90.0% |
| Baseline B v2 | 93.3% (13 rules) / 94.0% (14 rules, scaling run) |

Full scaling (small/medium/large/stress):

| Harness | small | medium | large | stress |
|---|---:|---:|---:|---:|
| Baseline A | 87.3% | 80.7% | 81.0% | 62.0% |
| Baseline B | 94.0% | 90.7% | 86.0% | 60.0% |
| Baseline B v2 | 94.0% | 90.0% | 85.0% | 74.0% |

Automated optimization (8 rounds from v2; 10 rounds from A) produced no accepted
improvement; the validation gate rejected all candidates. The validation
ablation (v2: 95.3% on `val`) found no meaningful rule removal; v2 is frozen as
`H*_small`. Full rule ablations are under
`experiments/qwen3_4b_ablation_full/`, `..._stress/`, and `..._val/`.

Real benchmarks (20 problems per task): GraphArena easy **74.5%**, hard
**61.0%**, GraphQA test **88.3%**. These expose the main remaining weaknesses:
GED, MCS, TSP, hard MIS and maximum flow.

Figures in `report/figures/`:

- `scaling_4b_accuracy_vs_size.*`
- `scaling_4b_accuracy_by_task.*`
- `test_small_baselines_4b*`
- `cpu_4b_vs_7b_by_task.png`
- `cross_model_accuracy_by_task.png`
- `cross_model_accuracy_vs_size.png`
- `real_benchmarks_by_task.png` / `.pdf`

See `experiments/RESULTS.md` for the full tables and caveats.
