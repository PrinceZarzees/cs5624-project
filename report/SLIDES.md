# Slide deck outline

This is an outline for the human-authored presentation; it is not final prose.

1. **Motivation** — can a harness learned on 4–50-node graphs transfer to 100–2000?
2. **Tasks** — shortest path, components, cycles, diameter, reachability,
   max clique, Hamiltonian path, common neighbors, MIS, MVC, GED, MCS, TSP, and
   GraphQA scalar tasks.
3. **Data** — synthetic splits, GraphArena 10 tasks, GraphQA 12 tasks.
4. **Harness** — prompt/rules, sandbox, tool loop, answer protocol.
5. **Models** — Qwen3-4B-Instruct-2507 primary; Qwen2.5-7B CPU fallback.
6. **Baselines** — test_small, 3 repeats: A 89.3±1.6%, B 94.4±0.6%, B v2 95.8±1.4%; fresh held-out split 88.4/93.8/95.8%.
7. **Optimization** — two runs, all automated edits rejected; validation gate.
8. **Scaling** — A/B/B v2 curves; stress 66.0/72.7/73.3%; B higher on large (90.7 vs 85.3).
9. **Ablation** — test_large and test_stress rule deltas; val selection.
10. **Real benchmarks** — GraphArena easy 74.5%, hard 61.0%, GraphQA 88.3%.
11. **Error analysis** — GED, MCS, TSP, hard MIS, maximum flow.
12. **Limitations** — synthetic vs real, test_small tuning, source-answer tasks.
13. **Conclusion** — tooling gains, human expert harness, scale-specific rules.
14. **Artifacts** — repository layout, commands, seeds, model snapshots.

Figures: `report/figures/`.
Tables: `report/tables/`.

<!-- >>> AI-DRAFT (E15, 2026-11-21): rewrite before submission -->
15. **Held-out estimate** — `report/tables/tab_heldout_small.tex`; v2 95.8±0.8%
    on 150 fresh problems, 3 repeats.
16. **Error bars** — `figures/fig_scaling.pdf`; mean ± std over 3 repeats.
17. **Tool access** — `figures/fig_tool_access.pdf`; 3 repeats, 8k context.
18. **Where accuracy comes from** — `figures/fig_characteristics.pdf` and
    `figures/fig_density.pdf`; logistic odds ratios in `tab_characteristics`.
19. **Optimization cost** — `tab_cost_counterfactual.tex`; large-graph
    optimization is estimated at 1.94x the small-graph run.
20. **Proxy transfer** — `figures/fig_proxy.pdf`; same 15 candidates, val vs
    test_large Spearman -0.25 / 0.29.
21. **Real benchmarks at 100/task** — `figures/fig_real_bench.pdf`,
    `tab_real_bench_per_task.tex`.
<!-- <<< AI-DRAFT -->
