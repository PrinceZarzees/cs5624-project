# Slide deck outline

This is an outline for the human-authored presentation; it is not final prose.

1. **Motivation** — can a harness learned on 4–50-node graphs transfer to 100–2000?
2. **Tasks** — shortest path, components, cycles, diameter, reachability,
   max clique, Hamiltonian path, common neighbors, MIS, MVC, GED, MCS, TSP, and
   GraphQA scalar tasks.
3. **Data** — synthetic splits, GraphArena 10 tasks, GraphQA 12 tasks.
4. **Harness** — prompt/rules, sandbox, tool loop, answer protocol.
5. **Models** — Qwen3-4B-Instruct-2507 primary; Qwen2.5-7B CPU fallback.
6. **Baselines** — A 87.3%, B 90.0%, B v2 93.3/94.0% on full test_small.
7. **Optimization** — two runs, all automated edits rejected; validation gate.
8. **Scaling** — A/B/B v2 curves; stress 62/60/74%; v2 best at 2000 nodes.
9. **Ablation** — test_large and test_stress rule deltas; val selection.
10. **Real benchmarks** — GraphArena easy 74.5%, hard 61.0%, GraphQA 88.3%.
11. **Error analysis** — GED, MCS, TSP, hard MIS, maximum flow.
12. **Limitations** — synthetic vs real, test_small tuning, source-answer tasks.
13. **Conclusion** — tooling gains, human expert harness, scale-specific rules.
14. **Artifacts** — repository layout, commands, seeds, model snapshots.

Figures: `report/figures/`.
Tables: `report/tables/`.
