# PR: HDA phase-1 — C++ harness synthesis on polynomial-time graph tasks

**Base:** `master`
**Compare:** https://github.com/PrinceZarzees/cs5624-project/compare/master...saminrk-experiments?expand=1

This PR reports only the new Harness Developing Agent (HDA) experiments. The
earlier Python/NetworkX experiments are intentionally not described here.

## What this PR adds

- `hda_pipeline/` — tested phase-1 pipeline: 10 polynomial-time tasks, graph
  generator and splits, C++ sandbox, static policy, prompts, agent/tool loop,
  grid runner, aggregation, human references and identifiability analysis.
- `report/hda_report.pdf` — standalone 11-page HDA report (method, loop
  diagram with prompt structures, results, limitations, reproduction).
- `report/main.pdf` — course report section updated with the same loop diagram
  and the all-task results.
- `report/figures/hda/hda_diagram_full.pdf` — loop diagram plus message
  structures A–E.
- `report/tables/hda/tab_hda_given_all.{csv,tex}` and
  `report/figures/hda/fig_hda_given_all.pdf` — 10-task results.

## Query-contract fix (main correctness change)

The input line is `k q1 ... qk`, where the first number is the **number of
arguments**, not a type code. The initial models treated the leading `2` in
`2 s t` as a tag and searched for it inside `g.query`, which only held `[s,t]`,
so shortest-path returned -1.

Fix:
- `Graph` exposes `query_k`; the loader sets it.
- The prompt and examples state that the leading number is `k`.
- The four k=2 task statements state `query_k=2`, `query[0]=s`, `query[1]=t`.
- A bare `<action>test</action>` without a `cpp` block is rejected; old code is
  not silently reused.
- Submission without code or with a compile error is rejected.
- Generation stops at `</action>`; the 1k/2k output caps were removed (a 4096
  runaway guard is available).
- Rolling context keeps only the latest reply/tool result.

## Runs completed

- H0: **19 tests pass**; data build 60 s; human C++ reference **100%** on every
  task and split (max CPU 0.45 s); identifiability at k=10 matches the runbook.
- Corrected single shortest-path run: validation **1.000**, all five test splits
  **100%**.
- Given-statement grid: 10 tasks × 3 iterations, k=5, Qwen3-4B-Instruct-2507
  Q4 on CPU, 32k context, temperature 0.2.

## Results (given statement, 3 iterations)

| Task | Val | small | medium | large | xl | stress |
|---|---:|---:|---:|---:|---:|---:|
| shortest_path | 100% | 100% | 100% | 100% | 100% | 100% |
| reachability | 100% | 100% | 100% | 100% | 100% | 100% |
| connected_components | 100% | 100% | 100% | 100% | 100% | 100% |
| cycle_detection | 100% | 100% | 100% | 100% | 100% | 100% |
| common_neighbors | 100% | 100% | 100% | 100% | 100% | 100% |
| triangle_count | 100% | 100% | 100% | 96.7% | 100% | 100% |
| bipartite_check | 100% | 100% | 100% | 100% | 100% | 100% |
| mst_weight | 100% | 100% | 100% | 100% | 100% | 100% |
| diameter | 90.0% | 77.5% | 80.0% | 83.3% | 70.0% | 80.0% |
| max_flow | 100% | 100% | 100% | 100% | 100% | **30.0%** |

Key finding: **max flow is a clean small-to-large failure** — perfect
validation and 100% through 2,000 nodes, then 30% on the 10k–20k stress split.
Diameter is the weakest task across sizes. Eight tasks are perfect on every
split.

## Not included

- Hidden-statement grid for all tasks (only a smoke test and single-case
  diagnostics were run).
- gpt-oss-120b / BF16 GPU grid: the model is not on disk and the B200 queue was
  reserved; the Qwen3-4B CPU run is the completed result.
- 25-iteration runs and NP-hard tasks (phase 2).

## Verification

- `pytest -q tests`: **19 passed**.
- Human references: 100% every task/split.
- Validation answers are never sent to the HDA; only aggregate feedback is.
- Only the compiled C++ block is scored; statement/reasoning are logged.
- Relabeling checks run on every validation/test evaluation.
- Reports compile: `report/hda_report.pdf` 11 pages, `report/main.pdf` 21 pages.

## Artifacts

- Pipeline: `hda_pipeline/`
- Runs: `experiments/hda/all_given_fixed_3it/`, `experiments/hda/given_fixed/`
- Standalone report: `report/hda_report.pdf`
- Loop + prompts: `report/figures/hda/hda_diagram_full.pdf`
- Results: `report/tables/hda/tab_hda_given_all.csv`,
  `report/figures/hda/fig_hda_given_all.pdf`
- Notes: `experiments/hda/NOTES.md`, `experiments/hda/JOBS.md`

## Review order

1. `report/hda_report.pdf` (method, loop diagram, results)
2. `experiments/hda/all_given_fixed_3it/shortest_path/final.json`
3. `hda_pipeline/hda/agent.py` and `templates/graph_template.cpp` (query_k fix)
4. `experiments/hda/NOTES.md`
