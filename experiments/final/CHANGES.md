# New scripts, modules, flags and splits for final experiments

## Code and flags

- `src/graphharness/stats.py` (+`tests/test_stats.py`): `wilson_ci`,
  `bootstrap_ci`, `mcnemar`, `aggregate_repeats`.
- `src/graphharness/evoskill.py` (+ test in `tests/test_optimizer.py`):
  EvoSkill-style skill library, validator, population loop. CLI mirrors
  `optimizer.py`, with `--critic-model` / `--critic-base-url`.
- `src/graphharness/meta_harness.py`: added `--critic-base-url` (default keeps
  the old behaviour of using the actor endpoint).

## Scripts

- `scripts/final/wait_for_vllm.sh`: poll an OpenAI-compatible endpoint.
- `scripts/final/check_disjoint.py`: canonical graph-hash leakage check.
- `scripts/final/run_e1_e2.sbatch`: E1 held-out repeats, E2 scaling repeats,
  E2 tool-access repeats on one server.
- `scripts/final/run_e2_fixup.sbatch`: replacement Baseline B `test_stress`
  repeat.
- `scripts/final/run_e3.sbatch` (full, cancelled), `run_e3_lite.sbatch`
  (reduced, submitted), `run_e4.sbatch` (full, cancelled).
- `scripts/final/run_e6.sbatch`: A/B/v2 on the expanded real benchmarks.
- `scripts/final/aggregate.py`: long-format run summary reader.
- `scripts/final/aggregate_e1_e2.py`: E1/E2 summaries, per-task rows,
  McNemar, Wilson CIs, Δ_gen.
- `scripts/final/e5_proxy_existing.py`: 17-candidate static proxy against
  existing `test_large` labels.
- `scripts/final/e5_proxy_same_candidates.py`: same 15 rule-ablation
  candidates labelled on both `val` and `test_large`, with bootstrap CIs.
- `scripts/final/e8_regression.py`: family x size table and logistic
  regression (Newton-Raphson; statsmodels is not installed).
- `scripts/final/e9_costs.py`: cost curves, GPU-hour/cost columns and the
  large-graph counterfactual.
- `scripts/final/make_tables.py`, `make_figures.py`: regenerate
  `report/tables/tab_*` and `report/figures/fig_*`.
- `scripts/final/lint_tex.py`: citation spacing, missing includes, AI-DRAFT
  inventory, optional Tectonic compile + page check.
- `scripts/final/list_ai_drafts.py`: prints every AI-DRAFT block (see
  `report/AI_DRAFTS.md`).

## Splits and configs

- `configs/final_test_small_fresh.json`: `test_small_fresh`, seed 56240,
  30/task x 5 tasks, |V| 4-50. Written to `data_final/` and
  `data/manifest_final.json`; the original `data/manifest.json` is untouched.
- `data/grapharena_eval100/`: GraphArena easy/hard, 100/task (2,000 records).
- `data/graphqa_eval50/`: GraphQA test, 50/source task (600 records).
  Both are gitignored; every converted record was truth-scored (bad = 0).

## Report

- `report/main.tex`: final-results section inside one AI-DRAFT block.
- `report/tables/tab_*.{csv,tex}`, `report/figures/fig_*.{pdf,png}`.
- `report/EDIT_LOG.md`, `report/AI_DRAFTS.md`, `experiments/final/E13_lint.md`.
