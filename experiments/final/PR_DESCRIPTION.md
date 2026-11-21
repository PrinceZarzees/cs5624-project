# PR: Final experiments and compiled report draft

**Suggested title**

> Final experiments: fresh held-out split, repeats, strong-critic optimizers, expanded real benchmarks, and a compiled report draft

**Base:** `master`  
**Compare:** https://github.com/PrinceZarzees/cs5624-project/compare/master...saminrk-experiments?expand=1

---

## What this PR contains

This branch executes the remaining experiments from the project runbook and
integrates every completed result into the report as clearly marked
**AI-DRAFT** text. The authors must rewrite that text before submission.

The final-experiment work is the last six commits:

- `01e6a4b` E1/E2 repeats, E5/E8/E9 CPU analyses, E3/E4/E11 job scripts
- `144f6d0` report final-results section, tables/figures, AI-DRAFT markers
- `2e0b45a` report numbers updated to 3-repeat means, setup-facts draft
- `e97c604` E3/E6/E11 job updates and aggregators
- `5cb495f` E3/E4 strong-critic results, E6/E11 evaluations
- `7ecd71c` report integration and compiled PDF

> Relative to `master` the branch also carries the earlier experiment
> artifacts, because `saminrk-dev` was never pushed. Reviewers can focus on the
> six commits above; the rest is prior work.

---

## Key results

### Fresh held-out estimate (`test_small_fresh`, seed 56240, 150 problems, 0 overlap)

| Harness | Accuracy (3 repeats) |
|---|---|
| Baseline A | 88.4 ± 1.3% |
| Baseline B | 93.8 ± 0.8% |
| Baseline B v2 | **95.8 ± 0.8%** |

v2's contaminated `test_small` value was 94.0%, so the held-out estimate is not
lower. Per task, v2 improves cycle detection over B (95.6% vs 82.2%) and is
lower on max clique (83.3% vs 88.9%).

### Scaling with error bars (mean ± std over 3 repeats)

| Harness | small | medium | large | stress |
|---|---|---|---|---|
| Baseline A | 89.3 ± 1.6 | 86.2 ± 3.9 | 84.3 ± 2.4 | 66.0 ± 2.8 |
| Baseline B | 94.4 ± 0.6 | 92.0 ± 1.1 | **90.7 ± 3.7** | 74.5 ± 8.4 |
| Baseline B v2 | **95.8 ± 1.4** | 90.7 ± 0.9 | 85.3 ± 0.5 | 73.3 ± 0.9 |

The earlier single-run claim that v2 is best on large graphs does not survive
repeats: B is higher on `test_large` in all three repeats (significant in
repeat 2, McNemar p=0.006), and the two are within ~1 point on stress.

### Tool access (3 repeats)

| Condition | small | medium | large (n=25) |
|---|---|---|---|
| A + tools | 90.4 ± 0.8 | 86.4 ± 1.9 | 84.0 ± 3.3 |
| A no tools (8k) | 39.6 ± 2.8 | 15.6 ± 0.3 | 0.0 ± 0.0 |
| v2 + tools | 96.4 ± 0.3 | 91.1 ± 1.9 | 92.0 ± 3.3 |
| v2 no tools (8k) | 35.8 ± 0.6 | 15.3 ± 0.5 | 0.0 ± 0.0 |

### No-tools at 32k (removes the 8k confound)

- Medium: 18.0% (A), 16.7% (v2); 141/150 prompts already fit in 8k, so the low
  medium score is not a context artifact.
- Large: 11.0% (A), 10.0% (v2); no prompt fits in 8k, 7/100 exceed even 32k
  (max ~40k tokens) and score 0.
- Tools remain at 84–92% on large.

### Optimizers with a stronger critic (Qwen2.5-7B-Instruct BF16, start = Baseline B)

| Optimizer | Accepted / rounds | Val init → best |
|---|---|---|
| SkillOpt-style | **1 / 6** | 98% → 100% |
| RRSI-lite | **1 / 5** | 96% → 98% |
| RRSI-lite (second run) | **1 / 6** | 97% → 99% |
| GEPA-style | 0 / 6 | 100% → 100% |
| Meta-Harness-lite | 0 / 6 | 98% → 98% |
| EvoSkill-style | 0 / 3 | 100% → 100% |

Accepted edits are generic verification rules; none contains a numeric node id.
Changed-harness evaluation (one run each, fresh/medium/large/stress):

- SkillOpt output: 98.7 / 94.0 / 94.0 / 76.0%
- RRSI output: 96.0 / 96.0 / 94.0 / 74.0%
- Baseline B means: 94.4 / 92.0 / 90.7 / 74.5%

The 2–4 point differences are within the observed repeat spread.

### Expanded real benchmarks (GraphArena 100/task, GraphQA 50/task)

| Benchmark | A | B | B v2 |
|---|---|---|---|
| GraphArena easy (n=1000) | 72.9% | **75.6%** | 73.7% |
| GraphArena hard (n=1000) | 59.1% | **62.6%** | 60.9% |
| GraphQA test (n=600) | 84.3% | 89.0% | **89.2%** |

On the original 20/task subset, v2 reproduces the frozen values within 2 points
(76.5/62.5/90.4 vs 74.5/61.0/88.3). B is best on GraphArena; v2 ties B on
GraphQA.

### Other completed analyses

- **Characteristics (E8):** family × size heatmap and logistic regression.
  Odds ratios: 0.63 per log-n (95% CI 0.55–0.72), 0.76 per 0.1 density
  (0.64–0.90), 1.91 for v2 vs A (1.29–2.81).
- **Costs (E9):** 10-round optimization directly on large graphs is estimated
  at 3.52M tokens / 0.74 GPU-hours, **1.94×** the average of the six
  4B-critic pilot runs.
- **Proxies (E5, reduced):** on the same 15 rule-ablation candidates, the
  learned static proxy gets Spearman −0.25 (95% CI −0.73 to 0.41) against val
  and +0.29 (−0.36 to 0.85) against `test_large`; top-3 recall is zero.

---

## Report artifacts

- **Compiled draft:** [`report/main.pdf`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/report/main.pdf)
  (Tectonic 0.17.0, 13 pages, 0 compile errors).
- **Source:** [`report/main.tex`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/report/main.tex)
- **AI text checklist:** [`report/AI_DRAFTS.md`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/report/AI_DRAFTS.md)
- **Change log (old → new numbers):** [`report/EDIT_LOG.md`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/report/EDIT_LOG.md)
- **Generated tables:** [`report/tables/`](https://github.com/PrinceZarzees/cs5624-project/tree/saminrk-experiments/report/tables)
- **Generated figures:** [`report/figures/`](https://github.com/PrinceZarzees/cs5624-project/tree/saminrk-experiments/report/figures)

Every new or changed block of report prose is inside
`% >>> AI-DRAFT ... % <<< AI-DRAFT` markers. Existing prose was not deleted;
numbers superseded by repeats were updated and logged.

---

## Experiment artifacts

Start with:

- [`experiments/final/RESULTS_INDEX.md`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/RESULTS_INDEX.md) — feedback → experiment → file map.
- [`experiments/final/JOBS.md`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/JOBS.md) — every Slurm job, status and output path.
- [`experiments/final/CHANGES.md`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/CHANGES.md) — new scripts, modules, flags and splits.
- [`experiments/final/INTERFACES.md`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/INTERFACES.md) — real CLI flags and serving pattern.

Per-experiment outputs:

| Experiment | Link |
|---|---|
| E0 smoke | [`E0_smoke_cpu/`](https://github.com/PrinceZarzees/cs5624-project/tree/saminrk-experiments/experiments/final/E0_smoke_cpu) |
| E1 held-out | [`E1_heldout/summary.csv`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E1_heldout/summary.csv) |
| E2 repeats | [`E2_repeats/summary.csv`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E2_repeats/summary.csv) |
| E2 tool access | [`E2_tool_access/summary.csv`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E2_tool_access/summary.csv) |
| E3 optimizers | [`E3_optimizers/summary.csv`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E3_optimizers/summary.csv) |
| E4 EvoSkill | [`E4_evoskill/baseline_b/`](https://github.com/PrinceZarzees/cs5624-project/tree/saminrk-experiments/experiments/final/E4_evoskill/baseline_b) |
| E5 proxies | [`E5_proxy/summary.csv`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E5_proxy/summary.csv) |
| E6 real benchmarks | [`E6_real_bench/summary.csv`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E6_real_bench/summary.csv) |
| E8 characteristics | [`E8_characteristics/regression.csv`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E8_characteristics/regression.csv) |
| E9 costs | [`E9_costs/counterfactual.csv`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E9_costs/counterfactual.csv) |
| E11 32k no-tools | [`E11_notools_longctx/summary.csv`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E11_notools_longctx/summary.csv) |
| E13 lint/page check | [`E13_lint.md`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/experiments/final/E13_lint.md) |

---

## Code changes

- [`src/graphharness/stats.py`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/src/graphharness/stats.py) — Wilson/bootstrap CIs, McNemar, repeat aggregation (+ tests).
- [`src/graphharness/evoskill.py`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/src/graphharness/evoskill.py) — EvoSkill-style skill library and loop (+ test).
- `meta_harness.py` — added `--critic-base-url` (default preserves old behavior).
- [`scripts/final/`](https://github.com/PrinceZarzees/cs5624-project/tree/saminrk-experiments/scripts/final) — run scripts, aggregators, table/figure generators, lint and AI-draft lister.
- [`data/manifest_final.json`](https://github.com/PrinceZarzees/cs5624-project/blob/saminrk-experiments/data/manifest_final.json) — manifest for the new split; the original `data/manifest.json` is untouched.

Large artifacts (predictions, scores, trajectories, sqlite caches, sandbox dirs,
data dirs) are gitignored.

---

## Verification

- `pytest -q`: **179 passed**, 1 warning.
- `python -m graphharness.verify_data`: all six original splits OK; default
  `build_data` output unchanged.
- New split leakage check: **0 overlap** with the six frozen splits.
- Expanded real-benchmark data: all 2,600 test records truth-scored, **0 bad**.
- Paper compiles with Tectonic: **13 pages, 0 errors**; only a pre-existing
  overfull path warning remains.
- No test split was read by optimizer, router or proxy training code.
- Every report change is inside an AI-DRAFT block or listed in
  `report/EDIT_LOG.md`.

---

## Not included / known limitations

- **E10** language × library 2×2 was not run.
- **E7** has only the existing 8-problem router pilot; the learned router and
  full 120-problem rerun were not run.
- **E5** is reduced: 15–18 candidates and no evolved rubric.
- The **RRSI overfitting probe** and **E14 cross-model check** were not run.
- Sections 1–5 still end on **page 2**, below the 3-page minimum. Intro and
  Related Work are empty and must be written by the authors; see
  `experiments/final/E13_lint.md`.
- **All AI-DRAFT prose must be rewritten by the authors before submission.**

---

## Suggested review order

1. `report/main.pdf` — compiled draft.
2. `experiments/final/RESULTS_INDEX.md` — what each result answers.
3. `report/EDIT_LOG.md` and `report/AI_DRAFTS.md` — what changed and what must
   be rewritten.
4. `experiments/final/JOBS.md` — how each number was produced.
