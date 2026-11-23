# Compiler-tool ablation (no-compiler prompt)

## Design

- HDA: Qwen3-4B-Instruct-2507, BF16 vLLM, Falcon a30.
- Frozen config: k=5, 3 iterations, 8 steps, temperature 0.2, 4,096-token
  replies, 32k context.
- Compiler-off (`--no-compiler`): no `compile`, `test`, `run` or `stress`
  action, no pre-submit compile check, and no compiler diagnostics in the
  submission-rejection message. The evaluator still compiles the submitted
  program; a compile failure scores zero.
- Compiler-on baseline: the matched unoptimized final grid in
  `experiments/hda/optimizers/final/unoptimized` (same config, same seeds).
- Grid: ten phase-1 tasks x {given, hidden} x seeds 0-2 = 60 compiler-off runs.
  Job 927197, 19:27-19:50, 19m50s.
- Pilot: `max_flow`/given, seeds 0-2, job 927115, completed in 6m27s.

## Main result

| Condition | Arm | val_opt | small | medium | large | xl | stress | prompt tokens | failed submits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| given | on | 0.986 | 0.983 | 0.967 | 0.958 | 0.960 | 0.957 | 53,755 | 2 |
| given | off | 0.888 | 0.876 | 0.876 | 0.880 | 0.873 | 0.810 | 12,878 | 11 |
| hidden | on | 0.521 | 0.584 | 0.564 | 0.551 | 0.573 | 0.580 | 99,621 | 2 |
| hidden | off | 0.338 | 0.412 | 0.402 | 0.390 | 0.412 | 0.417 | 22,125 | 6 |
| all | on | 0.754 | 0.784 | 0.766 | 0.754 | 0.767 | 0.768 | 76,688 | 4 |
| all | off | 0.613 | 0.644 | 0.639 | 0.635 | 0.642 | 0.613 | 17,502 | 17 |

The compiler tool improves validation by 0.098 (given) and 0.183 (hidden),
and test accuracy by about 0.10-0.17. It costs 4.4x more prompt tokens.

## Compile failures

- Compiler-on: 4 failed submissions in 60 runs (2 compile errors, 2 policy/no
  code).
- Compiler-off: 17 failed submissions in 60 runs, 16 with compiler errors.
- `connected_components`/given fails all three seeds without the compiler:
  every submission has a compile error (`find` used before `auto` deduction),
  so val=0 and no test result. Compiler-on solves it (one seed needs one fix).

## Heterogeneity

Per-task deltas are in `per_task_delta.csv`. Most hidden tasks get much worse
without the compiler (`common_neighbors` -0.54, `connected_components` -0.67,
`bipartite_check` -0.37, `max_flow` -0.32, `cycle_detection` -0.30). Two hidden
tasks look better without it (`shortest_path` +0.34, `reachability` +0.12), but
that is driven by compiler-on seed failures in statement identification, not by
compilation: compiler-on `shortest_path` seed 1 scored 0 instead of 1.0.

`max_flow`/given is the scale outlier: compiler-off stress is 0.533 vs 0.700,
and compiler-on seed 0 alone accounts for the medium/large/xl gap.

## Caveats

- Three seeds and one model; per-task differences can be one stochastic draw.
- Missing test splits are treated as 0 (the program did not compile), not as
  missing observations.
- The no-compiler arm still gets aggregate validation feedback after each
  submission, including whether the submission compiled; only diagnostics and
  interactive tools are removed.

## Paired bootstrap

Paired over the 60 (task, condition, seed) pairs, 10,000 resamples, 95% CI for
compiler-off minus compiler-on:

| Metric | Difference | 95% CI |
|---|---:|---|
| val_opt | -0.141 | [-0.240, -0.047] |
| test_small | -0.140 | [-0.235, -0.048] |
| test_medium | -0.127 | [-0.221, -0.037] |
| test_large | -0.119 | [-0.221, -0.023] |
| test_xl | -0.124 | [-0.225, -0.031] |
| test_stress | -0.155 | [-0.258, -0.057] |

Every interval excludes zero: the compiler-tool benefit is not explained by
seed noise at this sample size.
