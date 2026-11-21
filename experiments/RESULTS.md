# Current results snapshot

These are pilot and full-`test_small` results. Harness/tool versions changed
between some rows; the exact artifacts and run configs are in the referenced
directories.

## Qwen3-4B-Instruct-2507, BF16, one B200

| Run | Harness | Split / n | Accuracy | Artifact |
|---|---|---:|---:|---|
| Baseline A pilot | `baseline_a` | `test_small`, 25 | 20% | `qwen3_4b_pilot/baseline_a/` |
| Baseline B pilot | `baseline_b` | `test_small`, 25 | 72% | `qwen3_4b_pilot/baseline_b/` |
| Optimizer round 1 | `baseline_b` → candidate | `val`, 20 | 70% → 40% (rejected) | `qwen3_4b_pilot/optimizer_1round/` |
| Best harness eval | `baseline_b` | `test_small`, 25 | 68% | `qwen3_4b_pilot/optimizer_1round/eval_best/` |
| Full pipeline | A, B, B_v2 | `test_small`, 150 | see below | `qwen3_4b/` |
| Full CPU fallback, A | `baseline_a` | `test_small`, 150 | 84.7% | `qwen3_4b_cpu/baseline_a/test_small/` |
| Full CPU fallback, B_v2 | `baseline_b_v2` (13 rules) | `test_small`, 150 | **93.3%** | `qwen3_4b_cpu/baseline_b_v2/test_small/` |

The optimizer candidate switched to a full `edge_list` graph view, a clear
small-graph overfitting move, and was rejected by the validation gate.

Key failure findings from the 4B pilot:

- Baseline A often ignored the sandbox and invented node labels (`0,1,2`, `a,b,c`).
- Baseline B computed the diameter but then selected an arbitrary node pair for
  the witness path; it also omitted `print(...)` for connected components.
- These led to the sandbox echo, code-before-answer handling, final-answer turn,
  and `baseline_b_v2` rules.


## Qwen3-4B-Instruct-2507, BF16, one B200: full test_small

| Harness | Accuracy | Shortest path | Components | Cycles | Diameter | Max clique |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_a` | 87.3% | 83.3% | 100% | 73.3% | 86.7% | 93.3% |
| `baseline_b` | 90.0% | 93.3% | 76.7% | 100% | 100% | 80.0% |
| `baseline_b_v2` | **93.3%** | 100% | 100% | 96.7% | 86.7% | 83.3% |

Optimizer from `baseline_b_v2`: 8 rounds, initial validation 97.5%, best
97.5%; all candidates rejected (validation 45-75%). Optimizer from
`baseline_a`: 10 rounds, initial validation 87.5%, best 87.5%; all candidates
rejected (validation 37.5-72.5%). The validation gate prevented regressions,
but the automated critic did not find an improvement over the human harness.
The best harness remained the hand-written v2 harness.

## Full scaling: Qwen3-4B BF16

Full split sizes: small 150, medium 150, large 100, stress 50.

| Harness | small | medium | large | stress | Δ_gen (small-large) |
|---|---:|---:|---:|---:|---:|
| `baseline_a` | 87.3% | 80.7% | 81.0% | 62.0% | +6.3% |
| `baseline_b` | 94.0% | 90.7% | 86.0% | 60.0% | +8.0% |
| `baseline_b_v2` | 94.0% | 90.0% | 85.0% | **74.0%** | +9.0% |

The expert v2 harness is the best at 2,000 nodes by 12-14 points. All harnesses
lose accuracy with scale; the largest drop is on the stress split.

## Rule ablation on full `test_large` (100 problems)

Full harness accuracy: **85.0%**. `delta = full - without-rule`; positive means
the rule helped.

| Removed rule | Ablated accuracy | delta |
|---|---:|---:|
| 0. Use `G` directly | 89.0% | -4.0% |
| 1. Prefer exact routines | 88.0% | -3.0% |
| 2. Clique search policy | 88.0% | -3.0% |
| 3. Feasibility checks | 86.0% | -1.0% |
| 4. Keep output short | 89.0% | -4.0% |
| 5. Change failing code | 86.0% | -1.0% |
| 6. Answer format | 83.0% | +2.0% |
| 7. CC print/representatives | 86.0% | -1.0% |
| 8. Diameter witness pair | 90.0% | -5.0% |
| 9. Explicit print | 90.0% | -5.0% |
| 10. `nx.max_weight_clique` | 89.0% | -4.0% |
| 11. Stop after success | 89.0% | -4.0% |
| 12. No numeric node filtering | 87.0% | -2.0% |
| 13. Materialize generators | 88.0% | -3.0% |

Most rules help by 1-5 points; the diameter/print rules matter most. Removing
the answer-format rule slightly improves the measured result (+2), which is
within the noise of a 100-problem split and worth rechecking on more seeds.

## Qwen2.5-7B-Instruct Q4_K_M, CPU (llama.cpp)

| Run | Harness | Split / n | Accuracy | Artifact |
|---|---|---:|---:|---|
| Before parser/loop fixes | `baseline_b_v2` (old) | `test_small`, 25 | 20% | `qwen2_5_7b_cpu_pilot/baseline_b_v2/` |
| After fixes, 10-problem check | `baseline_b_v2` (old) | `test_small`, 10 | 60% | `qwen2_5_7b_cpu_pilot/baseline_b_v2_parserfix2/` |
| 25-problem pilot | `baseline_b_v2` (old) | `test_small`, 25 | 68% | `qwen2_5_7b_cpu_pilot/baseline_b_v2_25/` |
| Weak-task check, new rules | `baseline_b_v2` (new) | diameter+clique, 10 | 80% | `qwen2_5_7b_cpu_pilot/baseline_b_v2_newrules/` |
| Final 25-problem pilot | `baseline_b_v2` (new) | `test_small`, 25 | aborted at 19/25 | `qwen2_5_7b_cpu_pilot/baseline_b_v2_final25_aborted/` |
| Full 150-problem run | `baseline_b_v2` (13 rules) | `test_small`, 150 | **77.3%** | `qwen2_5_7b_cpu/baseline_b_v2/test_small/` |
| Full 150, 14-rule v2 | `baseline_b_v2` (14 rules) | `test_small`, 150 | 76.0% | `qwen2_5_7b_cpu/baseline_b_v2_14rules/test/small/` |

The CPU models are quantized; the 4B runs at about 18 tokens/s and the 7B at
about 12 tokens/s. The 4B is both smaller and more accurate on this benchmark
(93.3% vs 77.3%; 14-rule check 94.0% vs 76.0%), so Qwen3-4B-Instruct-2507 remains the primary model.

## Harness fixes found from these pilots

- Sandbox echoes a bare final expression (notebook-style).
- Tool/proposal parsing accepts Python-literal JSON (triple-quoted strings).
- Code is always executed before a same-message guessed answer is accepted.
- After the tool-step limit, the agent gets one final-answer turn.
- `baseline_b_v2` adds explicit connected-components, diameter-witness,
  max-clique and stop-after-success rules.

See `REPRODUCIBILITY.md` for model snapshots and environment versions.

## Rule ablation on full `test_stress` (50 problems)

Full harness accuracy: **76.0%**. Positive delta means the rule helped.

| Removed rule | Ablated accuracy | delta |
|---|---:|---:|
| 0. Use `G` directly | 76.0% | 0.0% |
| 1. Prefer exact routines | 66.0% | +10.0% |
| 2. Clique search policy | 70.0% | +6.0% |
| 3. Feasibility checks | 70.0% | +6.0% |
| 4. Keep output short | 76.0% | 0.0% |
| 5. Change failing code | 76.0% | 0.0% |
| 6. Answer format | 74.0% | +2.0% |
| 7. CC print/representatives | 76.0% | 0.0% |
| 8. Diameter witness pair | 72.0% | +4.0% |
| 9. Explicit print | 66.0% | +10.0% |
| 10. `nx.max_weight_clique` | 68.0% | +8.0% |
| 11. Stop after success | 72.0% | +4.0% |
| 12. No numeric node filtering | 74.0% | +2.0% |
| 13. Materialize generators | 70.0% | +6.0% |

At 2,000 nodes, exact-routine, explicit-print and max-clique rules matter most;
several rules are neutral. The generator-materialization rule adds 6 points,
validating the targeted diameter fix.

## Takeaways

- **Qwen3-4B-Instruct-2507 is the right lightweight starting model.** It is
  smaller than the 7B fallback and much more accurate on this benchmark
  (93.3% vs 77.3% full `test_small`).
- **Tooling mattered as much as the prompt.** Sandbox expression echo, relaxed
  JSON parsing, code-before-answer execution and the final-answer turn lifted
  the 4B pilot from 20% to 93.3%.
- **The human expert v2 harness is the best frozen harness.** Automated
  optimization from both `baseline_a` and `baseline_b_v2` rejected every
  candidate; the validation gate worked, but the 4B critic did not propose a
  better harness.
- **Generalization gap is 6-9 points from small to large**, but the stress
  split is much harder: 62% (A), 60% (B), 74% (v2). The v2 rules transfer best
  to 2,000 nodes.
- **Most v2 rules help on `test_large`** by 1-5 points; diameter-witness and
  explicit-print rules have the largest effect. The answer-format rule removal
  showed +2, within noise.

## Methodological caveat

`baseline_b_v2` was written after inspecting pilot failures on `test_small`.
Its `test_small` score is therefore **not a clean held-out estimate**; the
original `baseline_b` is the clean human baseline, and `test_medium`,
`test_large`, and `test_stress` are the more informative held-out scales for
v2. The automated optimizer, in contrast, uses only `train` and `val`; its
negative result is clean.

## Error patterns at scale

Per-task tables are in
`experiments/qwen3_4b_scaling_full/scaling/scaling_by_task.md`.

- Shortest path and connected components remain near 100% for v2 through
  `test_large`; stress shortest path is 80%.
- Cycle detection stays at 90-100% for v2 through all scales.
- Diameter collapses on stress (20% for v2; 10% for A, 40% for B), even though
  it is 90% on small/medium/large.
- Max clique is the main large-graph weakness (40% on `test_large`, 63% on
  medium, 80% on small and stress). Exact search becomes expensive and the
  model falls back to small cliques.

## Validation-based harness selection

Full `val` (150 problems) for v2: **95.3%**. Removing each rule gives:

| Removed rule | Val accuracy | delta |
|---|---:|---:|
| 0 | 92.7% | +2.7% |
| 1 | 94.0% | +1.3% |
| 2 | 93.3% | +2.0% |
| 3 | 92.7% | +2.7% |
| 4 | 93.3% | +2.0% |
| 5 | 83.3% | +12.0% |
| 6 | 92.7% | +2.7% |
| 7 | 90.7% | +4.7% |
| 8 | 96.0% | -0.7% |
| 9 | 90.7% | +4.7% |
| 10 | 95.3% | 0.0% |
| 11 | 92.0% | +3.3% |
| 12 | 93.3% | +2.0% |
| 13 | 92.0% | +3.3% |

Only rule 8 (diameter witness) improves on `val` when removed, and only by
0.7% (one problem). That same rule is worth +5 points on `test_large` and +4
on `test_stress`. Dropping it because of a one-problem `val` fluctuation would
be exactly the small-to-large overfitting the project is studying. We therefore
freeze v2 as `H*_small`.

The complete val ablation is in `experiments/qwen3_4b_ablation_val/`.

## Real benchmark evaluation (Qwen3-4B BF16, 20 problems per task)

`baseline_b_v2`, job `1540814`, artifacts in
`experiments/qwen3_4b_real_bench/bench/`.

| Benchmark | Split | n | Accuracy | Parsed | Feasible |
|---|---|---:|---:|---:|---:|
| GraphArena | `ga_easy` | 200 | **74.5%** | 95.0% | 84.0% |
| GraphArena | `ga_hard` | 200 | **61.0%** | 97.5% | 83.5% |
| GraphQA | `graphqa_test` | 240 | **88.3%** | 100% | 99.2% |

GraphArena per-task highlights:

- 100%: shortest path, connected components, common neighbors on easy/hard;
  components and common neighbors are 100% on both.
- 90%: diameter.
- 80-95%: max clique, MIS, MVC on easy/hard (MIS hard 20%).
- GED: 10% easy, 0% hard. The model cannot reliably produce a minimum edit
  mapping.
- MCS: 55% easy, 50% hard.
- TSP: 20% easy, 0% hard. The model finds tours but rarely the optimal one.

GraphQA per-task highlights:

- 100%: connected/disconnected nodes, edge count, edge existence, node count,
  node degree.
- 90-95%: shortest path, reachability, cycle detection, triangle counting.
- 75%: node classification (label propagation is not exposed in the question).
- 15%: maximum flow. GraphQA's maximum-flow answers are not recoverable from
  the displayed unweighted graph; the record uses the source answer and the
  model has no capacity information.

These are real-benchmark adaptation results, not the controlled synthetic
scaling experiments. They confirm the task implementations and expose clear
model/harness gaps on GED, MCS, TSP, MIS-hard and maximum flow.
