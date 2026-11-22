# Review experiment results

These are the results collected in response to the reviewer comments. They are
pilots, not the final full-budget study; sample sizes and caveats are stated.

## R2. Tool access: Python/NetworkX makes the tasks easier

Full BF16 Qwen3-4B, 150 small / 150 medium / 25 large problems per condition.

| Condition | test_small | test_medium | test_large |
|---|---:|---:|---:|
| Baseline A + tools | 91.3% | 88.7% | 80.0% |
| Baseline A, no tools | 43.3% | 16.0% | 0.0% |
| Baseline B v2 + tools | 96.0% | 93.3% | 92.0% |
| Baseline B v2, no tools | 35.3% | 15.3% | 0.0% |

The no-tools large condition has 25/25 context-length errors, not model wrong
answers. Reduced CPU-Q4 replication (5/task small, 2/task medium) shows the
same direction:

| Condition | small (25) | medium (10) |
|---|---:|---:|
| A + tools | 76% | 80% |
| A no tools | 28% | 30% |
| Bv2 + tools | 92% | 100% |
| Bv2 no tools | 24% | 20% |

**Interpretation:** Python/NetworkX is not a minor convenience; it is the
difference between solvable and unsolvable at scale. A harness still matters on
top of tools (Bv2 is 4.7 points over A on small, 4.6 medium, 12 large), but the
tool backend is the dominant factor. Any claim about optimized harnesses must
therefore be conditioned on tool availability.

## R1. Cheap proxies for large-graph transfer

We scored 18 real optimizer candidates with:

1. **Static rubric** from the judge: 7 dimensions, overall 0-100.
2. **Learned static proxy**: leave-one-out linear model over prompt/rule
   features (length, rule count, keyword coverage, tool view, numeric
   shortcuts).
3. **Cheap rollout proxy**: one validation problem per task per candidate.

| Proxy | Spearman vs full val | Pairwise ranking | Best in top-3 |
|---|---:|---:|---|
| Static rubric | **-0.217** | 41.8% | no |
| Learned static proxy | **0.553** | **70.1%** | — |
| Cheap rollout (1/task) | -0.172 | 58.2% | no |

**Interpretation:** a generic high-level rubric is not a usable proxy: all
candidates score 95-100, so it cannot rank them. A learned proxy over
harness-text features works much better (ρ=0.55) at near-zero oracle cost. A
single problem per task is too noisy (ρ=-0.17); the cheap proxy needs either more
problems or a better sampling/aggregation rule. This supports SkillLift’s core
point: rubrics must be **learned/aligned to oracle rankings**, not hand-written
and used zero-shot.

## R3. Graph characteristics

`experiments/analysis_by_characteristics.md` and
`report/tables/characteristics.csv`.

Key slices from Baseline A:

| Slice | Accuracy |
|---|---:|
| Mixed-component graphs | 100% |
| ER | 85.9% |
| WS | 82.2% |
| BA | 74.7% |
| Tree / cycle detection | 62.2% |
| Very sparse (<.01 density) | 77.5% |
| Dense (.10-.30) | 87.7% |

Most of the “large graph” drop is really topology/task-specific: trees/cycles and
sparse BA graphs fail much more than dense ER graphs.

## R4/R7. Optimizer baselines and regularization

| Optimizer | Start | Rounds | Initial val | Best val | Tokens | Time |
|---|---|---:|---:|---:|---:|---:|
| SkillOpt-style | Baseline A | 10 | 87.5% | 87.5% | 4.01M | 486s |
| SkillOpt-style | Bv2 | 8 | 97.5% | 97.5% | 3.53M | 383s |
| GEPA-style | Baseline A | 3 | 95.0% | 95.0% | 1.19M | 597s |
| RRSI-lite | Baseline A | 3 | 90.0% | 90.0% | 0.51M | 426s |
| RRSI-lite | Bv2 | 3 | 100.0% | 100.0% | 0.81M | 405s |
| Meta-Harness-lite | Baseline A | 3 | 90.0% | 90.0% | 0.87M | 426s |

All automated methods accepted **zero** improving edits in these runs. RRSI’s
edit budget rejected 2-3-rule proposals; GEPA’s prompt candidates dropped from
95% to 90%; Meta-Harness candidates dropped to 25-45% and one proposal failed
parsing. The hand-written v2 harness remains `H*_small`.

This is a clean negative result: under the current critic/model/budget, the
optimization bottleneck is proposal quality, not search cost.

## R5. Optimization cost

`experiments/optimization_costs.md`. The two SkillOpt runs cost about
$2.5-2.8 at assumed $0.50/$1.50 per million input/output tokens, or 6-8 minutes
on one B200. A full large-scale evaluation costs comparable or more tokens. The
problem is not the price of small-graph optimization; it is the lack of
validated improvements.

## R6. Routing and decomposition

BF16 B200 run on a controlled disconnected additive set (4 tasks, 2 problems
each, 400 nodes / 5 components; job `1543102`, larger context):

| Mode | Accuracy | Max clique | MIS | MVC | Components |
|---|---:|---:|---:|---:|---:|
| Global B v2 | 25.0% | 0% | 0% | 0% | 100% |
| Routed library | 37.5% | 50% | 0% | 0% | 100% |
| Decomposed per component | **50.0%** | **100%** | 0% | 0% | 100% |

The decomposed path runs one harness per connected component and correctly
aggregates for max clique and connected components. MIS/MVC still fail because
per-component answers are not optimal. The 1-problem CPU pilot showed the same
qualitative pattern (50% global, 50% decomposed; max clique correct under
decomposition).

**Interpretation:** decomposition is a real win when the task is additive and
the per-component harness is reliable. The router must respect decomposability
and safely fall back; a generic global harness cannot exploit the component
structure.

## C tool backend and Meta-Harness

- `CSandbox` compiles model-written C with `gcc` against a minimal
  `graph_io.h`/`graph_io.c`; no NetworkX.
- A 5-problem CPU-Q4 C pilot scored **0/5**. The BF16 B200 run (`1543097`)
  scored **12.0%** on 25 problems (cycle detection 60%, everything else 0-20%).
  The BF16 model can compile some C, but C without NetworkX is dramatically
  harder than Python/NetworkX (compare 96% on `test_small` with tools and
  35.3% text-only).
- `Meta-Harness-lite` lets the proposer edit the prompt, rules, graph view, tool
  backend, and sandbox helper preamble. Its 3-round pilot accepted no edit.

## Overall reading

- The reviewer is right that NetworkX makes the tasks much easier: tool access
  is worth 35-58 points on small/medium graphs and is the difference between a
  0% and 92% large-graph result.
- The harness still matters on top of tools, especially for verification,
  finding diameter witnesses, keeping components/representatives correct, and
  avoiding numeric-node shortcuts.
- Cheap learned proxies are promising; zero-shot rubrics and one-problem proxies
  are not.
- Automated harness development (SkillOpt, GEPA, RRSI, Meta-Harness) currently
  fails to beat the human v2 harness. The next bottleneck is better proposal
  generation/critique, not rollout cost.
- C/Meta-Harness/C-router experiments are implemented; the C pilot is 12% and
  the decomposition router is 50% vs 25% global on the controlled disconnected
  set. These are pilot results, not full-budget claims.
