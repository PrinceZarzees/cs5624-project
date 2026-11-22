# Review-driven experiment design

This document turns each reviewer comment into a controlled experiment. It is
written so the experiments can be run independently on the existing Qwen3-4B
harness setup.

Legend: **[done]** = results already produced in this repository;
**[design]** = specified here, ready to implement/run.

---

## R1. Cheap proxies for large-graph harness generalization (SkillLift-inspired)

**Question:** Can a rubric/ranking proxy predict which small-graph harness will
transfer to large graphs without running full agent rollouts on large graphs?

**Hypothesis:** A dense rubric learned from a small number of oracle rollouts
will rank candidate harnesses in the same order as expensive large-graph
rollouts, at 5-10x lower token cost (SkillLift’s rank-supervision insight).

**Conditions:**

1. Candidate pool: all harness checkpoints from the two optimizer runs
   (`experiments/qwen3_4b/optimizer/candidates/`,
   `experiments/qwen3_4b_opt_a/optimizer/candidates/`), plus hand-written
   variants (A, B, B v2, no-tools, router variants). Aim for 20-40 candidates.
2. Oracle labels: actual `val` accuracy (small graphs) and actual
   `test_medium`/`test_large` accuracy for a subset of candidates (the expensive
   target). We already have `val` labels for candidate edits in the optimizer
   histories.
3. Proxy families:
   - **Static rubric:** judge scores 0-5 on tool use, algorithm choice,
     verification/feasibility, scalability, no node-id shortcuts, output
     format, and task coverage.
   - **Learned/evolved rubric (SkillLift-style):** an outer loop generates
     rubric items and weights from a small set of oracle-ranked pairs; an inner
     loop ranks candidates with the frozen rubric; outer loop re-aligns weights
     by rank correlation after a few oracle rollouts.
   - **Cheap execution proxy:** 1-2 training problems per task (not full val)
     plus trajectory features (`tool_calls`, timeout rate, parsed rate).
4. Baselines: random ranking, prompt length, rule count, and static rubric.

**Metrics:**

- Spearman ρ and Kendall τ between proxy score and actual large-graph accuracy.
- Top-k recall: is the truly best candidate in the proxy’s top 3/5?
- Oracle cost: number of full agent rollouts vs proxy token cost.
- Regret: actual accuracy of the proxy-selected harness minus best actual.
- Calibration: mean absolute error of proxy-predicted accuracy.

**Expected evidence:** evolved rubric ρ > 0.5 against `test_large` accuracy,
top-3 recall ≥ 0.8, with ≤ 20% of the rollout tokens. If the static rubric
already correlates well, the learned rubric should still improve top-k ranking
and cost.

**Implementation:** `src/graphharness/rubric_proxy.py` (design hook), reuse
`Harness`, `run_experiment` for oracle labels, and a judge prompt with the
rubric. Run as a B200 job with `--candidates ... --oracle-split test_large
--oracle-limit-per-task 3`.

---

## R2. Python/NetworkX access and the value of an optimized harness

**Question:** If the agent can call NetworkX, many graph tasks become code
lookups. Does the harness still matter, or does tool access make the task
trivial?

**Hypothesis:** Tool access raises absolute accuracy but reduces the headroom of
prompt/rule optimization; no-tool agents degrade sharply with graph size and
density because the graph text is long and the model must reason symbolically.

**Conditions (2 x 2 + size):**

| Condition | Tools | Graph rendering | Harness |
|---|---|---|---|
| A-text | no | edge list / adjacency | Baseline A |
| A-tools | yes | summary | Baseline A |
| Bv2-text | no | edge list / adjacency | Baseline B v2 |
| Bv2-tools | yes | summary | Baseline B v2 |

Evaluate on full `test_small`, `test_medium`, `test_large`, `test_stress`.
Additional intermediate conditions:

- **Tool oracle:** a fixed tool-call template (`nx.shortest_path`,
  `nx.connected_components`, etc.) with no LLM choice of algorithm.
- **Tool timeout/no-retry:** disable the final-answer turn and error recovery.

**Metrics:** accuracy, parsed/feasible, tokens, latency, timeout rate, tool
calls, failure class (no tool call, bad node ids, timeout, wrong algorithm).
Optimization headroom: `Bv2 - A` in each condition.

**Expected evidence:** tools add the largest gain on `test_large`/`test_stress`
and on sparse/tree graphs; text-only accuracy collapses with context length.
The harness still matters for *verification* and *answer formatting* even with
tools, but algorithm-selection rules matter less.

**Implementation:** add `Harness.allow_tools: bool` and a no-tool rendering path;
the runner already records tool calls and timeouts. Run one B200 job per
condition.

---

## R3. Performance across graph characteristics

**Question:** When/where do optimized harnesses generalize? Which density,
topology, and task-difficulty regimes transfer?

**Status: [done]** `experiments/analysis_by_characteristics.md` and
`report/tables/characteristics.csv` already stratify the full scaling results by
density bin, graph family, and task.

Observed from the current run (Baseline A example):

- “mixed” component graphs: 100%, trees/cycle detection: 62%, BA graphs: 74.7%,
  WS: 82.2%, ER: 85.9%.
- Very sparse graphs (mean ~994 nodes): 77.5%; dense small graphs: 87.7%.

**Next experiment (controlled, not confounded):**

Create `configs/stratified_splits.json` with matched graph sets:

- Density: p ∈ {0.005, 0.01, 0.03, 0.1, 0.3} for ER, fixed n.
- Topology: ER, BA, WS, tree, disconnected mixture.
- Difficulty: task-specific (diameter/clique instances with small vs large
  ground-truth value; shortest path with short vs long hop distance).
- Sizes: 30, 100, 300, 1000, 2000.

**Analysis:** accuracy ~ density × family × task × size, with bootstrap 95%
CIs; logistic regression with harness fixed effects; report the largest
interaction terms. This separates “hard because large” from “hard because
dense/sparse/tree-like.”

**Expected evidence:** most of the apparent size effect is density/topology;
tree/cycle and dense-clique regimes are the main transfer failures.

---

## R4. Additional optimization baselines: GEPA, SkillOpt/EvoSkill

**Question:** Is the negative optimization result specific to our SkillOpt-style
loop, or do other prompt/skill evolution methods also fail to beat the human
harness?

**Conditions (same actor, critic, train/val, budget):**

1. **Baseline A** (no optimization).
2. **Human B / B v2**.
3. **GEPA-style prompt evolution:** reflective mutation of the system prompt
   only; 3 candidate prompts per round; Pareto selection on train/val; no rule
   list.
4. **SkillOpt-style rule evolution:** current optimizer (rule list + prompt),
   one candidate per round.
5. **EvoSkill-style library evolution:** add/remove one skill snippet at a
   time, keep an archive, select by validation and novelty.
6. **RRSI-regularized evolution** (see R7).

**Metrics:** val accuracy, test_small/medium/large/stress accuracy, accepted
edits, token cost, wall time, transfer gap `val - test_large`, edit complexity
(number of rules/characters).

**Expected evidence:** all automated methods either reject all edits or
overfit `val`; GEPA explores more prompt diversity but costs more; the human
v2 remains best or ties. If an automated method wins on val but loses on
`test_large`, that is direct evidence for the review’s overfitting concern.

**Implementation:** `src/graphharness/gepa.py` and
`src/graphharness/evoskill.py` share the optimizer’s candidate evaluation and
logging; run 5-10 rounds each on B200.

---

## R5. Optimization costs

**Status: [done]** `experiments/optimization_costs.md`.

Current measured costs:

| Run | Rounds | Val gain | Tokens | Wall time | Est. cost |
|---|---:|---:|---:|---:|---:|
| from Baseline A | 10 | 0.0 | 4.01M | 486 s | $2.80 |
| from B v2 | 8 | 0.0 | 3.53M | 383 s | $2.47 |

**Next experiment:** add cost per *accepted* edit, cost per +1% test accuracy,
and cumulative-token curves. Compare with:

- One full `test_large` evaluation (from scaling summary tokens).
- One full `test_stress` evaluation.
- A full large-graph rollout budget (100 problems).

**Expected evidence:** small-graph optimization costs single-digit dollars and
minutes, while one full large-scale evaluation costs comparable tokens; the
problem is not optimization cost but lack of accepted improvements. If a future
optimizer does improve, report dollars per transferred point.

---

## R6. Specialized harnesses + agentic router / decomposition

**Question:** Can a router choose a class-specific harness better than one
global harness? Can large graphs be decomposed and routed per subgraph?

**Harness library:**

- `sparse`: BFS/DFS-first, avoid dense matrices, emphasize components/trees.
- `dense`: exact clique/MIS/MVC, adjacency-matrix checks, branch-and-bound.
- `tree`: acyclicity shortcuts (`edges = n - components`), path/cycle logic.
- `large`: 2000-node timeout-safe algorithms, sampling, output truncation.
- `task-*`: task-specific variants for shortest path, diameter, clique.

**Router conditions:**

1. Global B v2 (current H*).
2. Oracle router (best harness per class on test).
3. Feature router: decision tree/logistic regression on `(n, density, degree,
   components, task)` trained on `val` with small graphs.
4. LLM router: critic sees only graph summary + task, picks a harness name.
5. Random router.
6. Decomposed router: for additive tasks (MIS, MVC, max clique, connected
   components), split into connected components or communities; route each
   subgraph; aggregate (union for MIS/MVC/components, best component for
   max clique). Report correctness and token overhead.

**Metrics:** accuracy, tokens, latency, router regret vs oracle, decomposition
overhead, failure from incorrect aggregation.

**Expected evidence:** feature router approaches oracle on sparse trees/dense
cliques; LLM router is competitive but costlier; decomposition helps additive
tasks on disconnected/large graphs but hurts path/diameter tasks (not additive),
so the router must respect task decomposability.

**Implementation:** `src/graphharness/router.py` with `HarnessLibrary`,
`FeatureRouter`, `LLMRouter`, and `DecomposedRunner`; unit tests with a mock
model; B200 evaluation on `test_medium`/`test_large`/`test_stress`.

---

## R7. RRSI-style regularization against overfitting

**Question:** Can regularization prevent the optimizer from learning
small-graph-specific rules and improve transfer to unseen sizes/topologies?

**RRSI-inspired constraints:**

1. **Temporal edit budget:** at most 1 edit in early rounds, 2 later; edits are
   one rule add/remove/rewrite, not a full prompt rewrite.
2. **Critic filter:** reject proposals mentioning specific node ids, numeric
   ranges, benchmark names, or exact answers.
3. **Pruner:** reject edits that are too small (no validation change), too
   expensive (prompt/token increase > 20%), or no longer useful (rule never
   used in failure trajectories).
4. **Multi-view validation:** accept only if the candidate improves or ties on
   (a) original `val`, (b) node-relabeled `val`, (c) a shifted family/density
   slice. RRSI’s rank correlation/regularization idea maps to requiring
   consistent rank improvement across views.
5. **Novelty archive:** penalize repeated proposals; encourage unexplored
   trajectories.

**Conditions:** vanilla optimizer vs RRSI from Baseline A and B v2; 8-10 rounds
each; same actor/critic.

**Metrics:** val gain, test_small/medium/large/stress accuracy, OOD gap
(`test_small - test_stress`), accepted edits, token cost, rule specificity
(percentage of rules with node ids/benchmark names), and rank stability across
perturbed validations.

**Expected evidence:** RRSI accepts fewer edits but has a smaller
in-distribution/OOD gap; vanilla may gain on val and lose on stress. This is the
most direct test of the review’s overfitting concern.

**Implementation:** `src/graphharness/rrs.py` wrapping the existing optimizer
with `EditBudget`, `SpecificityCritic`, `EditPruner`, and `MultiViewSelector`;
log every rejected edit and reason.

---

## Experiment matrix and priority

| # | Experiment | New model calls? | Est. cost | Priority | Evidence |
|---|---|---|---:|---|---|
| R3 | Characteristic stratification | no | 0 | done | where transfer fails |
| R5 | Cost accounting | no | 0 | done | cheap optimization |
| R2 | Tool access ablation | yes | ~$3-5 | high | need for harness |
| R1 | Rubric proxy / SkillLift | yes (judge + few rollouts) | ~$3-5 | high | cheap proxy ranking |
| R7 | RRSI vs vanilla | yes | ~$6-10 | high | overfitting control |
| R4 | GEPA/EvoSkill baselines | yes | ~$8-15 | medium | optimizer comparison |
| R6 | Router/decomposition | yes | ~$5-10 | medium | specialized transfer |

## What is already implemented

- Harness/task/benchmark infrastructure for 22 tasks.
- Full GraphArena (10/10) and GraphQA (12/12) adapters and real benchmark runs.
- Graph-characteristic analysis and cost accounting.
- A validator/optimizer that already rejects regressions and can be extended
  with the RRSI hooks above.

The remaining work is implementation and running of R1, R2, R4, R6, R7; the
designs above are written to reuse the existing runner, optimizer, sandbox, and
job templates.

## Run status log

- R1 static rubric: run on 18 optimizer candidates; Spearman(proxy, val) =
  -0.217, pairwise ranking = 41.8%, top-3 recall = false.
- R1 learned static proxy: leave-one-out linear proxy from prompt/rule features;
  Spearman = 0.553, pairwise = 70.1% (results in
  `experiments/review_rubric_proxy/learned_proxy_results.json`).
- R1 cheap rollout proxy: one `val` problem per task per candidate, running on
  CPU; results will be in
  `experiments/review_rubric_proxy/cheap_proxy_results.json`.
- R2 full GPU tool-access ablation: Slurm job `1541927`, queued.
- R2 reduced CPU tool-access ablation: running in
  `experiments/review_tool_access_cpu/`.
- R7 RRSI and R4 GEPA: in Slurm job `1542235`, queued behind `1541927`.
- R6 routed/decomposed: inside `1542235`.

## Updated run status

- R1: static rubric ρ=-0.217; learned static proxy ρ=0.553; cheap 1/task proxy
  ρ=-0.172. See `experiments/REVIEW_RESULTS.md`.
- R2: full GPU tool ablation completed: Bv2 tools 96.0/93.3/92.0% vs text
  35.3/15.3/0.0% on small/medium/large. Reduced CPU replication agrees.
- R4: GEPA, RRSI, Meta-Harness pilots accepted zero improving edits.
- R6: B200 disconnected-additive run: global 25%, routed 37.5%, decomposed
  50%; decomposed max clique 100%.
- C backend: implemented and unit-tested; CPU-Q4 5-problem pilot 0/5; BF16 B200
  pilot 12.0% on 25 problems.
- R5/R3: complete.
