# Harness Optimization for Graph Reasoning: Methods and Results

**Project:** CS 5624 — *Harness Optimization for Graph Reasoning: Learn Small, Run Big*  
**Authors:** Bikash Kumar Paul, Abdullah Al Noman, Samin Rahman Khan, Md Zarzees Uddin Shah Chowdhury  
**Report date:** 2026-11-21  
**Branch:** `saminrk-dev`  
**Status:** technical/experimental work complete; paper prose is human-written

This report consolidates all experiments run in this repository: the original
proposal experiments, the real GraphArena/GraphQA evaluation, and the
reviewer-driven experiments (tool access, proxies, extra optimizers, costs,
routing/decomposition, RRSI, C backend, and Meta-Harness).

---

# 1. Executive summary

- The primary model is **Qwen3-4B-Instruct-2507** (Apache-2.0, 4.02B), served
  with vLLM in BF16 on one NVIDIA B200.
- On full synthetic `test_small` (150 problems), the main baselines are:
  **Baseline A 87.3%, Baseline B 90.0%, Baseline B v2 93.3%**.
  A second full run of the 14-rule v2 reached **94.0%**.
- Full scaling (small/medium/large/stress): Baseline A
  **87.3/80.7/81.0/62.0%**, Baseline B **94.0/90.7/86.0/60.0%**, v2
  **94.0/90.0/85.0/74.0%**. The v2 harness is the best at 2,000 nodes.
- Automated harness evolution (SkillOpt-style, GEPA, RRSI, Meta-Harness)
  accepted **zero improving edits** in these pilots. The human-designed v2
  harness remains `H*_small`.
- Real benchmarks (20 problems/task): GraphArena easy **74.5%**, hard
  **61.0%**, GraphQA test **88.3%**.
- Tool access dominates: with Python/NetworkX, v2 reaches **96/93.3/92%** on
  small/medium/large; without tools it drops to **35.3/15.3/0%**. A C/gcc
  backend without NetworkX reaches only **12%** on 25 small problems.
- A learned static proxy ranks optimizer candidates reasonably
  (Spearman **0.553**), while a zero-shot rubric does not (**−0.217**).
- On a controlled disconnected additive set, decomposition improves accuracy
  from **25% global → 37.5% routed → 50% decomposed**, with max clique
  **0% → 50% → 100%**.
- Test suite: **174 passed, 0 skipped**. All converted synthetic,
  GraphArena, and GraphQA ground truths verify.

---

# 2. Methods

## 2.1 Research question

Can an LLM agent harness optimized only on small graphs (`|V| ∈ [4, 50]`)
transfer to much larger graphs (`|V| ∈ [100, 2000]`), or does it overfit to
small-scale structure? We measure the generalization gap

```text
Δ_gen = Score(H* | small) − Score(H* | large)
```

and additionally study tool access, graph characteristics, cheaper proxies,
optimization cost, specialized routing, and regularization.

## 2.2 Models and hardware

| Role | Model / environment |
|---|---|
| Primary actor | `Qwen/Qwen3-4B-Instruct-2507`, BF16, vLLM 0.34.0 |
| Primary hardware | 1× NVIDIA B200 (183 GB), CUDA 13.1 toolkit |
| Fallback | `Qwen/Qwen2.5-7B-Instruct`, Q4_K_M GGUF, llama-cpp-python |
| CPU primary | Qwen3-4B Q4_K_M GGUF, llama-cpp-python |
| Judge (proxies) | Qwen3-4B Q4_K_M |
| Proposed but not used | DeepSeek-V4.1-Flash, GLM (the proposal's other candidates) |

Model snapshots and environments are recorded in
`experiments/REPRODUCIBILITY.md`. Weights live under
`/projects/psgroup/saminrk/gh-qwen/`.

## 2.3 Tasks

The authoritative package implements **22 tasks**:

- Proposal core: shortest path, connected components, cycle detection, max
  clique, Hamiltonian path, diameter, reachability.
- GraphArena additions: common neighbors, maximum independent set, minimum
  vertex cover, graph edit distance, maximum common subgraph, traveling
  salesman.
- GraphQA additions: connected/disconnected nodes, node count, node degree,
  edge count, edge existence, triangle counting, node classification, maximum
  flow.

Coverage is **GraphArena 10/10** and **GraphQA 12/12**; see
`experiments/BENCHMARK_COVERAGE.md`.

## 2.4 Data

### Synthetic splits

`configs/data_splits.json` (seed 5624):

| Split | Nodes | Problems/task | Tasks | Total |
|---|---|---:|---|---:|
| train | 4–50 | 60 | 5 | 300 |
| val | 4–50 | 30 | 5 | 150 |
| test_small | 4–50 | 30 | 5 | 150 |
| test_medium | 100–300 | 30 | 5 | 150 |
| test_large | 500–1000 | 20 | 5 | 100 |
| test_stress | 2000 | 10 | 5 | 50 |

Core five tasks: shortest path, connected components, cycle detection,
diameter, max clique. Graph families: Erdős–Rényi, Barabási–Albert,
Watts–Strogatz; special task shapes add trees, disconnected components, planted
cliques, etc. Node IDs are randomized to prevent label shortcuts. A fresh build
is byte-identical and matches `data/manifest.json`.

### GraphArena

All 10 tasks were downloaded and converted:

```text
data/grapharena_smoke/   40 records (2/task)
data/grapharena_eval/    400 records (20/task, easy+hard)
```

GED/MCS use two-graph records; TSP uses weighted edges. Hard GED/MCS/TSP
instances outside exact small-instance limits retain the source answer/path and
mark `meta["ground_truth_source"]`.

### GraphQA

All 12 tasks were downloaded and converted:

```text
data/graphqa_smoke/    66 records
data/graphqa_eval/     660 records (20/task across published splits)
```

Node classification and maximum flow retain the source answer because the
question text does not expose enough information to recompute the generator's
semantics; this is marked in `meta`.

All 1,166 converted records' stored truths score correctly.

## 2.5 Harness representation

A harness is a JSON object, not a model weight update:

```python
Harness(
    name="baseline_b_v2",
    system_prompt="...",
    rules=["...", "..."],
    graph_view="summary",       # summary | edge_list | adjacency_list
    allow_tools=True,
    tool_backend="python",      # python | c
    tool_preamble="",           # optional helper code injected into the sandbox
    max_steps=8,
    tool_timeout_s=30,
    memory_mb=2048,
    temperature=0,
)
```

The model can call `run_python`/`run_c` and must finish with an answer in the
task-requested format (`<answer>...</answer>`). For node-list tasks this is a
bracketed list; scalar/string/mapping/two-list tasks use the corresponding
format.

## 2.6 Sandboxes and tool backends

### Python/NetworkX backend

`src/graphharness/sandbox.py` runs model-written Python in a fresh process per
call in a temporary directory. `G`, `params`, and `nodes` are injected; the
record's answer is never written. Limits: wall-clock timeout, CPU seconds,
address space, output size, fresh session/process group. A bare final
expression is echoed notebook-style. Files written in the working directory
persist across calls.

### C/gcc backend

`src/graphharness/csandbox.py` compiles model-written C with `gcc` and runs the
binary under the same limits. It provides:

- `graph_io.h` / `graph_io.c`
- `Graph { n, m, max_id, eu, ev, ew }`
- `load_graph()`, `get_param("source")`, `free_graph()`
- `graph.txt` (`n m`, then `u v weight`) and `params.txt`

No NetworkX or Python is available in the C condition. Node IDs must be
integers for this backend.

### No-tools mode

`Harness(allow_tools=False, graph_view="adjacency_list")` gives the model only
the rendered graph and question. Code/tool calls are rejected. This is the
tool-access ablation.

## 2.7 Agent loop

1. Build system prompt (prompt + rules + tool protocol + optional helper
   preamble).
2. Build user message (graph summary/rendering + question + answer-format
   instruction).
3. Call the model.
4. Parse the last meaningful action: JSON tool call, fenced code block, or
   final answer.
5. If code: execute in the selected sandbox; append observation.
6. Code wins over a same-message guessed answer: code is run first.
7. After the tool-step budget, one final-answer turn is allowed (no code).
8. Return the final response; `score_response` parses and checks it.

## 2.8 Harnesses

| Harness | Description |
|---|---|
| `baseline_a` | minimal prompt + tools, no strategy |
| `baseline_b` | original human expert graph-algorithm rules |
| `baseline_b_v2` | B plus explicit component, diameter-witness, max-clique, print, stop-after-success, no-numeric-filtering and generator rules |
| `baseline_*_text` | no-tools adjacency-list variants (ablation) |
| `baseline_*_c` | C/gcc tool backend (ablation) |
| `sparse/dense/tree/large` | router library variants |

The current frozen harness is **`baseline_b_v2`**, 14 rules. The 13-rule
version was used in the main full baseline run; the 14th rule
(materialize NetworkX generators) was added after diagnosing repeated diameter
failures.

## 2.9 Optimizers

### SkillOpt-style rule/prompt evolution

`src/graphharness/optimizer.py`:
sample small training batch → run harness → collect failures → critic proposes
JSON prompt/rules → validate → evaluate on small-graph validation → accept if
accuracy improves (or ties with shorter prompt). Saves history, candidates,
checkpoints, best harness, tokens. `--rrs` switches to RRSI-lite.

### GEPA-style prompt evolution

`src/graphharness/gepa.py`: multiple revised system prompts per round, rules
held fixed, best candidate on validation accepted.

### RRSI-lite

`optimizer.py --rrs`:
- one rule edit per round;
- reject system-prompt rewrites;
- reject rules with numeric node IDs/ranges;
- reject duplicates, no-op edits, or large prompt growth;
- evaluate two validation views and require consistent improvement.

### Meta-Harness-lite

`src/graphharness/meta_harness.py`: proposer may edit the system prompt, rules,
graph view, tool backend, and `tool_preamble` helper code. Preamble is injected
into the Python sandbox. Forbidden APIs and numeric node shortcuts are rejected.

## 2.10 Evaluation protocol

- Optimizer training: synthetic `train`; acceptance: synthetic `val`.
- Frozen evaluation: full `test_small`, `test_medium`, `test_large`,
  `test_stress`.
- Metrics: parsed, feasible, accuracy, mean nodes, prompt/completion tokens,
  latency, errors, tool calls.
- Generalization gap: `test_small − test_large`.
- Real benchmarks: GraphArena easy/hard and GraphQA test, 20 problems/task.
- Rule ablations: full split versus one-rule-removed variants.
- All candidate harnesses use temperature 0 unless stated; vLLM generation is
  not perfectly deterministic, so repeat runs vary by a few points.

## 2.11 Review-driven experiments

- **Tool access (R2):** Baseline A/B v2 × tools/no-tools on small/medium/large.
- **Graph characteristics (R3):** accuracy by density, family, task.
- **Proxy ranking (R1):** static rubric, learned static proxy, cheap rollout
  proxy vs actual validation accuracy.
- **Optimizer baselines (R4/R7):** GEPA, RRSI, Meta-Harness.
- **Cost accounting (R5):** tokens, wall time, estimated dollars.
- **Router/decomposition (R6):** feature router and per-component execution on
  additive tasks.
- **C backend:** language/library ablation with gcc and no NetworkX.

---

# 3. Results

## 3.1 Main full `test_small` baselines

Main BF16 B200 run, 150 problems:

| Harness | Accuracy | Parsed | Feasible | Tokens |
|---|---:|---:|---:|---:|
| Baseline A | 87.3% | 99.3% | 90.0% | 595,014 |
| Baseline B | 90.0% | 100% | 92.0% | 534,810 |
| Baseline B v2 (13 rules) | 93.3% | 100% | 98.0% | 963,818 |
| Baseline B v2 (14 rules, second run) | 94.0% | 100% | 96.7% | 939,899 |

Per-task (main 13-rule v2 run):

| Task | Baseline A | Baseline B | Baseline B v2 |
|---|---:|---:|---:|
| shortest path | 83.3% | 93.3% | 100% |
| connected components | 100% | 76.7% | 100% |
| cycle detection | 73.3% | 100% | 96.7% |
| diameter | 86.7% | 100% | 86.7% |
| max clique | 93.3% | 80.0% | 83.3% |

The 14-rule v2 full `test_small` run gives 94.0% overall.

## 3.2 Full scaling

Full split sizes: 150/150/100/50.

| Harness | small | medium | large | stress | Δ_gen (small−large) |
|---|---:|---:|---:|---:|---:|
| Baseline A | 87.3% | 80.7% | 81.0% | 62.0% | +6.3% |
| Baseline B | 94.0% | 90.7% | 86.0% | 60.0% | +8.0% |
| Baseline B v2 | 94.0% | 90.0% | 85.0% | 74.0% | +9.0% |

Per-task, Baseline B v2:

| Split | shortest path | components | cycles | diameter | max clique |
|---|---:|---:|---:|---:|---:|
| test_small | 100% | 100% | 100% | 90.0% | 80.0% |
| test_medium | 100% | 100% | 93.3% | 93.3% | 63.3% |
| test_large | 100% | 100% | 95.0% | 90.0% | 40.0% |
| test_stress | 80.0% | 100% | 90.0% | 20.0% | 80.0% |

Per-task tables for A and B are in
`experiments/qwen3_4b_scaling_full/scaling/scaling_by_task.md`.

Key observations: components are consistently strong; cycle detection is
90-100% for v2; diameter collapses at 2,000 nodes; max clique is the main
large-graph weakness.

## 3.3 Rule ablations

### Full `test_large` (100 problems), v2 full = 85.0%

`delta = full − ablated`; positive means the rule helped.

| Removed rule | Ablated | delta |
|---|---:|---:|
| 0 use `G` directly | 89.0% | −4.0 |
| 1 prefer exact routines | 88.0% | −3.0 |
| 2 clique policy | 88.0% | −3.0 |
| 3 feasibility checks | 86.0% | −1.0 |
| 4 keep output short | 89.0% | −4.0 |
| 5 change failing code | 86.0% | −1.0 |
| 6 answer format | 83.0% | +2.0 |
| 7 CC print/representatives | 86.0% | −1.0 |
| 8 diameter witness | 90.0% | −5.0 |
| 9 explicit print | 90.0% | −5.0 |
| 10 `nx.max_weight_clique` | 89.0% | −4.0 |
| 11 stop after success | 89.0% | −4.0 |
| 12 no numeric node filtering | 87.0% | −2.0 |
| 13 materialize generators | 88.0% | −3.0 |

Most rules help by 1–5 points. Removing the answer-format rule shows +2, within
noise of a 100-problem split.

### Full `test_stress` (50 problems), v2 full = 76.0%

| Removed rule | Ablated | delta |
|---|---:|---:|
| 0 | 76.0% | 0.0 |
| 1 | 66.0% | +10.0 |
| 2 | 70.0% | +6.0 |
| 3 | 70.0% | +6.0 |
| 4 | 76.0% | 0.0 |
| 5 | 76.0% | 0.0 |
| 6 | 74.0% | +2.0 |
| 7 | 76.0% | 0.0 |
| 8 | 72.0% | +4.0 |
| 9 | 66.0% | +10.0 |
| 10 | 68.0% | +8.0 |
| 11 | 72.0% | +4.0 |
| 12 | 74.0% | +2.0 |
| 13 | 70.0% | +6.0 |

At 2,000 nodes, exact-routine, explicit-print, and max-clique rules matter most;
the generator rule adds 6 points.

### Full `val` (150 problems), v2 full = 95.3%

Only rule 8 removal improved val, by 0.7% (one problem, 96.0%). The same rule is
worth +5 on `test_large` and +4 on stress. Dropping it would be a
small-to-large overfitting mistake; v2 remains frozen.

## 3.4 Optimizer baselines and costs

| Optimizer | Start | Rounds | Initial val | Best val | Tokens | Time | Est. $ |
|---|---|---:|---:|---:|---:|---:|---:|
| SkillOpt-style | A | 10 | 87.5% | 87.5% | 4.01M | 486 s | $2.80 |
| SkillOpt-style | B v2 | 8 | 97.5% | 97.5% | 3.53M | 383 s | $2.47 |
| GEPA-style | A | 3 | 95.0% | 95.0% | 1.19M | 597 s | — |
| RRSI-lite (rule-only) | A | 3 | 90.0% | 90.0% | 0.51M | 426 s | — |
| RRSI-lite (rule-only) | B v2 | 3 | 100% | 100% | 0.81M | 405 s | — |
| Meta-Harness-lite | A | 3 | 90.0% | 90.0% | 0.87M | 426 s | — |

All methods accepted **zero improving edits**. RRSI rejected 2–3-rule
proposals; GEPA’s prompt candidates fell to 90%; Meta-Harness candidates fell
to 25–45% and one proposal failed parsing. The validation gate prevented
regressions, but proposal quality is the bottleneck.

Best-harness evaluation on 25 `test_small` problems:

| Run | Harness result |
|---|---:|
| RRSI from A | 92% (baseline A unchanged) |
| RRSI from B v2 | 96% (baseline B v2 unchanged) |
| GEPA from A | 96% (baseline A unchanged) |
| Meta-Harness from A | 96% (baseline A unchanged) |

Costs are cheap: single-digit dollars and minutes. A full large-scale
evaluation costs comparable or more tokens, so optimization cost is not the
limiting factor.

## 3.5 Real GraphArena and GraphQA evaluation

Qwen3-4B BF16, `baseline_b_v2`, 20 problems/task.

| Benchmark | Split | n | Accuracy | Parsed | Feasible |
|---|---|---:|---:|---:|---:|
| GraphArena | easy | 200 | **74.5%** | 95.0% | 84.0% |
| GraphArena | hard | 200 | **61.0%** | 97.5% | 83.5% |
| GraphQA | test | 240 | **88.3%** | 100% | 99.2% |

GraphArena per-task:

| Task | easy | hard |
|---|---:|---:|
| shortest path | 100% | 100% |
| connected components | 100% | 100% |
| common neighbors | 95% | 100% |
| diameter | 90% | 90% |
| max clique | 95% | 80% |
| MIS | 95% | 20% |
| MVC | 85% | 70% |
| GED | 10% | 0% |
| MCS | 55% | 50% |
| TSP | 20% | 0% |

GraphQA per-task:

| Task | Accuracy |
|---|---:|
| connected nodes | 100% |
| disconnected nodes | 100% |
| node count | 100% |
| node degree | 100% |
| edge count | 100% |
| edge existence | 100% |
| shortest path | 95% |
| reachability | 95% |
| cycle detection | 90% |
| triangle counting | 90% |
| node classification | 75% |
| maximum flow | 15% |

Real-benchmark weaknesses are GED, MCS, TSP, hard MIS, and maximum flow.
GraphQA maximum-flow answers are not recoverable from the displayed unweighted
graph; the source answer is retained and the model does not have the hidden
capacity information.

## 3.6 Cross-model comparison

Full `test_small`:

| Model | Precision | Hardware | Harness | Accuracy |
|---|---|---|---|---:|
| Qwen3-4B | BF16 | 1× B200 | Baseline A | 87.3% |
| Qwen3-4B | BF16 | 1× B200 | Baseline B | 90.0% |
| Qwen3-4B | BF16 | 1× B200 | B v2 (13) | 93.3% |
| Qwen3-4B | BF16 | 1× B200 | B v2 (14) | 94.0% |
| Qwen3-4B | Q4_K_M | 96 CPU | Baseline A | 84.7% |
| Qwen3-4B | Q4_K_M | 96 CPU | B v2 (13) | 93.3% |
| Qwen2.5-7B | Q4_K_M | 48/96 CPU | B v2 (13) | 77.3% |
| Qwen2.5-7B | Q4_K_M | 96 CPU | B v2 (14) | 76.0% |

Qwen3-4B is both smaller and more accurate than the 7B fallback; it remains the
primary model.

## 3.7 Tool access (review R2)

Full BF16 B200:

| Condition | small | medium | large |
|---|---:|---:|---:|
| A + tools | 91.3% | 88.7% | 80.0% |
| A no tools | 43.3% | 16.0% | 0.0% |
| B v2 + tools | 96.0% | 93.3% | 92.0% |
| B v2 no tools | 35.3% | 15.3% | 0.0% |

No-tools large runs hit the 8k context limit on **25/25** problems. Reduced
CPU-Q4 replication (5/task small, 2/task medium):

| Condition | small | medium |
|---|---:|---:|
| A + tools | 76% | 80% |
| A no tools | 28% | 30% |
| B v2 + tools | 92% | 100% |
| B v2 no tools | 24% | 20% |

**Conclusion:** NetworkX/tool access is the dominant factor. Harness rules still
matter on top of tools (B v2 is 4–12 points above A), but without tools the tasks
collapse at scale.

## 3.8 Graph characteristics (review R3)

Baseline A full scaling by density:

| Density bin | n | mean nodes | accuracy |
|---|---:|---:|---:|
| very sparse (<.01) | 182 | 994 | 77.5% |
| sparse (.01–.03) | 100 | 197 | 78.0% |
| medium (.03–.10) | 74 | 59 | 86.5% |
| dense (.10–.30) | 73 | 24 | 87.7% |
| very dense (>.30) | 21 | 8 | 81.0% |

By family:

| Family | n | mean nodes | accuracy |
|---|---:|---:|---:|
| BA | 95 | 477 | 74.7% |
| ER | 85 | 494 | 85.9% |
| mixed components | 90 | 442 | 100% |
| tree | 90 | 459 | 62.2% |
| WS | 90 | 426 | 82.2% |

Task-level: components 100%, shortest path 84.4%, diameter 77.8%, max clique
80.0%, cycle detection 62.2%. Most of the apparent “size” effect is topology
and task difficulty, especially trees/cycles and sparse BA graphs.

## 3.9 Proxy ranking (review R1)

18 optimizer candidates with known validation accuracy:

| Proxy | Spearman | Pairwise ranking | Best in top-3 |
|---|---:|---:|---|
| Static LLM rubric | **−0.217** | 41.8% | no |
| Learned static proxy (LOO) | **0.553** | **70.1%** | — |
| Cheap rollout, 1 problem/task | −0.172 | 58.2% | no |

The static rubric gives almost every candidate 95–100, so it cannot rank. A
learned proxy over prompt/rule features is useful at zero oracle cost. One
problem per task is too noisy; cheap proxies need more samples or better
aggregation.

## 3.10 Router and decomposition (review R6)

Controlled disconnected additive set: 400 nodes, 5 components, 4 tasks, 2
problems/task, BF16 B200, 16k context.

| Mode | Accuracy | Max clique | MIS | MVC | Components |
|---|---:|---:|---:|---:|---:|
| Global B v2 | 25.0% | 0% | 0% | 0% | 100% |
| Routed library | 37.5% | 50% | 0% | 0% | 100% |
| Decomposed per component | **50.0%** | **100%** | 0% | 0% | 100% |

Decomposition correctly solves max clique and connected components per
component. MIS/MVC fail because per-component answers are not optimal; safe
fallback is required. A 1-problem CPU pilot agreed qualitatively.

## 3.11 C backend

| Condition | Problems | Accuracy |
|---|---:|---:|
| Qwen3-4B BF16, C/gcc, B v2 | 25 | **12.0%** |
| Qwen3-4B Q4 CPU, C/gcc | 5 | 0.0% |

C cycle detection reached 60%, but shortest path, components, diameter and max
clique were 0–20%. The model often wrote uncompilable C, placeholder loops, or
non-answer text. Compare: Python/NetworkX v2 is 96% on small and 92% on large.
Removing the graph library is a much harder intervention than changing the
prompt.

## 3.12 Meta-Harness

`meta_harness.py` evolves prompt, rules, graph view, tool backend, and helper
preamble. Three rounds from Baseline A: 90% → 90%, zero accepted edits, 0.87M
tokens, 426 s. Candidates fell to 25–45% and one failed parsing. Code-level
evolution is implemented but did not beat the human harness in this pilot.

---

# 4. Discussion

## 4.1 What the results say

- **Tool access is the largest effect.** Python/NetworkX turns many tasks into
  library calls; without it, large-graph accuracy collapses. C/gcc without a
  graph library is much harder still.
- **Harness rules still matter on top of tools.** v2 beats A/B on component
  handling, diameter witnesses, cycle checks, and answer formatting; the
  largest gains are on the 2,000-node stress split.
- **Large-graph failures are task-specific.** Diameter and max clique degrade
  most; components are robust; cycle detection is strong on sparse tree-like
  graphs.
- **Automated evolution is proposal-limited.** SkillOpt, GEPA, RRSI and
  Meta-Harness all rejected or failed to improve; better critics/proposers, not
  cheaper rollouts, are the next bottleneck.
- **Cheap proxies need learning.** A zero-shot rubric cannot rank; a learned
  static proxy reaches ρ=0.55. SkillLift’s rank-supervised rubric is the right
  direction.
- **Decomposition helps additive tasks.** On disconnected graphs, routing per
  component doubled overall accuracy and solved max clique; it is not valid for
  path/diameter tasks and must fall back safely.
- **Optimization is cheap.** 0.5–4M tokens and 6–10 minutes per run; the
  problem is not the price of small-graph search.

## 4.2 Limitations

- Synthetic tasks dominate the controlled scaling study; GraphArena hard
  graphs are ≤50 nodes and GraphQA is mostly small.
- `test_small` was used for the initial human failure analysis that produced
  B v2, so v2’s `test_small` score is not a clean held-out estimate. B is the
  clean human baseline; medium/large/stress are more informative for v2.
- vLLM generation is not perfectly deterministic; repeated full runs differ by
  up to ~4 points (e.g., B small 90% vs 94%).
- The static rubric judge is Qwen3-4B Q4; a stronger judge may change R1.
- Cheap proxy used only one problem per task; its negative result is partly
  sample noise.
- C backend requires integer node IDs and is a pilot, not a full C harness
  implementation.
- GraphQA maximum flow and node classification use source answers because the
  question text does not contain enough information to recompute the generator
  semantics.
- RRSI/GEPA/Meta pilots use a 4B critic; a stronger critic may change the
  optimization conclusion.

## 4.3 Recommended next steps

1. Run the learned-rubric proxy with a larger candidate pool and multiple
   oracle rollouts; compare top-k selection cost.
2. Use a stronger critic (7B/14B or a frontier API) for GEPA/RRSI/Meta and
   compare proposal quality.
3. Run C backend with a model explicitly trained for C and a small C algorithm
   library, separating language from library effects.
4. Scale the router/decomposition experiment to more additive graphs and a
   learned router trained on small graphs.
5. Expand GraphArena/GraphQA evaluation beyond 20/task, especially GED/MCS/TSP.

---

# 5. Reproducibility

## 5.1 Environment

```text
Primary: Python 3.12, vLLM 0.34.0, torch 2.14.0+cu130, CUDA 13.1
CPU:     Python 3.12, llama-cpp-python 0.3.37
Client:  Python 3.13 with graphharness 0.2.0
```

Model snapshots:

| Model | Snapshot |
|---|---|
| Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Qwen2.5-7B-Instruct Q4 GGUF | `bartowski`, `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031` |
| Qwen3-4B Q4 GGUF | `unsloth`, `a06e946bb6b655725eafa393f4a9745d460374c9` |

## 5.2 Commands

```bash
python -m graphharness.build_data
python -m graphharness.verify_data
python -m graphharness.graphqa_adapter
python -m graphharness.grapharena_adapter --download
pytest -q
```

Experiment launch scripts:

```text
scripts/run_qwen_pipeline.sbatch          baselines, optimizer, scaling
scripts/run_harness.sbatch                one harness/split
scripts/run_optimizer.sbatch              optimizer
scripts/run_scaling.sbatch                scaling
scripts/serve_vllm.sbatch                 GPU vLLM server
scripts/serve_llama_cpp_cpu.sbatch        CPU llama.cpp server
scripts/run_review_tool_access.sbatch     tool-access ablation
scripts/run_review_optimizers.sbatch      GEPA/RRSI/router
scripts/run_review_rrsi2.sbatch           RRSI rule-only rerun
scripts/run_meta_harness.sbatch           Meta-Harness
scripts/run_review_c.sbatch               C backend
scripts/run_review_router.sbatch          router/decomposition
```

## 5.3 Jobs

| Job | Purpose |
|---|---|
| `1537011` | full Qwen3-4B baselines, optimizer, pilot scaling |
| `1537830` | optimizer from Baseline A |
| `1538020` | full `test_large` ablation |
| `1538032` | full scaling |
| `1538186` | full `test_stress` ablation |
| `1537117` | Qwen2.5-7B CPU full test_small |
| `1538817` | full `val` ablation |
| `1540814` | real GraphArena/GraphQA evaluation |
| `1541927` | tool-access ablation |
| `1542235` | GEPA/RRSI/Meta/router pilots |
| `1542947` | RRSI rule-only rerun |
| `1542951` | Meta-Harness |
| `1543097` | C backend |
| `1543102` | 16k-context router/decomposition |

## 5.4 Tests and integrity

```text
174 passed, 0 skipped
```

`python -m graphharness.verify_data` passes all six synthetic splits and the
manifest checksum. All real benchmark converted records score their stored
truths correctly (GraphArena 400, GraphQA 660, plus smoke sets).

## 5.5 Artifact index

```text
src/graphharness/                     authoritative implementation
  agent.py, sandbox.py, csandbox.py   harness, Python/C execution
  models.py, runner.py                model clients and runs
  optimizer.py, gepa.py, meta_harness.py, rrs.py hooks
  router.py, rubric_proxy.py          review experiments
  characteristics.py, cost_analysis.py
experiments/
  RESULTS.md                          consolidated original results
  REVIEW_RESULTS.md                   reviewer experiment results
  REVIEW_EXPERIMENTS.md               review experiment designs
  BENCHMARK_COVERAGE.md               10/10 GraphArena, 12/12 GraphQA
  REPRODUCIBILITY.md                  snapshots, jobs, data inventory
  analysis_by_characteristics.md      density/topology/task tables
  optimization_costs.md               token/time/cost curves
  qwen3_4b/                           main GPU baselines/optimizer/scaling
  qwen3_4b_opt_a/                     optimizer from A
  qwen3_4b_scaling_full/              full scaling + per-task tables
  qwen3_4b_ablation_full/             test_large ablation
  qwen3_4b_ablation_stress/           test_stress ablation
  qwen3_4b_ablation_val/              val ablation / H* selection
  qwen3_4b_real_bench/                real GraphArena/GraphQA
  qwen3_4b_cpu/                       4B CPU fallback
  qwen2_5_7b_cpu/, qwen2_5_7b_cpu_pilot/
  review_tool_access/, review_tool_access_cpu/
  review_rubric_proxy/
  review_optimizers/, review_optimizers2/
  review_meta_harness/
  review_router/, review_router_cpu/
  review_c_backend/
report/
  main.tex, SLIDES.md, README.md
  figures/, tables/
data/
  manifest.json, train/val/test splits (gitignored generated JSONL)
  grapharena_eval/, graphqa_eval/ (gitignored)
```

## 5.6 Large-file policy

Large generated artifacts are ignored by Git:

```text
data/grapharena_*/
data/graphqa_*/
experiments/**/trajectories/
experiments/**/predictions.jsonl
experiments/**/scores.jsonl
experiments/**/*.sqlite
experiments/**/sandbox/
```

The raw GraphArena zip and model weights remain on shared storage for
reproduction; they are not committed.
