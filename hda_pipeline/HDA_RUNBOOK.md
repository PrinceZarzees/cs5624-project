# HDA_RUNBOOK: Harness Developing Agent experiments (phase 1, polynomial-time tasks)

**Audience:** a coding agent with shell access to VT ARC and the `graphharness` repository.
**Goal:** run the new Harness Developing Agent (HDA) experiments and write them into the final report with a methodology section a reader can follow without any other document.
**Date:** 2026-11-21. **Decided by Samin:** C++17 with the standard library only; polynomial-time tasks first (NP-hard tasks later); k = 10 examples per iteration, fixed (k sweep later); the HDA may revise its problem statement every iteration; log statements, answers and reasoning traces every iteration; prompt-level and automatic anti-memorization.
**This runbook is separate from `AGENT_RUNBOOK.md`.** Do not edit that file or remove any existing experiment. The old Python/NetworkX results stay in the report as the library-assisted reference.

---

## 0. Hard rules

1. **Branch:** do not create a new branch. Commit and run everything on the existing `saminrk-experiments` branch (Samin, 2026-11-21). Never force-push. Push with `git push -u origin saminrk-experiments`.
2. **Do not touch existing results.** Everything under `experiments/` that already exists is frozen. New outputs go to `experiments/hda/` (runs, summaries) and `report/tables/hda/`, `report/figures/hda/`.
3. **Report edits are drafts.** The course penalizes AI-written text and Samin rewrites all prose. Wrap every block of text you write in `report/main.tex` in
   ```latex
   % >>> AI-DRAFT (HDA-<step>, <date>): rewrite before submission
   ...
   % <<< AI-DRAFT
   ```
   Keep drafted prose short and factual. Mechanical edits (`\input` of a table, figure includes) need no marker but go in `report/EDIT_LOG.md`. Commit report edits separately with prefix `report:`.
4. **No test-set tuning.** Prompts, step budget, iteration count, time limits and the policy checker are frozen before any `test_*` split is scored. The HDA never sees validation or test graphs, only aggregate validation feedback.
5. **Negative results are results.** If the HDA fails a task at scale, record it. Do not loosen time limits after seeing test results.
6. **Ledger:** every sbatch job goes in `experiments/hda/JOBS.md` (job ID, script, exported variables, purpose, start/end, status, output dir).
7. **Ask the humans first** before downloading a model over ~70 GB that is not already on ARC, before using any paid API, or if the P0+P1 compute exceeds ~40 GPU-hours.
8. **Markers:** `[VERIFY]` means a name on ARC (partition, path, flag) that you must check before use. `[CREATE]` means something you write.

---

## 1. What the experiment is (read this before running anything)

A **Harness Developing Agent (HDA)** is an LLM that writes a program instead of answering one instance. For one graph task, it gets small I/O examples and must produce a single C++17 file that reads any graph from stdin and prints the answer. The C++ program is the harness we evaluate. It is developed only on small graphs (4 to 50 nodes) and then tested on held-out graphs up to 100,000 nodes. This replaces "the agent calls NetworkX" (too easy) with "the agent must implement the algorithm itself", and it tests learn small, run big directly: a correct but slow algorithm passes validation and fails at scale.

### 1.1 Agent input (exactly what the model sees)

Defined in `hda/prompts.py`; the report must summarize it and the appendix should quote it.

| Part | Content |
|---|---|
| System prompt | Role; C++17 and std library only (no Boost, no graph libraries, no file I/O, threads or system calls); hidden graphs are up to 1000x larger with permuted node IDs; explicit ban on memorizing or hardcoding example answers, node IDs, sizes or lookup tables, and a statement that such programs are rejected and relabeling-checked; limits (1 to 5 s CPU, 1 GiB, 8 MiB stack); input format; the graph-loading template (`templates/graph_template.cpp`); the tool protocol. |
| Problem statement | **Given** condition: a one-paragraph statement (`hda/tasks.py`). **Hidden** condition: no statement; instead an instruction to list at least three candidate problems, argue for and against each against every example (by hand or with test code), keep the simplest consistent one, and write it in `<statement>`; it may revise the statement in any later step or iteration. |
| Examples | k = 10 **new** examples per iteration, drawn without replacement from the `examples` split (4 to 50 nodes, seen families). The prompt shows the current iteration's 10; the `test` tool runs all examples seen so far (10, 20, ... 50). |
| Iteration context | Iteration number, aggregate validation feedback for the last submission, the best program so far with its validation accuracy, and the HDA's current statement. |

**Input format** (every graph, examples and tests alike):
```
n m directed weighted        # directed, weighted in {0,1}
u v [w]                      # m lines, nodes 0..n-1, no self-loops, no duplicates
k q1 ... qk                  # query arguments (k = 0 or 2 in phase 1)
```
Note for the report: even in the hidden condition the header reveals whether the graph is directed or weighted and how many query nodes there are. This narrows the task family but does not identify the task (see 1.6).

### 1.2 Agent tools (one action per reply)

| Action | What happens |
|---|---|
| `test` | Compile the latest ```cpp block; run it on every example seen so far; return verdict, output and expected output per example. |
| `run` | Run the program on an input the HDA writes in an ```input block. |
| `stress` | Compile a C++ generator from a ```gen block (policy-checked), feed its output (up to 64 MiB) to the program, return CPU time and output. This lets the HDA test speed on large graphs it builds itself. |
| `submit` | End the iteration and submit the latest program. If the step budget (8) runs out, the latest program is submitted automatically. |

### 1.3 Agent output

One C++17 source file per iteration (plus, in every reply, its current `<statement>`). The run's final harness **H\*** is the submission with the best validation accuracy across iterations (ties go to the later one).

### 1.4 Training loop (per run: task x condition x seed)

```
for it in 1..n (n = 5):
    give 10 new examples + feedback + best program + current statement
    HDA works for <= 8 tool steps, may revise <statement>, then submits
    policy check  -> reject (score 0) on any violation, with the reason as feedback
    compile       -> reject on error, compiler output as feedback
    score on hidden validation (40 graphs, 4-50 nodes, seen families)
    relabeling check: 5 validation graphs with fresh node IDs must give the same answer
                      (each failure reduces the iteration score proportionally)
    feedback = accuracy, verdict counts (AC/WA/TLE/RE/MLE), accuracy by size bucket and family, max CPU time
H* = best-validation submission -> evaluated once on test_small/medium/large/xl/stress
```
Validation graphs and answers are never shown to the HDA. The problem statement is not scored during training, but it is logged at every step so the team can see whether inference works.

### 1.5 Test generator (`hda/generator.py`, `configs/splits.json`)

- **Families.** Seen (examples, val): Erdős–Rényi, Barabási–Albert, random trees (bushy and long), disjoint unions of components. Unseen (added only in test): Watts–Strogatz, 2-D grids with 10% edges removed, caterpillar/path-like (long diameter), dense ER (p 0.3 to 0.6, n ≤ 1000), random regular, powerlaw-cluster.
- **Density bins:** average degree 1.2 to 3, 3 to 8, 8 to 20, cycled per family.
- **Task shaping:** diameter and MST graphs are connected; cycle detection is balanced 50/50 between forests and forests plus 1 to 3 chords; bipartite is balanced between bipartite graphs and ones with one odd-cycle edge; reachability queries are 50/50 reachable; shortest-path and max-flow queries are 85% reachable; common-neighbor queries pick a 2-hop pair 70% of the time; directed graphs keep about 25% of edges in both directions; weights 1 to 100 (capacities 1 to 50).
- **Anti-shortcut:** node IDs are a random permutation; edge order and endpoint order are shuffled.
- **Ground truth:** human-written C++ reference solvers (`reference/*.cpp`), cross-checked against NetworkX on 600 random graphs in `tests/test_reference.py`.

| Split | Nodes | Per task | Families | CPU limit | Seed offset |
|---|---|---:|---|---:|---:|
| examples (pool) | 4–50 | 120 | seen | 1 s | 1 |
| val (hidden, scored each iteration) | 4–50 | 40 | seen | 1 s | 2 |
| test_small | 4–50 | 40 | all | 1 s | 3 |
| test_medium | 100–300 | 30 | all | 2 s | 4 |
| test_large | 500–1000 | 30 | all | 2 s | 5 |
| test_xl | 2000 | 20 | all | 3 s | 6 |
| test_stress | 10k–100k (diameter ≤ 5k, max flow ≤ 20k) | 10 | all (no dense) | 5 s | 7 |

Base seed 5625; every split uses its own seed, so no graph is shared between splits.

**Tasks (phase 1, exact single-token answers):** shortest path (weighted, s t → distance or -1), reachability (directed, s t → YES/NO), connected components (count), cycle detection (YES/NO), diameter (connected, unweighted), common neighbors (u v → count), triangle count, bipartite check (YES/NO), MST weight, max flow (directed capacities, s t → value). NP-hard tasks (max clique, MIS, MVC, TSP, GED, MCS, Hamiltonian path) are phase 2.

### 1.6 Identifiability of hidden tasks (already measured on the pilot pool)

`hda/identifiability.py` checks, for k random examples, how many candidate tasks with the same input signature match every output. Candidates are the 10 tasks plus 37 distractors (edge count, max degree, is-connected, has-triangle, edge weight, strongly connected, widest path, out-capacity of s, and so on). Probability that the true task is the only survivor (400 trials, pilot pool; regenerate on ARC):

| Task | k=1 | k=3 | k=5 | k=10 | Main confusers |
|---|---:|---:|---:|---:|---|
| shortest_path | .81 | .98 | 1.00 | 1.00 | edge weight |
| reachability | .00 | .18 | .45 | **.86** | strongly connected, edge exists |
| connected_components | .76 | 1.00 | 1.00 | 1.00 | |
| cycle_detection | .01 | .41 | .67 | **.93** | has triangle, is connected |
| diameter | .78 | 1.00 | 1.00 | 1.00 | (diameter of largest component is identical on connected graphs) |
| common_neighbors | .60 | .97 | 1.00 | 1.00 | min degree |
| triangle_count | .43 | .90 | .97 | 1.00 | cycle rank |
| bipartite_check | .23 | .74 | .92 | 1.00 | is connected, is tree |
| mst_weight | .58 | .93 | .98 | 1.00 | total weight, max spanning tree |
| max_flow | .16 | .54 | .77 | .98 | min(out(s), in(t)), widest path |

With k = 10, every task is identifiable from the examples in at least 86% of draws, and later iterations add 10 more. This supports Samin's choice of k = 10 and gives an upper bound for the identification metric. Report this table.

### 1.7 Anti-memorization (both layers go in the report)

1. **Prompt:** rule 4 of the system prompt forbids memorizing or hardcoding answers, node IDs, edge lists, sizes, lookup tables or special cases, and says such programs are rejected and relabeling-checked.
2. **Static policy** (`hda/policy.py`): header whitelist (standard library only), bans on `system`/`popen`/`fork`/`exec`, file I/O, threads/OpenMP, inline asm, local includes; rejects literal arrays of 20+ numbers, more than 24 non-trivial numeric literals, and code containing two or more example answers verbatim (3+ digits).
3. **Behavioral:** validation and test graphs are fresh, hidden and never shown; node IDs are permuted; 5 graphs per evaluation are re-run with new node IDs and must give the same answer.
4. **Sandbox** (`hda/sandbox.py`): empty working directory, input on stdin only, CPU/memory/stack/output limits, so a program cannot read test files.

### 1.8 Logging (`<run>/log.jsonl`, one JSON event per line)

| Event | Fields |
|---|---|
| `config` | all run settings |
| `examples` | iteration, example IDs, expected answers |
| `prompt` | full system and user prompt per iteration |
| `reply` | raw reply, **reasoning trace** (`reasoning_content` from vLLM for gpt-oss), action, **current statement**, current code, token counts |
| `tool` | compile log, per-example outputs and verdicts, custom-run and stress results |
| `submission` | policy result, code, **the program's answers on every example and every validation graph** with verdicts, validation summary |
| `iteration` | validation accuracy, best so far, statement, verdict mix |
| `judge` | hidden condition: LLM judge verdict (EQUIVALENT/PARTIAL/WRONG) on the final statement |
| `test` | per-split summary and per-graph rows |

`final.json` has the validation curve, the statement at the end of every iteration, the judge label, tokens and all test summaries. Logs are large; keep them out of git (add `experiments/hda/**/log.jsonl` and `**/it*/` to `.gitignore`) and commit only `final.json`, `best_solution.cpp` and aggregates.

---

## 2. What already exists

A tested prototype is in project files: `/mnt/project-files/hda/hda_pipeline/` (also `hda_pipeline.tgz`). It was built and run in a cloud container (Python 3.13, g++ 13.3, NetworkX 3.x), not on ARC.

| File | Purpose |
|---|---|
| `hda/tasks.py` | 10 task definitions, statements, NetworkX references |
| `hda/generator.py` | families, density bins, task shaping, relabeling, splits |
| `hda/sandbox.py` | g++ compile, resource-limited run, verdicts |
| `hda/policy.py` | static policy and hardcoding checks |
| `hda/evaluate.py` | split scoring, breakdowns by size/family/density/seen, relabeling check |
| `hda/prompts.py` | every prompt the HDA sees |
| `hda/agent.py` | the training loop, tools, logging, judge, test evaluation |
| `hda/grid.py` | parallel, restartable grid runner |
| `hda/aggregate.py` | CSV + LaTeX tables, majority-answer baseline |
| `hda/human_baseline.py` | human reference solvers through the same sandbox |
| `hda/identifiability.py` | the analysis in 1.6 |
| `templates/graph_template.cpp` | loader given to the HDA |
| `reference/*.cpp` | human C++ solutions = ground truth and upper bound |
| `scripts/run_hda_grid.sbatch` | serve gpt-oss-120b with vLLM and run the grid |
| `tests/` | 18 tests: reference vs NetworkX, balance, verdicts (AC/WA/TLE/MLE/RE), 8 MiB stack overflow, policy, scripted agent loop, stress tool |

Verified in the container: all 18 tests pass; the full data build takes about 70 s; the human references score 100% on every split with max CPU 0.68 s (diameter, 5,000 nodes) against limits of 1 to 5 s. A naive Floyd–Warshall diameter scores 100% on val, medium and large and **0% (all TLE) on test_xl and test_stress**, which is the small-to-large failure mode the experiment is designed to expose. A recursive DFS for components still passes stress, so stress does not yet punish deep recursion (see H8).

---

## 3. Priority order

| Step | What | Priority | Compute |
|---|---|---|---|
| H0 | Install, tests, data, human baseline, identifiability | P0 | CPU, < 1 h |
| H1 | Serve gpt-oss-120b, smoke run, confirm reasoning traces are logged | P0 | 1 GPU, < 1 h |
| H2 | Main grid: 10 tasks x {given, hidden} x 3 seeds, k = 10, n = 5 | P0 | 1 GPU, ~4–8 h |
| H3 | Baselines: one-shot (n = 1), majority answer, human reference, old Python/NetworkX numbers | P0 | 1 GPU, ~1–2 h |
| H4 | Aggregate tables and figures | P0 | CPU |
| H5 | Methodology and results sections in the report | P0 | none |
| H6 | Qwen3-4B as HDA (comparison) | P1 | 1 GPU, ~2 h |
| H7 | No-feedback ablation (`--feedback none`) | P1 | 1 GPU, ~3 h |
| H8 | Deep-recursion split (paths of 200k–500k nodes) for linear tasks | P2 | CPU |
| Later | k sweep, NP-hard tasks with approximation scoring, validation including medium graphs, GraphArena real-graph families | phase 2 | |

If time runs short, H0–H5 alone are a complete experiment for the report.

---

## 4. Steps

### H0. Install and verify (P0, CPU)

```bash
cd <graphharness repo>; git fetch origin saminrk-experiments; git checkout saminrk-experiments; git pull --ff-only
cp -r /mnt/project-files/hda/hda_pipeline/* ./hda_pipeline/   # [VERIFY] how project files reach ARC; else scp the .tgz
cd hda_pipeline
module load gcc  # [VERIFY] g++ >= 9 with C++17
pip install -r requirements.txt   # into the existing graphharness venv
python -m pytest -q tests          # expect 18 passed
python -m hda.generator --out data/hda
python -m hda.human_baseline --out ../experiments/hda/human_reference
python -m hda.identifiability --data data/hda --trials 400 --out ../experiments/hda/identifiability.json
```
Checks: every human reference accuracy is 1.0 and max CPU is below a third of the split's limit. If ARC's CPUs are slower and a reference uses more than a third of a limit, raise that split's limit in `configs/splits.json` **now**, before any model run, and record it. Add `hda_pipeline/data/` and run logs to `.gitignore`. Commit code and configs as `hda: add HDA pipeline`.

### H1. Serve the HDA model and smoke test (P0)

- Model: **gpt-oss-120b** (MXFP4, fits one B200/H100 80 GB). [VERIFY] whether the weights are already on ARC under `/projects/psgroup/saminrk/`; if not, download from Hugging Face `openai/gpt-oss-120b` (about 65 GB, under the 70 GB rule).
- Start vLLM (inside `scripts/run_hda_grid.sbatch`, or interactively): `vllm serve <path> --served-model-name openai/gpt-oss-120b --max-model-len 65536`. Sampling: temperature 1.0, `reasoning_effort` medium (gpt-oss defaults).
- Smoke: `python -m hda.agent --task diameter --condition hidden --n-iter 2 --out ../experiments/hda/smoke --base-url http://HOST:PORT/v1 --test-splits test_small test_xl`
- Confirm in `log.jsonl`: `reply.reasoning` is non-empty (if it is null, check the vLLM version's reasoning parser flag [VERIFY] and fix before H2); `reply.statement` changes across steps; `submission.val_rows` and `example_outputs` are present; `final.json` exists. Record time per iteration to refine the budget.

### H2. Main grid (P0)

```bash
sbatch --export=ALL,HDA_OUT=../experiments/hda/main,HDA_CONDITIONS="given hidden",HDA_SEEDS="0 1 2",HDA_K=10,HDA_NITER=5,HDA_STEPS=8 scripts/run_hda_grid.sbatch
```
60 runs. The runner skips finished runs, so resubmitting after a timeout continues. Any run with `error.txt` is rerun once; if it fails again, report it as a failure and keep it in the counts.

### H3. Baselines (P0)

1. **One-shot HDA (no refinement):** same grid with `HDA_NITER=1`, `HDA_OUT=../experiments/hda/oneshot`. The gap between this and H2 is the value of iterative refinement.
2. **Majority answer:** produced by `hda.aggregate` (`hda_constant_baseline.csv`). Any task where this is high (for example common neighbors or triangle count on sparse graphs) must be flagged in the report.
3. **Human reference C++:** from H0, the upper bound.
4. **Library-assisted reference:** the existing Qwen3-4B + Python/NetworkX v2 numbers for the overlapping tasks (shortest path, components, cycle detection, diameter, reachability, common neighbors, triangle counting, max flow), cited from the existing results, clearly labeled as a different setting (per-instance agent with a graph library, different data).

### H4. Aggregate (P0)

```bash
python -m hda.aggregate --runs ../experiments/hda/main --data data/hda --out ../report/tables/hda
python -m hda.aggregate --runs ../experiments/hda/oneshot --data data/hda --out ../report/tables/hda_oneshot
```
Figures to make [CREATE] `hda/plots.py`, matplotlib, PDF + PNG into `report/figures/hda/`:
1. Accuracy vs graph size bucket (tiny … stress), lines for given and hidden, mean ± std over seeds, with the human reference and one-shot lines.
2. Validation accuracy per iteration (learning curve), given vs hidden.
3. Verdict mix (AC/WA/TLE/RE) by size bucket, to show whether large-graph failures are wrong answers or too slow.
4. Seen vs unseen families on test_small and test_large.
Tables: main (task x condition x split, plus gap = test_small − test_xl), identification (hidden: judge label, whether the statement changed across iterations, and functional identification = test_small accuracy ≥ 95%), cost (tokens, wall time, tool calls per run). Also pick 3 example statement trajectories from hidden runs (one corrected over iterations, one right from the start, one wrong) for the report, from `final.json["statements"]`.

### H5. Report (P0)

Add a new methodology section (or subsection, depending on the paper's structure) that a reader can follow with no other document. It must cover, in this order, with one table or figure where marked:
1. **Motivation:** why the setup moved from Python/NetworkX to C++ without libraries (cite the existing tool-access result: NetworkX does the algorithmic work).
2. **HDA input:** system prompt rules, statement given or hidden, k = 10 examples per iteration, template, tools. *Table: agent input and output.*
3. **HDA output:** one C++17 file; H\* = best validation iteration.
4. **Input/output format** of every graph (the three-part format in 1.1) and what the header reveals.
5. **Tasks:** the 10 phase-1 tasks with their answer formats. *Table.*
6. **Test generator:** families (seen vs unseen), density bins, task shaping, relabeling, ground truth. *Table: splits with sizes, counts, families, limits.*
7. **Training loop:** the pseudo-code in 1.4, step budget, iteration count, feedback content, what is hidden.
8. **Problem-statement conditions:** given vs hidden, the self-debate instruction, revisable statements, and how identification is measured (judge + functional). *Identifiability table from 1.6.*
9. **Anti-memorization:** the four layers in 1.7.
10. **Evaluation:** metrics (accuracy, verdict mix, gap, by family/density/size, identification, cost), baselines (H3), seeds and how ± is computed.
11. **Logging:** what is recorded each iteration (1.8), so results can be audited.
12. **Model and compute:** gpt-oss-120b, vLLM version, GPU, sampling settings, wall time.
Then a results subsection with the H4 tables and figures, and a short note in Limitations: phase 1 is polynomial-time tasks only; the header leaks the task family; the judge is the same model as the HDA. Keep the old Python/NetworkX sections; label them as the library-assisted setting. All drafted text uses the AI-DRAFT markers. Quote the full system prompt in an appendix.

### H6. Qwen3-4B as HDA (P1)

Same grid with `HDA_MODEL=Qwen/Qwen3-4B-Instruct-2507`, `HDA_MODEL_PATH=/projects/psgroup/saminrk/gh-qwen/...` [VERIFY], conditions `given hidden`, seeds `0 1 2`, `HDA_EFFORT=""` and `--reasoning-effort` unset (add `HDA_EXTRA` if needed), max model length 32k. Expect low scores (the earlier C pilot scored 12%); it shows how much the result depends on model strength.

### H7. No-feedback ablation (P1)

`HDA_EXTRA="--feedback none"`, given and hidden, seeds 0–2. Isolates the effect of validation feedback from simply getting more examples.

### H8. Deep-recursion split (P2)

[CREATE] a `test_deep` split: path and caterpillar graphs of 200k–500k nodes for connected components, cycle detection, reachability, shortest path and bipartite check, 5 per task, 5 s limit. Add it to `configs/splits.json` with a new seed offset **before** looking at any H2 test result, or report it as a post-hoc robustness check.

---

## 5. Sanity checks for every run

- `best_val_acc` > 0 for most given-condition runs; if all runs score 0, inspect `submission` events for policy rejections or compile errors before continuing.
- No `final.json` has `test` empty unless `best_it` is 0 (nothing valid submitted). Count those as 0 accuracy.
- Relabeling failures should be rare; if a program passes val but fails relabeling, keep it and report it.
- Time limits are the same for every condition and model.
- Hidden-condition prompts contain no statement text (grep `prompt.user` for the task's statement).

## 6. Budget

P0 (H1–H3): about 120 runs x up to 40 model calls, roughly 20–30M generated tokens. On one B200 with 16 parallel runs this should take about 6–10 GPU-hours. Re-estimate from H1. P1 adds about 5 GPU-hours.

## 7. Output layout

```
hda_pipeline/                       code (committed)
experiments/hda/
  JOBS.md, NOTES.md, human_reference/, identifiability.json
  smoke/, main/, oneshot/, qwen4b/, nofeedback/    one dir per run: final.json, best_solution.cpp (committed); log.jsonl, it*/ (ignored)
report/tables/hda/*.csv|.tex
report/figures/hda/*.pdf|.png
report/EDIT_LOG.md
```

## 8. Hand-off checklist

- [ ] 18 tests pass on ARC; data built; human reference 100%
- [ ] Smoke run logs reasoning traces, statements and answers
- [ ] H2 and H3 complete (or failures recorded)
- [ ] Tables and four figures in `report/`
- [ ] Methodology section with all 12 items, AI-DRAFT marked
- [ ] `JOBS.md`, `NOTES.md`, `EDIT_LOG.md` updated; branch `saminrk-experiments` pushed
