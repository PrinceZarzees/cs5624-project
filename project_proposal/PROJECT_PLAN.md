# CS 5624 Project Plan — Harness Optimization for Graph Reasoning

## What's Already Done (Phase 1 ✅)

- [x] Repository setup (src/graphharness/, configs, data, tests, scripts)
- [x] Synthetic data pipeline (train/val/test splits, 4–2000 nodes, seed-reproducible)
- [x] 6 task types: shortest_path, connected_components, cycle_detection, diameter, max_clique, reachability
- [x] 3 graph families: Erdős–Rényi, Barabási–Albert, Watts–Strogatz
- [x] Ground truth solvers (NetworkX-based exact solvers for all tasks)
- [x] Answer scoring pipeline (parse → verify → score, CLI batch mode)
- [x] GraphArena adapter (pickle → unified JSONL)
- [x] GraphQA adapter (HuggingFace → unified JSONL)
- [x] Graph encoding for prompts (edge_list, adjacency_list)
- [x] 50+ pytest tests
- [x] Slurm scripts for ARC cluster

---

## Reviewer Feedback Tracker

- [ ] F1: SkillLift-style proxy scorer for cheap harness ranking (arXiv:2609.15396)
- [ ] F2: Tools-on vs tools-off analysis (does NetworkX access trivialize tasks?)
- [ ] F3: Stratified analysis by graph density, topology, task difficulty
- [ ] F4: Additional baselines — GEPA (prompt evolution), SkillOpt/EvoSkill (skill evolution)
- [ ] F5: Report optimization costs (tokens, time, $)
- [ ] F6: Specialized harnesses per graph class + agentic router
- [ ] F7: RRSI-style overfitting prevention (arXiv:2609.24972)

---

## WEEK 1 (Nov 22–28): Agent Sandbox, Baselines, Optimization

### Member 1 — Agent Environment & Tool Modes

- [ ] Build LLM interaction loop (system prompt → model call → parse answer → score)
- [ ] Support API models (DeepSeek-V4.1-Flash / GLM-5.3 on ARC) and local models
- [ ] Add execution timeout (30s) and token-budget guards
- [ ] Implement tools-on/off switch (F2): --tools flag for sandboxed NetworkX calls
- [ ] Write run_experiment.py (takes harness config + split + model → preds.jsonl)
- [ ] Add cost instrumentation: log wall-clock time, token counts, API cost per problem (F5)
- [ ] Smoke test: end-to-end run of Baselines A & B on 20 train problems

### Member 2 — Baselines & Graph Feature Metadata

- [ ] Write Baseline A (zero-shot): minimal prompt, no instructions, no tools
- [ ] Write Baseline B (hand-crafted expert harness): best-practice instructions for graph reasoning
- [ ] Extend record metadata with graph features (F3): density, avg clustering coefficient, degree stats (mean, max, std)
- [ ] Run Baselines A & B on train, val, test_small (both tool modes)
- [ ] Run Baselines A & B on test_medium, test_large, test_stress (both tool modes)

### Member 3 — Optimization Loop & Overfitting Prevention

- [ ] Implement GEPA-style optimizer (F4): run harness → collect failures → critic LLM edits → accept if val improves
- [ ] Implement SkillOpt-style variant (F4): evolve textual skill snippets appended to system prompt
- [ ] Implement SkillLift-inspired proxy scorer (F1): rubric-based cheap ranking of candidate harnesses
- [ ] Add RRSI-inspired overfitting prevention (F7): validation gating, node-ID randomization, complexity penalty
- [ ] Run 15–25 optimization rounds on small training graphs
- [ ] Log every harness version, scores, token costs, wall-clock time
- [ ] Save frozen best harness H*_small

### Member 4 — Evaluation Infrastructure & Preliminary Analysis

- [ ] Build stratified analysis pipeline (F3): breakdowns by task × graph family × density bins × node-count bins
- [ ] Build cost tracking dashboard (F5): tokens, time, $ per optimization round, per split, per task
- [ ] Write visualization scripts: accuracy vs graph size curves, optimization convergence, generalization gap, cost breakdown
- [ ] Generate preliminary figures from baseline + early optimization results
- [ ] Identify and report any pipeline bugs or issues

---

## WEEK 2 (Nov 29 – Dec 6): Large-Scale Evaluation, Ablation, Report

### Member 1 — Large-Scale Runs & Cross-Model Transfer

- [ ] Run optimized harness H* on test_medium, test_large, test_stress (both tool modes)
- [ ] Cross-model transfer: run best harness from Model 1 on a different Model 2
- [ ] Write up tools-on vs tools-off findings (F2): which tasks become trivial with NetworkX?
- [ ] Draft report section: Methodology (agent sandbox, tool modes, experiment setup)

### Member 2 — Rule Ablation & Error Analysis

- [ ] Diff H* vs Baseline A: identify every distinct rule/skill added by optimizer
- [ ] Rule-by-rule ablation: disable each rule on test_large, measure accuracy drop
- [ ] Error taxonomy: classify large-graph failures (context overflow, wrong algorithm, timeout, hallucinated edges, etc.)
- [ ] Generalization analysis: which rules transferred well? Cross-reference with graph characteristics (F3)
- [ ] Draft report section: Results (ablation tables, error categories, qualitative examples)

### Member 3 — Specialized Harnesses & Router

- [ ] Optimize separate harnesses per graph class (F6): dense graphs, sparse/tree-like graphs, NP-hard tasks
- [ ] Build agentic router (F6): given graph features → select best specialized harness
- [ ] (Stretch) Subgraph decomposition: if large graph has components, apply different harness per subgraph
- [ ] Evaluate router: single H* vs router-selected specialized harnesses
- [ ] Draft report section: Optimization (loop design, convergence, RRSI, proxy scorer, router)

### Member 4 — Full Evaluation, Final Plots & Report Lead

- [ ] Run complete evaluation pipeline on all methods × all splits
- [ ] Compute generalization gap: Score(H* | D_small) − Score(H* | D_large) for all variants
- [ ] Produce cost-effectiveness analysis (F5): accuracy improvement per token/dollar spent
- [ ] Produce stratified performance heatmap (F3): tasks × families × density × scale
- [ ] Polish all paper figures (6+ figures)
- [ ] Draft report sections: Introduction, Related Work (add SkillLift + RRSI refs), Experimental Setup, Conclusion
- [ ] Compile all sections, coordinate final proofreading

---

## Final Deliverables Checklist

- [ ] Agent sandbox with tools-on/off modes
- [ ] Baseline A (zero-shot) and B (hand-crafted) scores across all splits
- [ ] GEPA-style optimizer implementation
- [ ] SkillOpt-style optimizer implementation
- [ ] SkillLift-inspired proxy scorer
- [ ] RRSI-inspired overfitting prevention
- [ ] Frozen best harness H*_small with full optimization logs
- [ ] Specialized harnesses (dense, sparse, NP-hard)
- [ ] Agentic router for harness selection
- [ ] Full evaluation across all scales (4–2000 nodes)
- [ ] Stratified analysis by density, topology, task difficulty
- [ ] Optimization cost report (tokens, time, $)
- [ ] Rule ablation study
- [ ] Error taxonomy on large-graph failures
- [ ] Cross-model transfer experiment
- [ ] Generalization gap table and scaling curves
- [ ] Final report (ACL format)
- [ ] Clean GitHub repository with reproducible code

---

## Priority Tiers

- P0 (Must Have): Agent sandbox, baselines, single-harness optimization, large-scale evaluation, generalization gap, cost report, ablation, final report
- P1 (Should Have): Router + specialized harnesses (F6), cross-model transfer, stratified heatmaps
- P2 (Nice to Have): Proxy scorer (F1), subgraph decomposition routing

---

## Key Deadlines

- [ ] Nov 27: Agent sandbox working, baselines running
- [ ] Nov 28: Optimization complete, H*_small frozen, baseline scores done
- [ ] Dec 2: All large-scale evaluation complete
- [ ] Dec 3: Ablation, router experiments, error analysis done
- [ ] Dec 4: All figures finalized
- [ ] Dec 6: Final report submitted

---

## References to Add

- SkillLift (arXiv:2609.15396) — rubric evolution for cheap harness ranking
- RRSI (arXiv:2609.24972) — overfitting prevention during harness optimization
