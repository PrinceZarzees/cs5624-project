# Optimizer notes

- O0 interface recorded in `INTERFACE.md`.
- O1 `val_opt` built with seed offset 12; 0 overlap.
- O2 override, validator and rollout runner implemented; tests in
  `hda_pipeline/tests/opt/`.
- O3 baseline pending (60 rollouts).

## O3 baseline (complete)

Empty prompt, acceptance set A = 20 instances x seeds 200/201/202, 60 rollouts:
- overall `val_opt` mean **0.8296**; seed means 0.859 / 0.769 / 0.861.
- per-instance headroom: given mostly 1.0; hidden is low:
  diameter 0.308, max flow 0.300, MST 0.475, triangle 0.667, components/cycle 0.700.
- cost: 4.21M prompt tokens, 0.84M completion tokens, 682 reply calls.
- baseline job 1554804.

## O0-O3 status

- O0 interface: `INTERFACE.md`.
- O1 `val_opt`: 400 records, seed offset 12, 0 overlap.
- O2 override/validator/rollout: implemented and unit-tested (8 optimizer tests,
  24 total).
- O3 baseline: 60 rollouts, mean `val_opt` 0.8296; given mostly 1.0, hidden
  provides signal (diameter 0.308, max flow 0.300, MST 0.475).
- O4 code for GEPA/SkillOpt/Meta-Harness/EvoSkill is in `hda/opt/run.py`;
  GEPA is queued on falcon a30 (job 926163, 18:00). O5 final grids wait on the
  optimized prompts.

## CPU proposer dry runs (one round, two instances, one HDA iteration)

All four optimizer entry points completed on the Q4 CPU server. No test split
was read.

- GEPA: proposal rejected; named `shortest path` and `Dijkstra`.
- SkillOpt: valid edit, child mean 0.75 vs parent 0.830 -> not accepted.
- Meta-Harness: proposal rejected; named `Dijkstra` and `shortest path`.
- EvoSkill: one skill added to the frontier; child mean 0.75; best unchanged.
- Proposer: Qwen3-4B Q4; 1 call each, ~5.4k prompt tokens (Evo 2.3k).
- Recheck: shortest given seed 201 -> 1.0; no optimized prompt improved A.

This validates the loops/validator but shows the 4B proposer is too weak/task-
specific on CPU. The GPU run uses the same proposer unless a stronger model is
available.

## GEPA GPU run (falcon a30, job 926163)

8 rounds, **0 accepted edits**. 7/8 proposals were rejected by the validator for
naming benchmark tasks/algorithms (`shortest path` 7x, `Dijkstra` 7x,
`reachability` 5x, `connected component` 1x). The one valid child
(`iteration_note`) scored 0.7838 on A vs the parent's 0.8296 and was rejected by
the 0.02 margin. The run crashed during the empty recheck when a hidden rollout
exceeded the 1800 s subprocess timeout; the recheck is now skipped when no edit
is accepted and per-rollout timeouts are recorded as failures instead of
aborting the optimizer.
