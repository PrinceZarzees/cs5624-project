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
