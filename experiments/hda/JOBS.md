# HDA job ledger

| Job ID | Script | Exports | Purpose | Start | End | Status | Output |
|---|---|---|---|---|---|---|---|
| 1550735 | hda_pipeline/scripts/run_hda_qwen.sbatch | Qwen3-4B BF16, 32k | H6 Qwen HDA main grid | pending | | pending | experiments/hda/qwen_main |
| 1550735 | hda_pipeline/scripts/run_hda_qwen.sbatch | Qwen3-4B BF16, 32k | H6 main grid (10 tasks x 2 conditions x 3 seeds) | 19:10 (projected) | 01:10 | PENDING | experiments/hda/qwen_main |
| 1550755 | hda_pipeline/scripts/run_hda_qwen_smoke.sbatch | Qwen3-4B BF16, 32k | k=10 given smoke (diameter, 2 iters) | 00:46 (projected) | 01:26 | PENDING | experiments/hda/qwen_smoke |

## H0 and CPU pilot

- H0 ran on the login node: 18/18 tests pass; data build 60 s; human reference
  100% on every split, max CPU 0.45 s; identifiability 400 trials matches the
  runbook table.
- Qwen3-4B Q4 CPU pilot (`qwen_cpu_pilot2`): given condition, k=3, 1 iteration,
  4 steps, 2048 max tokens. Diagnostics only; the BF16 GPU grid is the intended
  H6 run.
- A hidden-condition Q4 smoke (`smoke_qwen_cpu`) spent all steps on candidate
  statements and submitted no code.

## HDA optimizer jobs

| Job ID | Cluster | Script | Purpose | Start | End | Status | Output |
|---|---|---|---|---|---|---|---|
| 1554803 | owl | hda_pipeline/scripts/run_opt_baseline.sbatch | baseline A (60 rollouts) | 14:54 | 15:11 | COMPLETED | experiments/hda/optimizers/baseline |
| 1554804 | owl | run_opt_baseline.sbatch | baseline retry (correct OUT) | 14:54 | 15:11 | COMPLETED | same |
| 926162 | falcon | run_opt_gepa_falcon.sbatch | GEPA on l40s | – | – | CANCELLED (Resources) | – |
| 926163 | falcon | run_opt_gepa_a30.sbatch | GEPA on a30 | 18:00 (projected) | 22:00 | PENDING | experiments/hda/optimizers/gepa |
| 926164/5 | falcon | test-only preempt/t4 | probe | – | – | not queued | – |
