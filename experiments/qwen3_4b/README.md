# Qwen3-4B full experiment run

Primary model: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, 4.02B), BF16 on one
NVIDIA B200 via vLLM 0.34.0. Slurm job `1537011` on `owl-sr002` completed in
15 minutes.

Pipeline: `scripts/run_qwen_pipeline.sbatch`

## Full `test_small` baselines (150 problems)

| Harness | Accuracy | Notes |
|---|---:|---|
| `baseline_a` | 87.3% | minimal protocol + tools |
| `baseline_b` | 90.0% | original expert rules |
| `baseline_b_v2` | **93.3%** | 13-rule expert harness used in this run |

The current 14-rule v2 adds the generator-materialization rule; it fixed all
four previously failing diameter cases in a targeted check.

## Optimizer

`optimizer/` contains the 8-round optimizer starting from `baseline_b_v2`.
Initial and best validation accuracy were both 97.5%; every candidate was
rejected (validation 45-75%). `optimizer/best_harness.json` is therefore the
v2 harness.

A second optimizer from `baseline_a` is under `../qwen3_4b_opt_a/` (10 rounds,
initial/best validation 87.5%, all candidates rejected).

## Scaling

`scaling/` contains a 3-problem-per-task scaling pilot. A full-size run is in
`../qwen3_4b_scaling_full/`.

## Rule ablation

`../qwen3_4b_ablation_full/` contains the full 100-problem `test_large`
rule-by-rule ablation for the current v2 harness.

## Files

```text
baseline/<harness>/         full test_small predictions, scores, trajectories
optimizer/                  best_harness.json, history, candidates, checkpoints
optimizer/eval_best_test_small/
scaling/                    pilot scaling summary and per-run artifacts
logs/                       vLLM server log and Slurm output
failure_analysis.txt        task-level failure notes
```

Model/environment locations (shared project storage, not tracked by Git):

```text
/projects/psgroup/saminrk/gh-qwen/venv
/projects/psgroup/saminrk/gh-qwen/hf/models/models--Qwen--Qwen3-4B-Instruct-2507/
```
