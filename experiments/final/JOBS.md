# Final experiment job ledger

All jobs use `--account=psgroup --partition=b200_normal_q --qos=owl_b200_normal_short`
(1x B200 unless noted). Output logs are under `experiments/final/logs/`.

| Job ID | Script | Exports | Purpose | Start | End | Status | Output |
|---|---|---|---|---|---|---|---|
| 1546227 | scripts/serve_vllm.sbatch | Qwen3-4B BF16, port 8000, 8k | standalone actor server | – | – | cancelled (blocked the self-contained E1/E2 job) | experiments/final/logs/actor_1546227.out |
| 1546237 | scripts/final/run_e1_e2.sbatch | Qwen3-4B BF16, 8k, 8 workers | E1 held-out 3x + E2 scaling 2x + E2 tool-access 2x | 2026-11-21 10:08 | 2026-11-21 16:08 (limit) | RUNNING | experiments/final/E1_heldout, E2_repeats, E2_tool_access |
| 1546238 | scripts/final/run_e3.sbatch | 2 GPUs, full grid | E3 full optimizer grid | – | – | cancelled (start projected 21:41, after the session) | – |
| 1546342 | scripts/final/run_e4.sbatch | 2 GPUs | E4 full EvoSkill | – | – | cancelled (superseded by 1547527) | – |
| 1546376 | scripts/final/run_e6.sbatch | Qwen3-4B BF16, 8k | E6 A/B/v2 on GraphArena easy/hard + GraphQA | 2026-11-21 10:53 | 2026-11-21 16:53 (limit) | RUNNING | experiments/final/E6_real_bench |
| 1546388 | scripts/final/run_e3.sbatch | 1 GPU, full grid | E3 full optimizer grid | – | – | cancelled (start projected 16:53, 6 h, after the session) | – |
| 1546394 | scripts/final/run_e4.sbatch | 1 GPU | E4 full EvoSkill | – | – | cancelled (start projected 22:54) | – |
| 1547370 | scripts/final/run_e2_fixup.sbatch | Qwen3-4B BF16, 8k | E2 replacement repeat for Baseline B `test_stress` (rep1 had 6/50 no-prediction errors; reps 1 and 3 are flagged) | pending | | pending | experiments/final/E2_repeats/baseline_b/rep4 |
| 1547527 | scripts/final/run_e3_lite.sbatch | 1 GPU, 4B actor + 7B critic | Reduced E3: SkillOpt/GEPA/RRSI/Meta-Harness from B, 6 rounds | 2026-11-21 11:19 | 2026-11-21 12:49 (limit) | COMPLETED (SkillOpt, GEPA, RRSI, Meta done; Evo cut) | experiments/final/E3_optimizers |
| 1547590 | scripts/final/run_e11.sbatch | Qwen3-4B BF16, 32k context, port 8010 | E11 no-tools A/v2 on medium+large | 2026-11-21 11:59 | 2026-11-21 12:10 | COMPLETED | experiments/final/E11_notools_longctx |
| 1547895 | scripts/final/run_e3b.sbatch | 1 GPU, B actor/critic, ports 8002/8003 | second RRSI run (5 rounds) + EvoSkill attempt | 2026-11-21 12:11 | 2026-11-21 12:39 | COMPLETED | E3_optimizers/rrsi_baseline_b, E4_evoskill (mixed) |
| 1547901 | scripts/final/run_e6b.sbatch | Qwen3-4B BF16, port 8020 | balanced 30/task fallback | – | – | CANCELLED (E6 finished all 100/task runs) | – |
| 1548109 | scripts/final/run_e3_eval.sbatch | 1 GPU, ports 8030/8031 | eval + clean EvoSkill, first attempt | – | – | CANCELLED (actor KV-cache failure at 0.25) | – |
| 1548123 | scripts/final/run_e3_eval.sbatch | 1 GPU, ports 8030/8031 | eval + clean EvoSkill, resubmitted | – | – | CANCELLED (superseded by 1548177) | – |
| 1548177 | scripts/final/run_e3_eval.sbatch | 1 GPU, 0.32/0.55 sequential servers | clean EvoSkill (3 rounds) + E3d eval of changed harnesses | 2026-11-21 12:57 | 2026-11-21 13:41 | COMPLETED | E3_optimizers/eval_*, E4_evoskill/baseline_b |

## Notes on job changes

- `1546227` was cancelled because it held a GPU while the self-contained `1546237`
  waited. `1546237` then started an hour earlier than the original projection.
- The full E3/E4 jobs were projected to start after the interactive session ended
  (21:41 / 22:54 and 16:53 + 6 h). They were replaced by `1547527`, which runs from
  Baseline B only, 6 rounds, val 10/task, and adds one EvoSkill run.
- No GPU experiment was run on a test split for tuning; optimizers read `train`
  and `val` only.

## Remaining-experiment jobs (2026-11-21 afternoon)

| Job ID | Script | Purpose | Start | End | Status | Output |
|---|---|---|---|---|---|---|
| 1548282 | run_e3c.sbatch | E3c overfitting probe + checkpoint eval | 14:00 | 14:35 | COMPLETED | E3_probe/ |
| 1548283 | run_e14.sbatch | E14 cross-model (7B actor) | – | – | CANCELLED (GPU contention) | – |
| 1548286/87 | run_e5*.sbatch | full E5 attempts | – | – | CANCELLED (no GPU) | – |
| 1548382 | run_e7.sbatch | learned router + decomposition | – | – | CANCELLED (resubmitted) | – |
| 1548421 | run_e10.sbatch | E10 2x2 | – | – | CANCELLED (resubmitted) | – |
| 1548451 | run_remaining.sbatch | combined E7/E10/E5 | – | – | CANCELLED (16-CPU request did not fit) | – |
| 1548462 | run_probe.sbatch | short E7 val probe | 14:35 | 14:40 | FAILED (GPU OOM while E3c held the GPU) | – |
| 1548488 | run_e7.sbatch | E7 global/rule/learned + decomposition | 14:40 | 16:10 (limit) | RUNNING | E7_router/ |
| 1548735 | run_e10_e5.sbatch | E10 2x2 on small + E5 labels for 40 candidates + judges | 17:49 (projected) | 19:04 (limit) | PENDING | E10_lang_lib/, E5_proxy/ |
| 1548490/1548567/68 | run_e5_*.sbatch | split E5 attempts | – | – | CANCELLED (no GPU) | – |
| 1549257 | run_e5_a30.sbatch | E5 labels/judges for 20 candidates | 19:46 (projected) | 20:16 | PENDING | E5_proxy/ |
| 1549258 | run_e10.sbatch | E10 2x2 on test_small | 20:16 (projected) | 20:41 | PENDING | E10_lang_lib/ |
| 1549291 | run_e7_learned.sbatch | learned router rerun after interface fix | 20:42 (projected) | 21:12 | PENDING | E7_router/disc/learned |
