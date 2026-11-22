#!/bin/bash
#SBATCH --job-name=gh_final_e3eval
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Clean EvoSkill run (the earlier two jobs shared one output dir), then evaluate
# every optimizer run that accepted an edit on four splits.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${ACTOR_PATH:?}"; : "${CRITIC_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E3_optimizers; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox_eval PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
# wait until the two earlier EvoSkill writers are gone
sleep 360
if [ -d experiments/final/E4_evoskill/baseline_b ]; then
  mv experiments/final/E4_evoskill/baseline_b experiments/final/E4_evoskill/baseline_b_mixed
fi
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$ACTOR_PATH" --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8030 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.32 --max-num-seqs 8 --seed 6 --disable-log-requests \
  > "$OUT/logs/eval_actor.log" 2>&1 &
A=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8030 420 || { tail -60 "$OUT/logs/eval_actor.log"; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$CRITIC_PATH" --served-model-name qwen2.5-7b \
  --host 127.0.0.1 --port 8031 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.55 --max-num-seqs 8 --seed 7 --disable-log-requests \
  > "$OUT/logs/eval_critic.log" 2>&1 &
C=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8031 420 || { tail -60 "$OUT/logs/eval_critic.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8030/v1 --max-tokens 768 --request-timeout 180"
MC="--critic-model qwen2.5-7b --critic-base-url http://127.0.0.1:8031/v1"
"$CLIENT_PYTHON" -m graphharness.evoskill --start baseline_b --rounds 3 \
  --train-per-task 4 --val-per-task 10 --out experiments/final/E4_evoskill/baseline_b \
  $M $MC --cache "$OUT/cache_eval.sqlite"

for RUN in skillopt_baseline_b rrsi_baseline_b meta_baseline_b; do
  HARN="$OUT/$RUN/best_harness.json"
  [ -f "$HARN" ] || continue
  grep -q '"accepted": true' "$OUT/$RUN/history.jsonl" || continue
  for SPLIT in test_small_fresh test_medium test_large test_stress; do
    DATA="data"; [ "$SPLIT" = "test_small_fresh" ] && DATA="data_final"
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$HARN" --split "$SPLIT" --data-dir "$DATA" \
      --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$OUT/eval_${RUN}/${SPLIT}" $M
  done
done
EVO=experiments/final/E4_evoskill/baseline_b
if [ -f "$EVO/best_harness.json" ] && grep -q '"accepted": true' "$EVO/history.jsonl"; then
  for SPLIT in test_small_fresh test_medium test_large test_stress; do
    DATA="data"; [ "$SPLIT" = "test_small_fresh" ] && DATA="data_final"
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$EVO/best_harness.json" --split "$SPLIT" --data-dir "$DATA" \
      --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$OUT/eval_evoskill_baseline_b/${SPLIT}" $M
  done
fi
kill $A $C 2>/dev/null || true; wait $A $C 2>/dev/null || true
echo done
