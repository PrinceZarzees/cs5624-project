#!/bin/bash
#SBATCH --job-name=gh_final_e3b
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Continuation of E3-lite: RRSI-lite and EvoSkill from Baseline B.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${ACTOR_PATH:?}"; : "${CRITIC_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E3_optimizers; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox_e3b PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$ACTOR_PATH" --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8002 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.25 --max-num-seqs 8 --seed 3 --disable-log-requests \
  > "$OUT/logs/actor_b.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$CRITIC_PATH" --served-model-name qwen2.5-7b \
  --host 127.0.0.1 --port 8003 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.65 --max-num-seqs 8 --seed 4 --disable-log-requests \
  > "$OUT/logs/critic_b.log" 2>&1 &
C=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8002 420 || { tail -80 "$OUT/logs/actor_b.log"; exit 1; }
scripts/final/wait_for_vllm.sh 127.0.0.1 8003 420 || { tail -80 "$OUT/logs/critic_b.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8002/v1 --max-tokens 768 --request-timeout 180"
MC="--critic-model qwen2.5-7b --critic-base-url http://127.0.0.1:8003/v1"
S=baseline_b

# E3d: evaluate every changed harness on the four scaling splits (1 run each).
run_eval() {
  local run="$1" harness="$2"
  for SPLIT in test_small_fresh test_medium test_large test_stress; do
    local DATA="data"; [ "$SPLIT" = "test_small_fresh" ] && DATA="data_final"
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$harness" --split "$SPLIT" --data-dir "$DATA" \
      --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$OUT/eval_${run}/${SPLIT}" $M
  done
}
for RUN in skillopt_baseline_b gepa_baseline_b; do
  HARN="$OUT/$RUN/best_harness.json"
  if [ -f "$HARN" ] && grep -q '"accepted": true' "$OUT/$RUN/history.jsonl"; then
    run_eval "$RUN" "$HARN"
  fi
done

"$CLIENT_PYTHON" -m graphharness.optimizer --start "$S" --rrs --rounds 4 \
  --train-per-task 4 --val-per-task 10 --out "$OUT/rrsi_${S}" $M $MC --cache "$OUT/cache_b.sqlite"
"$CLIENT_PYTHON" -m graphharness.evoskill --start "$S" --rounds 3 \
  --train-per-task 4 --val-per-task 10 --out "experiments/final/E4_evoskill/${S}" $M $MC --cache "$OUT/cache_b.sqlite"
kill $A $C 2>/dev/null || true; wait $A $C 2>/dev/null || true
echo done
