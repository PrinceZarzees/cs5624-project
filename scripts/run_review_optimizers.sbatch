#!/bin/bash
#SBATCH --job-name=gh_review_opt
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen
OUT=experiments/review_optimizers
LOGS=$OUT/logs; mkdir -p "$LOGS"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
: "${PORT:=8000}"
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port "$PORT" \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.60 --max-num-seqs 8 \
  --disable-log-requests > "$LOGS/vllm_server.log" 2>&1 &
PID=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1 && { echo ready; break; }
  kill -0 $PID 2>/dev/null || break
  sleep 2
done
curl -sf http://127.0.0.1:${PORT}/health >/dev/null || { tail -100 "$LOGS/vllm_server.log"; exit 1; }
MODEL_ARGS="--model qwen3-4b --base-url http://127.0.0.1:${PORT}/v1 --max-tokens 768 --request-timeout 180"
run() { echo "=== $* ==="; "$@"; echo "=== exit=$? ==="; }

# R7: RRSI from A and v2
run "$CLIENT_PYTHON" -m graphharness.optimizer --start baseline_a --rrs --rounds 5 \
  --train-per-task 4 --val-per-task 3 --out "$OUT/rrsi_a" --cache "$OUT/cache.sqlite" $MODEL_ARGS
run "$CLIENT_PYTHON" -m graphharness.optimizer --start baseline_b_v2 --rrs --rounds 5 \
  --train-per-task 4 --val-per-task 3 --out "$OUT/rrsi_bv2" --cache "$OUT/cache.sqlite" $MODEL_ARGS

# R4: GEPA prompt evolution from A
run "$CLIENT_PYTHON" -m graphharness.gepa --start baseline_a --rounds 3 --candidates-per-round 2 \
  --train-per-task 4 --val-per-task 4 --out "$OUT/gepa_a" --cache "$OUT/cache.sqlite" $MODEL_ARGS

# Evaluate RRSI/GEPA best harnesses on a small held-out slice
for spec in "rrsi_a:$OUT/rrsi_a/best_harness.json" "rrsi_bv2:$OUT/rrsi_bv2/best_harness.json" "gepa_a:$OUT/gepa_a/best_harness.json"; do
  name="${spec%%:*}"; harness="${spec#*:}"
  run "$CLIENT_PYTHON" -m graphharness.runner --harness "$harness" --split test_small \
    --limit-per-task 5 --max-steps 8 --workers 8 --no-resume --no-trajectories \
    --out "$OUT/eval_$name" $MODEL_ARGS
done

# R6: routed and decomposed on the disconnected additive set
run "$CLIENT_PYTHON" -m graphharness.router --library experiments/review_router/harnesses \
  --split disconnected_additive --data-dir data --mode routed --limit-per-task 2 \
  --out "$OUT/router_routed" $MODEL_ARGS
run "$CLIENT_PYTHON" -m graphharness.router --library experiments/review_router/harnesses \
  --split disconnected_additive --data-dir data --mode decomposed --limit-per-task 2 \
  --out "$OUT/router_decomposed" $MODEL_ARGS

kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
echo done
