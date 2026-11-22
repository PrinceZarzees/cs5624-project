#!/bin/bash
#SBATCH --job-name=gh_rrsi2
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/review_optimizers2; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.60 --max-num-seqs 8 \
  --disable-log-requests > "$OUT/logs/vllm_server.log" 2>&1 &
PID=$!
for i in $(seq 1 180); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 2; done
curl -sf http://127.0.0.1:8000/health >/dev/null || { tail -100 "$OUT/logs/vllm_server.log"; exit 1; }
MODEL_ARGS="--model qwen3-4b --base-url http://127.0.0.1:8000/v1 --max-tokens 768 --request-timeout 180"
for start in baseline_a baseline_b_v2; do
  name="${start//_/}"
  "$CLIENT_PYTHON" -m graphharness.optimizer --start "$start" --rrs --rounds 3 \
    --train-per-task 4 --val-per-task 3 --out "$OUT/$name" --cache "$OUT/cache.sqlite" $MODEL_ARGS
  "$CLIENT_PYTHON" -m graphharness.runner --harness "$OUT/$name/best_harness.json" \
    --split test_small --limit-per-task 5 --max-steps 8 --workers 8 --no-resume --no-trajectories \
    --out "$OUT/eval_$name" $MODEL_ARGS
done
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
