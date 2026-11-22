#!/bin/bash
#SBATCH --job-name=gh_qwen_pipeline
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_base
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Self-contained local-model pipeline: start vLLM, run selected stages, stop it.
#
# Required environment:
#   VENV          Python 3.12 venv containing vllm (shared storage)
#   MODEL_PATH    local model snapshot directory (shared storage)
#   CLIENT_PYTHON Python with graphharness installed (shared storage)
#   OUT           output directory (shared storage)
#
# Optional:
#   SERVED_NAME (default qwen3-4b), PORT (8000), MAX_MODEL_LEN (8192),
#   STAGES (default "baseline optimize scaling"), START_HARNESS (baseline_b_v2)
#   ROUNDS (8), TRAIN_PER_TASK (6), VAL_PER_TASK (8),
#   SCALING_SPLITS, SCALING_LIMIT_PER_TASK (default: full),
#   LIMIT_PER_TASK (empty = full baseline split)
#
# Example:
#   VENV=/projects/.../venv MODEL_PATH=/projects/.../snapshot \
#   CLIENT_PYTHON=/home/.../venv/bin/python OUT=/projects/.../experiments \
#   sbatch scripts/run_qwen_pipeline.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${VENV:?set VENV to a vLLM environment}"
: "${MODEL_PATH:?set MODEL_PATH to a model snapshot}"
: "${CLIENT_PYTHON:?set CLIENT_PYTHON to a Python with graphharness installed}"
: "${OUT:?set OUT to a shared output directory}"
: "${SERVED_NAME:=local-model}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.60}"
: "${MAX_NUM_SEQS:=8}"
: "${CUDA_MODULE:=cuda13.1/toolkit/13.1.1}"
: "${STAGES:=baseline optimize scaling}"
: "${START_HARNESS:=baseline_b_v2}"
: "${ROUNDS:=8}"
: "${TRAIN_PER_TASK:=6}"
: "${VAL_PER_TASK:=8}"
: "${SCALING_SPLITS:=test_small test_medium test_large test_stress}"
: "${SCALING_LIMIT_PER_TASK:=}"
: "${LIMIT_PER_TASK:=}"
: "${BEST_EVAL_LIMIT_PER_TASK:=}"
: "${ABLATION_HARNESS:=baseline_b_v2}"
: "${ABLATION_SPLIT:=test_large}"
: "${ABLATION_LIMIT_PER_TASK:=}"
: "${BENCH_SPECS:=ga_easy:data/grapharena_eval ga_hard:data/grapharena_eval graphqa_test:data/graphqa_eval}"
: "${BENCH_HARNESS:=baseline_b_v2}"
: "${BENCH_LIMIT_PER_TASK:=20}"
: "${HF_HOME:=${HOME}/.cache/huggingface}"

REPO="$(pwd)"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1
export GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR="$OUT/sandbox"
export HF_HOME
export PYTHONPATH="$REPO/src"

module load "$CUDA_MODULE"
export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}"
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "job=${SLURM_JOB_ID} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "model=$MODEL_PATH served_name=$SERVED_NAME"
echo "stages=$STAGES out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvcc --version | tail -1

"$VENV/bin/vllm" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --host "$HOST" --port "$PORT" \
  --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" --disable-log-requests \
  > "$LOGS/vllm_server.log" 2>&1 &
SERVER_PID=$!
ready=0
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    ready=1; echo "vllm ready after $((i*2))s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vllm died during startup"; break
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  tail -100 "$LOGS/vllm_server.log"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi
BASE="http://127.0.0.1:${PORT}/v1"
MODEL_ARGS=(--model "$SERVED_NAME" --base-url "$BASE")

run_stage() {
  local name="$1"; shift
  echo
  echo "=== $name ==="
  "$@"
  local rc=$?
  echo "=== $name exit=$rc ==="
  return $rc
}

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

if has_stage baseline; then
  LIMIT_ARGS=()
  if [[ -n "$LIMIT_PER_TASK" ]]; then LIMIT_ARGS=(--limit-per-task "$LIMIT_PER_TASK"); fi
  for HARNESS in baseline_a baseline_b baseline_b_v2; do
    run_stage "baseline $HARNESS" \
      "$CLIENT_PYTHON" -m graphharness.runner \
        --harness "$HARNESS" --split test_small "${LIMIT_ARGS[@]}" \
        --max-steps 8 --workers 8 \
        --out "$OUT/baseline/$HARNESS" "${MODEL_ARGS[@]}" \
        --max-tokens 768 --request-timeout 180
  done
fi

BEST="$OUT/optimizer/best_harness.json"
if has_stage optimize; then
  run_stage "optimizer" \
    "$CLIENT_PYTHON" -m graphharness.optimizer \
      --start "$START_HARNESS" --train-split train --val-split val \
      --rounds "$ROUNDS" --train-per-task "$TRAIN_PER_TASK" \
      --val-per-task "$VAL_PER_TASK" --max-steps 8 --workers 8 \
      --min-gain 0.02 --seed 5624 \
      --critic-model "$SERVED_NAME" --critic-max-tokens 1800 \
      --critic-temperature 0.5 \
      --out "$OUT/optimizer" --cache "$OUT/cache.sqlite" "${MODEL_ARGS[@]}"
  if [[ -f "$BEST" ]]; then
    BEST_ARGS=(--harness "$BEST" --split test_small --max-steps 8 --workers 8
               --no-resume --no-trajectories
               --out "$OUT/optimizer/eval_best_test_small")
    if [[ -n "$BEST_EVAL_LIMIT_PER_TASK" ]]; then
      BEST_ARGS+=(--limit-per-task "$BEST_EVAL_LIMIT_PER_TASK")
    fi
    run_stage "best harness on test_small" \
      "$CLIENT_PYTHON" -m graphharness.runner "${BEST_ARGS[@]}" "${MODEL_ARGS[@]}" \
        --max-tokens 768 --request-timeout 180
  fi
fi

if has_stage scaling; then
  HARNESSES=(baseline_a baseline_b baseline_b_v2)
  if [[ -f "$BEST" ]]; then HARNESSES+=("$BEST"); fi
  read -r -a SPLIT_ARRAY <<< "$SCALING_SPLITS"
  SCALE_ARGS=(--harnesses "${HARNESSES[@]}" --splits "${SPLIT_ARRAY[@]}"
              --max-steps 8 --workers 8 --out "$OUT/scaling")
  if [[ -n "$SCALING_LIMIT_PER_TASK" ]]; then
    SCALE_ARGS+=(--limit-per-task "$SCALING_LIMIT_PER_TASK")
  fi
  run_stage "scaling" \
    "$CLIENT_PYTHON" -m graphharness.scaling "${SCALE_ARGS[@]}" "${MODEL_ARGS[@]}" \
      --max-tokens 768 --request-timeout 180 --cache "$OUT/cache.sqlite"
fi

if has_stage ablation; then
  ABL_ARGS=(--harness "$ABLATION_HARNESS" --split "$ABLATION_SPLIT"
            --max-steps 8 --workers 8 --out "$OUT/ablation")
  if [[ -n "$ABLATION_LIMIT_PER_TASK" ]]; then
    ABL_ARGS+=(--limit-per-task "$ABLATION_LIMIT_PER_TASK")
  fi
  run_stage "ablation" \
    "$CLIENT_PYTHON" -m graphharness.ablation "${ABL_ARGS[@]}" "${MODEL_ARGS[@]}" \
      --max-tokens 768 --request-timeout 180 --cache "$OUT/cache.sqlite"
fi

if has_stage bench; then
  for spec in $BENCH_SPECS; do
    split="${spec%%:*}"; data_dir="${spec#*:}"
    run_stage "bench $BENCH_HARNESS / $split" \
      "$CLIENT_PYTHON" -m graphharness.runner \
        --harness "$BENCH_HARNESS" --split "$split" --data-dir "$data_dir" \
        --limit-per-task "$BENCH_LIMIT_PER_TASK" --max-steps 8 --workers 8 \
        --out "$OUT/bench/$split" "${MODEL_ARGS[@]}" \
        --max-tokens 768 --request-timeout 180 --cache "$OUT/cache.sqlite"
  done
fi

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
cp "$LOGS/vllm_server.log" "$LOGS/vllm_server_${SLURM_JOB_ID}.log" 2>/dev/null || true
for ext in out err; do
  cp "${SLURM_SUBMIT_DIR:-.}/gh_qwen_pipeline_${SLURM_JOB_ID}.${ext}" "$LOGS/" 2>/dev/null || true
done
echo "pipeline done"
