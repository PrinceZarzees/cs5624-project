#!/bin/bash
#SBATCH --job-name=gh_llama_pipeline
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_base
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Self-contained CPU fallback pipeline: start llama.cpp, run baseline_b_v2,
# stop the server.
#
#   LLAMA_VENV=/path/to/llama-cpp-venv \
#   MODEL_GGUF=/path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
#   CLIENT_PYTHON=/path/to/python-with-graphharness \
#   OUT=/path/to/experiments/qwen2_5_7b_cpu \
#   sbatch scripts/run_llama_cpp_cpu_pipeline.sbatch

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${LLAMA_VENV:?set LLAMA_VENV}"
: "${MODEL_GGUF:?set MODEL_GGUF}"
: "${CLIENT_PYTHON:?set CLIENT_PYTHON}"
: "${OUT:?set OUT}"
: "${MODEL_ALIAS:=qwen2.5-7b}"
: "${PORT:=8001}"
: "${N_CTX:=8192}"
: "${N_THREADS:=64}"
: "${N_THREADS_BATCH:=96}"
: "${CHAT_FORMAT:=chatml}"
: "${SPLIT:=test_small}"
: "${HARNESS:=baseline_b_v2}"
: "${LIMIT_PER_TASK:=}"
: "${MAX_STEPS:=8}"

REPO="$(pwd)"
LOGS="$OUT/logs"
mkdir -p "$LOGS"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR="$OUT/sandbox"

echo "job=${SLURM_JOB_ID} host=$(hostname) model=$MODEL_GGUF"
"$LLAMA_VENV/bin/python" -m llama_cpp.server \
  --model "$MODEL_GGUF" --model_alias "$MODEL_ALIAS" \
  --host 127.0.0.1 --port "$PORT" --n_ctx "$N_CTX" --n_batch 512 \
  --n_threads "$N_THREADS" --n_threads_batch "$N_THREADS_BATCH" \
  --chat_format "$CHAT_FORMAT" \
  > "$LOGS/llama_server.log" 2>&1 &
SERVER_PID=$!
ready=0
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1; echo "llama.cpp ready after $((i*2))s"; break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  tail -100 "$LOGS/llama_server.log"
  kill "$SERVER_PID" 2>/dev/null || true
  exit 1
fi

ARGS=(--harness "$HARNESS" --split "$SPLIT" --max-steps "$MAX_STEPS" --workers 1
      --model "$MODEL_ALIAS" --base-url "http://127.0.0.1:${PORT}/v1"
      --max-tokens 512 --request-timeout 300 --out "$OUT/$HARNESS/$SPLIT")
if [[ -n "$LIMIT_PER_TASK" ]]; then ARGS+=(--limit-per-task "$LIMIT_PER_TASK"); fi

"$CLIENT_PYTHON" -m graphharness.runner "${ARGS[@]}"
rc=$?
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
for ext in out err; do
  cp "${SLURM_SUBMIT_DIR:-.}/gh_llama_pipeline_${SLURM_JOB_ID}.${ext}" "$LOGS/" 2>/dev/null || true
done
echo "pipeline exit=$rc"
exit $rc
