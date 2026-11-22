#!/bin/bash
#SBATCH --job-name=gh_e7_decomp
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; E7=experiments/final/E7_router
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$E7/sandbox_decomp PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8062 \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.55 --max-num-seqs 8 \
  --seed 71 --disable-log-requests > "$E7/logs/decomp_vllm.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8062 360 || { tail -50 "$E7/logs/decomp_vllm.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8062/v1 --max-tokens 768 --request-timeout 180"
IDS=$(cat "$E7/ids12.txt")
LIB=experiments/review_router/harnesses
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --ids $IDS --mode decomposed --repair-components --max-steps 8 --out "$E7/disc/decomp_rule12" $M
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --ids $IDS --mode decomposed --repair-components --learned-router "$E7/router.json" \
  --max-steps 8 --out "$E7/disc/decomp_learned12" $M
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
echo done
