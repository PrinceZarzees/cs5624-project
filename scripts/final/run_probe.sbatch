#!/bin/bash
#SBATCH --job-name=gh_probe
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E7_router
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8090 \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.60 --max-num-seqs 16 \
  --seed 61 --disable-log-requests > "$OUT/logs/probe_vllm.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8090 300 || { tail -40 "$OUT/logs/probe_vllm.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8090/v1 --max-tokens 768 --request-timeout 180"
for H in baseline_b_v2 sparse dense tree large; do
  "$CLIENT_PYTHON" -m graphharness.runner --harness "experiments/review_router/harnesses/$H.json" --split val \
    --max-steps 8 --workers 12 --no-resume --no-trajectories --out "$OUT/val/$H" $M
done
"$CLIENT_PYTHON" -m graphharness.learned_router --scores "$OUT/val/*/scores.jsonl" --split val --out "$OUT/router.json"
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
echo done
