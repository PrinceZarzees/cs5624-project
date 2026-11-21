#!/bin/bash
#SBATCH --job-name=gh_final_e6b
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Balanced fallback: Baseline B and v2 at 30 problems/task.
# A's 100/task run is filtered to the same 30/task ids when aggregating.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E6_real_bench30; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8020 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.60 --max-num-seqs 8 \
  --seed 5 --disable-log-requests > "$OUT/logs/vllm.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8020 360 || { tail -60 "$OUT/logs/vllm.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8020/v1 --max-tokens 768 --request-timeout 180"
for spec in "ga_easy:data/grapharena_eval100" "ga_hard:data/grapharena_eval100" "graphqa_test:data/graphqa_eval50"; do
  split="${spec%%:*}"; data="${spec#*:}"
  for H in baseline_b baseline_b_v2; do
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$H" --split "$split" --data-dir "$data" \
      --limit-per-task 30 --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$OUT/$split/$H" $M
  done
done
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
