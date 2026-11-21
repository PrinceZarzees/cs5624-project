#!/bin/bash
#SBATCH --job-name=gh_final_e14
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# E14: cross-model check with Qwen2.5-7B-Instruct as actor (the larger model on disk).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E14_cross_model; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen2.5-7b --host 127.0.0.1 --port 8050 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.65 --max-num-seqs 8 \
  --seed 21 --disable-log-requests > "$OUT/logs/vllm.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8050 420 || { tail -60 "$OUT/logs/vllm.log"; exit 1; }
M="--model qwen2.5-7b --base-url http://127.0.0.1:8050/v1 --max-tokens 768 --request-timeout 180"
for H in baseline_a baseline_b_v2; do
  for spec in "test_small_fresh:data_final" "test_large:data"; do
    SPLIT="${spec%%:*}"; DATA="${spec#*:}"
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$H" --split "$SPLIT" --data-dir "$DATA" \
      --max-steps 8 --workers 8 --no-resume --no-trajectories --out "$OUT/$H/$SPLIT" $M
  done
done
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
echo done
