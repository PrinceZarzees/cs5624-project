#!/bin/bash
#SBATCH --job-name=gh_final_e4
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${ACTOR_PATH:?}"; : "${CRITIC_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E4_evoskill; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$ACTOR_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.25 --max-num-seqs 8 --seed 1 --disable-log-requests > "$OUT/logs/actor.log" 2>&1 &
A=$!
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$CRITIC_PATH" --served-model-name qwen2.5-7b --host 127.0.0.1 --port 8001 --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.65 --max-num-seqs 8 --seed 2 --disable-log-requests > "$OUT/logs/critic.log" 2>&1 &
C=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8000 360; scripts/final/wait_for_vllm.sh 127.0.0.1 8001 360
M="--model qwen3-4b --base-url http://127.0.0.1:8000/v1 --max-tokens 768 --request-timeout 180"
MC="--critic-model qwen2.5-7b --critic-base-url http://127.0.0.1:8001/v1"
for S in baseline_a baseline_b; do
  "$CLIENT_PYTHON" -m graphharness.evoskill --start "$S" --rounds 10 --train-per-task 4 --val-per-task 10 \
    --out "$OUT/$S" $M $MC --cache "$OUT/cache.sqlite"
done
kill $A $C 2>/dev/null || true; wait $A $C 2>/dev/null || true
