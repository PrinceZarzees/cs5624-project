#!/bin/bash
#SBATCH --job-name=gh_final_e2fix
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Replacement repeat for cells whose existing run broke the >5% infra-error rule.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E2_repeats
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.60 --max-num-seqs 8 \
  --seed 4 --disable-log-requests > "$OUT/logs/fixup_vllm.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8000 240 || { tail -60 "$OUT/logs/fixup_vllm.log"; exit 1; }
"$CLIENT_PYTHON" -m graphharness.runner --harness baseline_b --split test_stress \
  --max-steps 8 --workers 8 --no-resume --no-trajectories --model qwen3-4b \
  --base-url http://127.0.0.1:8000/v1 --max-tokens 768 --request-timeout 180 \
  --out "$OUT/baseline_b/rep4/test_stress"
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
