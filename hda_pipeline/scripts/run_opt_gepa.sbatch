#!/bin/bash
#SBATCH --job-name=hda_opt_gepa
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
OUT=experiments/hda/optimizers/gepa
mkdir -p "$OUT"
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8011 \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 32 \
  --disable-log-requests > "$OUT/vllm_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8011 420 || { tail -60 "$OUT/vllm_${SLURM_JOB_ID}.log"; exit 1; }
cd hda_pipeline
"$CLIENT_PYTHON" -m hda.opt.run --optimizer gepa --out ../experiments/hda/optimizers/gepa \
  --model qwen3-4b --base-url http://127.0.0.1:8011/v1 --rounds 8 --workers 32
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
