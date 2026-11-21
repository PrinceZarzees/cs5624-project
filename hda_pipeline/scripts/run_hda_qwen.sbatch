#!/bin/bash
#SBATCH --job-name=hda_qwen
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
OUT=../experiments/hda/qwen_main
mkdir -p "$OUT"
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8011 \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 16 \
  --disable-log-requests > "$OUT/vllm.log" 2>&1 &
PID=$!
for i in $(seq 1 240); do curl -sf http://127.0.0.1:8011/v1/models >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:8011/v1/models >/dev/null || { tail -60 "$OUT/vllm.log"; exit 1; }
"$CLIENT_PYTHON" -m hda.grid --conditions given hidden --seeds 0 1 2 --k 10 --n-iter 5 --steps 8 \
  --out "$OUT" --model qwen3-4b --base-url http://127.0.0.1:8011/v1 \
  --reasoning-effort "" --parallel 8 --max-tokens 2048
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
echo done
