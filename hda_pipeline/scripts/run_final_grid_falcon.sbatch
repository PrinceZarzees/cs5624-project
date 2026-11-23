#!/bin/bash
#SBATCH --job-name=hda_final_grid
#SBATCH --partition=a30_normal_q
#SBATCH --qos=fal_a30_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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
OUT=experiments/hda/optimizers
mkdir -p "$OUT/final"
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8011 \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 32 \
  --disable-log-requests > "$OUT/final/vllm_${SLURM_JOB_ID}.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8011 1200 || { tail -60 "$OUT/final/vllm_${SLURM_JOB_ID}.log"; exit 1; }
cd hda_pipeline
for P in ${HDA_PROMPTS:-unoptimized}; do
  echo "=== prompt $P ==="
  HDA_PROMPT_FILE="../$OUT/prompts/$P.json" "$CLIENT_PYTHON" -m hda.grid \
    --conditions given hidden --seeds 0 1 2 --k 5 --n-iter 3 --steps 8 \
    --temperature 0.2 --max-tokens 4096 --reasoning-effort "" --parallel 16 \
    --data data/hda --out "../$OUT/final/$P" \
    --model qwen3-4b --base-url http://127.0.0.1:8011/v1
done
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
