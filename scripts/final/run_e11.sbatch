#!/bin/bash
#SBATCH --job-name=gh_final_e11
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# E11: no-tools harnesses with a 32k context, to separate "no tools" from
# "graph does not fit in 8k".
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E11_notools_longctx; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.80 --max-num-seqs 4 \
  --seed 1 --disable-log-requests > "$OUT/logs/vllm32k.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8000 420 || { tail -80 "$OUT/logs/vllm32k.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8000/v1 --max-tokens 768 --request-timeout 300"
for H in baseline_a_text baseline_b_v2_text; do
  for S in test_medium test_large; do
    "$CLIENT_PYTHON" -m graphharness.runner \
      --harness "experiments/review_tool_access/harnesses/${H}.json" --split "$S" \
      --max-steps 8 --workers 4 --no-resume --no-trajectories \
      --out "$OUT/$H/$S" $M
    "$VENV/bin/python" scripts/final/e11_prompt_sizes.py \
      --harness "experiments/review_tool_access/harnesses/${H}.json" --split "$S" \
      --data-dir data --run "$OUT/$H/$S" --tokenizer "$MODEL_PATH" \
      --out "$OUT/fit_rows.csv"
  done
done
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
