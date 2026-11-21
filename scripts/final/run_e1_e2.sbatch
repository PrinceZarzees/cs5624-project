#!/bin/bash
#SBATCH --job-name=gh_final_e1e2
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
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen
OUT=experiments/final
mkdir -p "$OUT/logs" "$OUT/E1_heldout" "$OUT/E2_repeats" "$OUT/E2_tool_access"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.60 --max-num-seqs 8 \
  --seed 1 --disable-log-requests > "$OUT/logs/e1e2_vllm.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8000 240 || { tail -100 "$OUT/logs/e1e2_vllm.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8000/v1 --max-tokens 768 --request-timeout 180"

# E1: fresh held-out small split, 3 repeats.
for H in baseline_a baseline_b baseline_b_v2; do
  for R in 1 2 3; do
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$H" --split test_small_fresh \
      --data-dir data_final --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$OUT/E1_heldout/${H}/rep${R}" $M
  done
done

# E2: two new scaling repeats per harness/split (existing full scaling = rep 1).
for H in baseline_a baseline_b baseline_b_v2; do
  for R in 2 3; do
    for S in test_small test_medium test_large test_stress; do
      "$CLIENT_PYTHON" -m graphharness.runner --harness "$H" --split "$S" \
        --max-steps 8 --workers 8 --no-resume --no-trajectories \
        --out "$OUT/E2_repeats/${H}/rep${R}/${S}" $M
    done
  done
done

# E2 tool-access repeats (existing review_tool_access = rep 1).
for R in 2 3; do
  for C in baseline_a_tools baseline_a_text baseline_b_v2_tools baseline_b_v2_text; do
    for S in test_small test_medium; do
      "$CLIENT_PYTHON" -m graphharness.runner \
        --harness "experiments/review_tool_access/harnesses/${C}.json" --split "$S" \
        --max-steps 8 --workers 8 --no-resume --no-trajectories \
        --out "$OUT/E2_tool_access/rep${R}/${C}/${S}" $M
    done
    "$CLIENT_PYTHON" -m graphharness.runner \
      --harness "experiments/review_tool_access/harnesses/${C}.json" --split test_large \
      --limit-per-task 5 --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$OUT/E2_tool_access/rep${R}/${C}/test_large" $M
  done
done
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
echo done
