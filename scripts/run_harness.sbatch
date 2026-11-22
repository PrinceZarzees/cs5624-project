#!/bin/bash
#SBATCH --job-name=cs5624_harness
#SBATCH --partition=a30_normal_q
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Generic Phase 2/4 run. Submit from the repository root, e.g.:
#   MODEL=... BASE_URL=... HARNESS=baseline_b SPLIT=test_medium \
#     sbatch scripts/run_harness.sbatch
# For a local vLLM endpoint, BASE_URL=http://127.0.0.1:8000/v1 and
# GRAPH_HARNESS_API_KEY=EMPTY are enough.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${HARNESS:=baseline_b}"
: "${SPLIT:=test_small}"
: "${MODEL:?set MODEL to the model name served by the endpoint}"
: "${BASE_URL:=${GRAPH_HARNESS_BASE_URL:-http://127.0.0.1:8000/v1}}"
: "${WORKERS:=8}"
: "${OUT:=experiments/${HARNESS}/${SPLIT}}"
: "${LIMIT_PER_TASK:=}"

export PYTHONUNBUFFERED=1
export GRAPH_HARNESS_MODEL="${MODEL}"
export GRAPH_HARNESS_BASE_URL="${BASE_URL}"

ARGS=(--harness "${HARNESS}" --split "${SPLIT}" --out "${OUT}"
      --model "${MODEL}" --base-url "${BASE_URL}" --workers "${WORKERS}" --cache)
if [[ -n "${LIMIT_PER_TASK}" ]]; then
  ARGS+=(--limit-per-task "${LIMIT_PER_TASK}")
fi

echo "harness=${HARNESS} split=${SPLIT} model=${MODEL} workers=${WORKERS}"
echo "out=${OUT}"
python3 -m graphharness.runner "${ARGS[@]}"
