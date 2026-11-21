#!/bin/bash
#SBATCH --job-name=cs5624_scale
#SBATCH --partition=a30_normal_q
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Submit from the repository root:
#   MODEL=... BASE_URL=... HARNESSES="baseline_a baseline_b" \
#     sbatch scripts/run_scaling.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${HARNESSES:=baseline_a baseline_b}"
: "${SPLITS:=test_small test_medium test_large test_stress}"
: "${MODEL:?set MODEL to the model name served by the endpoint}"
: "${BASE_URL:=${GRAPH_HARNESS_BASE_URL:-http://127.0.0.1:8000/v1}}"
: "${WORKERS:=8}"
: "${LIMIT_PER_TASK:=}"
: "${OUT:=experiments/scaling}"

export PYTHONUNBUFFERED=1
export GRAPH_HARNESS_MODEL="${MODEL}"
export GRAPH_HARNESS_BASE_URL="${BASE_URL}"

read -r -a HARNESS_ARRAY <<< "${HARNESSES}"
read -r -a SPLIT_ARRAY <<< "${SPLITS}"
ARGS=(--harnesses "${HARNESS_ARRAY[@]}" --splits "${SPLIT_ARRAY[@]}"
      --out "${OUT}" --model "${MODEL}" --base-url "${BASE_URL}"
      --workers "${WORKERS}" --cache)
if [[ -n "${LIMIT_PER_TASK}" ]]; then
  ARGS+=(--limit-per-task "${LIMIT_PER_TASK}")
fi

echo "harnesses=${HARNESSES} splits=${SPLITS} model=${MODEL}"
python3 -m graphharness.scaling "${ARGS[@]}"
