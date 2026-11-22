#!/bin/bash
#SBATCH --job-name=cs5624_opt
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
#   MODEL=... BASE_URL=... sbatch scripts/run_optimizer.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${START:=baseline_b}"
: "${ROUNDS:=15}"
: "${MODEL:?set MODEL to the model name served by the endpoint}"
: "${BASE_URL:=${GRAPH_HARNESS_BASE_URL:-http://127.0.0.1:8000/v1}}"
: "${WORKERS:=8}"
: "${TRAIN_PER_TASK:=6}"
: "${VAL_PER_TASK:=5}"
: "${OUT:=experiments/optimizer}"

export PYTHONUNBUFFERED=1
export GRAPH_HARNESS_MODEL="${MODEL}"
export GRAPH_HARNESS_BASE_URL="${BASE_URL}"

echo "start=${START} rounds=${ROUNDS} model=${MODEL} out=${OUT}"
python3 -m graphharness.optimizer \
  --start "${START}" --rounds "${ROUNDS}" --out "${OUT}" \
  --model "${MODEL}" --base-url "${BASE_URL}" --workers "${WORKERS}" \
  --train-per-task "${TRAIN_PER_TASK}" --val-per-task "${VAL_PER_TASK}" --cache
