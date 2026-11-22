#!/bin/bash
#SBATCH --job-name=gh_final_e7
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# E7: train a learned router on val, then compare global/rule/learned/decomposed
# on 120 disconnected graphs and on standard medium/large.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E7_router; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8060 \
  --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.60 --max-num-seqs 16 \
  --seed 41 --disable-log-requests > "$OUT/logs/vllm.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8060 420 || { tail -60 "$OUT/logs/vllm.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8060/v1 --max-tokens 768 --request-timeout 180"
LIB=experiments/review_router/harnesses

# 1) library harnesses on val, for router training labels
if [ ! -f "$OUT/router.json" ]; then
  for H in baseline_b_v2 sparse dense tree large; do
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$LIB/$H.json" --split val \
      --max-steps 8 --workers 8 --no-resume --no-trajectories --out "$OUT/val/$H" $M
  done
  "$CLIENT_PYTHON" -m graphharness.learned_router --scores "$OUT/val/*/scores.jsonl" \
    --split val --out "$OUT/router.json"
fi

# 2) modes on the 120-problem disconnected set
"$CLIENT_PYTHON" -m graphharness.runner --harness "$LIB/baseline_b_v2.json" \
  --split disconnected120 --data-dir data_final --max-steps 8 --workers 8 \
  --no-resume --no-trajectories --out "$OUT/disc/global" $M
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --mode routed --max-steps 8 --out "$OUT/disc/rule" $M
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --mode routed --learned-router "$OUT/router.json" --max-steps 8 --out "$OUT/disc/learned" $M
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --mode decomposed --repair-components --max-steps 8 --out "$OUT/disc/decomp_rule" $M
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --mode decomposed --repair-components --learned-router "$OUT/router.json" \
  --max-steps 8 --out "$OUT/disc/decomp_learned" $M

# 3) learned router on standard connected splits (no decomposition)
for S in test_medium test_large; do
  "$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split "$S" \
    --mode routed --max-steps 8 --out "$OUT/$S/rule" $M
  "$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split "$S" \
    --mode routed --learned-router "$OUT/router.json" --max-steps 8 --out "$OUT/$S/learned" $M
done
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
echo done
