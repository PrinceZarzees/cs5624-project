#!/bin/bash
#SBATCH --job-name=gh_final_e10
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# E10: language x library 2x2 on test_small and test_large.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E10_lang_lib; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8080 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.60 --max-num-seqs 8 \
  --seed 51 --disable-log-requests > "$OUT/logs/vllm.log" 2>&1 &
PID=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8080 420 || { tail -60 "$OUT/logs/vllm.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8080/v1 --max-tokens 768 --request-timeout 180"
run_cell () {
  local name="$1" harness="$2" rep="$3"
  for SPLIT in test_small; do
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$harness" --split "$SPLIT" \
      --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$OUT/cells/${name}/rep${rep}/${SPLIT}" $M
  done
}
run_cell py_lib    baseline_b_v2 1
run_cell py_nolib  "$OUT/harnesses/baseline_b_v2_py_nolib.json" 1
run_cell c_nolib   experiments/review_c_backend/harnesses/baseline_b_v2_c.json 1
run_cell c_lib     "$OUT/harnesses/baseline_b_v2_c_lib.json" 1
run_cell py_nolib  "$OUT/harnesses/baseline_b_v2_py_nolib.json" 2
run_cell c_lib     "$OUT/harnesses/baseline_b_v2_c_lib.json" 2
kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
echo done
