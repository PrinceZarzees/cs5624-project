#!/bin/bash
#SBATCH --job-name=gh_final_e10e5
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# E10 2x2 on test_small + E5 oracle labels for 40 candidates + rubric judges.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CRITIC_PATH:?}"; : "${CLIENT_PYTHON:?}"
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

# ---------------- E5 (40 candidates) ----------------
E5=experiments/final/E5_proxy; mkdir -p "$E5/logs"
while IFS=, read -r ID _rest; do
  [ "$ID" = "id" ] && continue
  H="$E5/pool/$ID.json"; [ -f "$H" ] || continue
  for spec in "proxy_large:data_proxy_large:10" "val:data:10" "proxy_medium:data_proxy_medium:3"; do
    SPLIT="${spec%%:*}"; REST="${spec#*:}"; DATA="${REST%%:*}"; K="${REST##*:}"
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$H" --split "$SPLIT" --data-dir "$DATA" \
      --limit-per-task "$K" --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$E5/runs/$ID/${SPLIT}_k$K" $M
  done
done < "$E5/pool40.csv"
"$CLIENT_PYTHON" scripts/final/e5_judge.py --pool "$E5/pool40.csv" \
  --model qwen3-4b --base-url http://127.0.0.1:8080/v1 --out "$E5/judge_4b.json"
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$CRITIC_PATH" --served-model-name qwen2.5-7b \
  --host 127.0.0.1 --port 8091 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.42 --max-num-seqs 8 --seed 72 --disable-log-requests \
  > "$E5/logs/critic.log" 2>&1 &
C=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8091 420 || { tail -60 "$E5/logs/critic.log"; exit 1; }
"$CLIENT_PYTHON" scripts/final/e5_judge.py --pool "$E5/pool40.csv" \
  --model qwen2.5-7b --base-url http://127.0.0.1:8091/v1 --out "$E5/judge_7b.json"
kill $PID $C 2>/dev/null || true; wait $PID $C 2>/dev/null || true
echo done
