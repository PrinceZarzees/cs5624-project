#!/bin/bash
#SBATCH --job-name=gh_final_remaining
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# One GPU, sequential: E7 (learned router + decomposition), E10 (language x
# library), then E5 oracle labels and rubric judges.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${ACTOR_PATH:?}"; : "${CRITIC_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
export GRAPH_HARNESS_SANDBOX_DIR=experiments/final/sandbox_remaining
mkdir -p experiments/final/sandbox_remaining experiments/final/logs

CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$ACTOR_PATH" --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8090 --dtype bfloat16 --max-model-len 16384 \
  --gpu-memory-utilization 0.50 --max-num-seqs 16 --seed 61 --disable-log-requests \
  > experiments/final/logs/remaining_actor.log 2>&1 &
A=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8090 420 || { tail -80 experiments/final/logs/remaining_actor.log; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8090/v1 --max-tokens 768 --request-timeout 180"

# ---------------- E7 ----------------
E7=experiments/final/E7_router; LIB=experiments/review_router/harnesses; mkdir -p "$E7/logs"
for H in baseline_b_v2 sparse dense tree large; do
  "$CLIENT_PYTHON" -m graphharness.runner --harness "$LIB/$H.json" --split val \
    --max-steps 8 --workers 12 --no-resume --no-trajectories --out "$E7/val/$H" $M
done
"$CLIENT_PYTHON" -m graphharness.learned_router --scores "$E7/val/*/scores.jsonl" \
  --split val --out "$E7/router.json"
"$CLIENT_PYTHON" -m graphharness.runner --harness "$LIB/baseline_b_v2.json" \
  --split disconnected120 --data-dir data_final --max-steps 8 --workers 12 \
  --no-resume --no-trajectories --out "$E7/disc/global" $M
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --mode routed --max-steps 8 --out "$E7/disc/rule" $M
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --mode routed --learned-router "$E7/router.json" --max-steps 8 --out "$E7/disc/learned" $M
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --mode decomposed --repair-components --max-steps 8 --out "$E7/disc/decomp_rule" $M
"$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split disconnected120 --data-dir data_final \
  --mode decomposed --repair-components --learned-router "$E7/router.json" \
  --max-steps 8 --out "$E7/disc/decomp_learned" $M
# learned vs rule on test_medium only (time)
for MODE in rule learned; do
  EXTRA=""; [ "$MODE" = "learned" ] && EXTRA="--learned-router $E7/router.json"
  "$CLIENT_PYTHON" -m graphharness.router --library "$LIB" --split test_medium \
    --mode routed $EXTRA --max-steps 8 --out "$E7/test_medium/$MODE" $M
done

# ---------------- E10 ----------------
E10=experiments/final/E10_lang_lib; mkdir -p "$E10/logs"
run_cell () {
  local name="$1" harness="$2"
  for SPLIT in test_small test_large; do
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$harness" --split "$SPLIT" \
      --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$E10/cells/${name}/$SPLIT" $M
  done
}
run_cell py_lib    baseline_b_v2
run_cell py_nolib  "$E10/harnesses/baseline_b_v2_py_nolib.json"
run_cell c_nolib   experiments/review_c_backend/harnesses/baseline_b_v2_c.json
run_cell c_lib     "$E10/harnesses/baseline_b_v2_c_lib.json"

# ---------------- E5 ----------------
E5=experiments/final/E5_proxy; mkdir -p "$E5/logs"
while IFS=, read -r ID _rest; do
  [ "$ID" = "id" ] && continue
  H="$E5/pool/$ID.json"
  [ -f "$H" ] || continue
  for spec in "proxy_large:data_proxy_large:10" "val:data:10" "proxy_medium:data_proxy_medium:3"; do
    SPLIT="${spec%%:*}"; REST="${spec#*:}"; DATA="${REST%%:*}"; K="${REST##*:}"
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$H" --split "$SPLIT" --data-dir "$DATA" \
      --limit-per-task "$K" --max-steps 8 --workers 12 --no-resume --no-trajectories \
      --out "$E5/runs/$ID/${SPLIT}_k$K" $M
  done
done < "$E5/pool.csv"
"$CLIENT_PYTHON" scripts/final/e5_judge.py --pool "$E5/pool.csv" \
  --model qwen3-4b --base-url http://127.0.0.1:8090/v1 --out "$E5/judge_4b.json"

CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$CRITIC_PATH" --served-model-name qwen2.5-7b \
  --host 127.0.0.1 --port 8091 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.42 --max-num-seqs 8 --seed 62 --disable-log-requests \
  > experiments/final/logs/remaining_critic.log 2>&1 &
C=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8091 420 || { tail -80 experiments/final/logs/remaining_critic.log; exit 1; }
"$CLIENT_PYTHON" scripts/final/e5_judge.py --pool "$E5/pool.csv" \
  --model qwen2.5-7b --base-url http://127.0.0.1:8091/v1 --out "$E5/judge_7b.json"
kill $A $C 2>/dev/null || true; wait $A $C 2>/dev/null || true
echo done
