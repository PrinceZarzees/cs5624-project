#!/bin/bash
#SBATCH --job-name=gh_final_e5
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# E5: oracle labels for every pooled harness + 4B/7B rubric judges.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${ACTOR_PATH:?}"; : "${CRITIC_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E5_proxy; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$ACTOR_PATH" --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8070 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.32 --max-num-seqs 16 --seed 31 --disable-log-requests \
  > "$OUT/logs/actor.log" 2>&1 &
A=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8070 420 || { tail -80 "$OUT/logs/actor.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8070/v1 --max-tokens 768 --request-timeout 180"

# Oracle labels. val and proxy_large at 10/task, proxy_medium at 3/task.
while IFS=, read -r ID _rest; do
  [ "$ID" = "id" ] && continue
  H="$OUT/pool/$ID.json"
  for spec in "proxy_large:data_proxy_large:10" "val:data:10" "proxy_medium:data_proxy_medium:3"; do
    SPLIT="${spec%%:*}"; REST="${spec#*:}"; DATA="${REST%%:*}"; K="${REST##*:}"
    "$CLIENT_PYTHON" -m graphharness.runner --harness "$H" --split "$SPLIT" --data-dir "$DATA" \
      --limit-per-task "$K" --max-steps 8 --workers 12 --no-resume --no-trajectories \
      --out "$OUT/runs/$ID/${SPLIT}_k$K" $M
  done
done < "$OUT/pool.csv"

# Rubric judges (4B and 7B).
"$CLIENT_PYTHON" scripts/final/e5_judge.py --pool "$OUT/pool.csv" \
  --model qwen3-4b --base-url http://127.0.0.1:8070/v1 --out "$OUT/judge_4b.json"
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$CRITIC_PATH" --served-model-name qwen2.5-7b \
  --host 127.0.0.1 --port 8071 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.55 --max-num-seqs 8 --seed 32 --disable-log-requests \
  > "$OUT/logs/critic.log" 2>&1 &
C=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8071 420 || { tail -80 "$OUT/logs/critic.log"; exit 1; }
"$CLIENT_PYTHON" scripts/final/e5_judge.py --pool "$OUT/pool.csv" \
  --model qwen2.5-7b --base-url http://127.0.0.1:8071/v1 --out "$OUT/judge_7b.json"
kill $A $C 2>/dev/null || true; wait $A $C 2>/dev/null || true
echo done
