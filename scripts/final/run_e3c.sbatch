#!/bin/bash
#SBATCH --job-name=gh_final_e3c
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_short
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# E3c: deliberately overfit-prone acceptance (10 training problems) for
# SkillOpt-style and RRSI-lite, then evaluate every accepted checkpoint on val
# and test_large.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
: "${VENV:?}"; : "${ACTOR_PATH:?}"; : "${CRITIC_PATH:?}"; : "${CLIENT_PYTHON:?}"
ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/final/E3_probe; mkdir -p "$OUT/logs"
export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
module load cuda13.1/toolkit/13.1.1
export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$ACTOR_PATH" --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8040 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.32 --max-num-seqs 8 --seed 11 --disable-log-requests \
  > "$OUT/logs/actor.log" 2>&1 &
A=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8040 420 || { tail -80 "$OUT/logs/actor.log"; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$VENV/bin/vllm" serve "$CRITIC_PATH" --served-model-name qwen2.5-7b \
  --host 127.0.0.1 --port 8041 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.55 --max-num-seqs 8 --seed 12 --disable-log-requests \
  > "$OUT/logs/critic.log" 2>&1 &
C=$!
scripts/final/wait_for_vllm.sh 127.0.0.1 8041 420 || { tail -80 "$OUT/logs/critic.log"; exit 1; }
M="--model qwen3-4b --base-url http://127.0.0.1:8040/v1 --max-tokens 768 --request-timeout 180"
MC="--critic-model qwen2.5-7b --critic-base-url http://127.0.0.1:8041/v1"
"$CLIENT_PYTHON" -m graphharness.optimizer --start baseline_b --rounds 8 \
  --train-per-task 4 --val-per-task 2 --val-split train \
  --out "$OUT/skillopt_probe" $M $MC --cache "$OUT/cache.sqlite"
"$CLIENT_PYTHON" -m graphharness.optimizer --start baseline_b --rrs --rounds 8 \
  --train-per-task 4 --val-per-task 2 --val-split train \
  --out "$OUT/rrsi_probe" $M $MC --cache "$OUT/cache.sqlite"

# Evaluate every accepted checkpoint plus the start harness on val and test_large.
for RUN in skillopt_probe rrsi_probe; do
  for SPLIT in val test_large; do
    "$CLIENT_PYTHON" -m graphharness.runner --harness baseline_b --split "$SPLIT" \
      --max-steps 8 --workers 8 --no-resume --no-trajectories \
      --out "$OUT/$RUN/start/$SPLIT" $M
    for CKPT in "$OUT/$RUN"/checkpoints/*.json; do
      [ -e "$CKPT" ] || continue
      NAME=$(basename "$CKPT" .json)
      "$CLIENT_PYTHON" -m graphharness.runner --harness "$CKPT" --split "$SPLIT" \
        --max-steps 8 --workers 8 --no-resume --no-trajectories \
        --out "$OUT/$RUN/$NAME/$SPLIT" $M
    done
  done
done
kill $A $C 2>/dev/null || true; wait $A $C 2>/dev/null || true
echo done
