#!/bin/bash
# Phase-1 HDA grid on ARC: serves gpt-oss-120b with vLLM on one GPU and runs the grid against it.
# [VERIFY] partition/account/qos names and the vLLM module/venv on ARC before submitting.
#SBATCH --job-name=hda-grid
#SBATCH --partition=a100_normal_q        # [VERIFY] the B200 partition used for earlier jobs
#SBATCH --account=psgroup                # [VERIFY]
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=runs/hda/slurm-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ${HDA_VENV:-.venv}/bin/activate                    # [VERIFY] env with vllm + networkx + openai
MODEL=${HDA_MODEL:-openai/gpt-oss-120b}
MODEL_PATH=${HDA_MODEL_PATH:-$MODEL}                      # e.g. /projects/psgroup/saminrk/gh-gptoss/... [VERIFY]
PORT=${HDA_PORT:-8011}
OUT=${HDA_OUT:-runs/hda/main}
vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --port $PORT --max-model-len 65536 \
     --gpu-memory-utilization 0.90 > runs/hda/vllm-${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
until curl -sf http://localhost:$PORT/v1/models >/dev/null; do sleep 10; done
python -m hda.grid --base-url http://localhost:$PORT/v1 --model "$MODEL" --out "$OUT" \
       --conditions ${HDA_CONDITIONS:-given hidden} --seeds ${HDA_SEEDS:-0 1 2} --k ${HDA_K:-10} \
       --n-iter ${HDA_NITER:-5} --steps ${HDA_STEPS:-8} --reasoning-effort ${HDA_EFFORT:-medium} \
       --parallel ${HDA_PARALLEL:-16} ${HDA_EXTRA:-}
kill $VLLM_PID
