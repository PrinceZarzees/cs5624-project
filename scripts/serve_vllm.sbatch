#!/bin/bash
#SBATCH --job-name=gh_vllm
#SBATCH --partition=b200_normal_q
#SBATCH --qos=owl_b200_normal_base
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Start an OpenAI-compatible vLLM server. Submit from the repository root:
#
#   VENV=/path/to/vllm-venv \
#   MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507/snapshot \
#   sbatch scripts/serve_vllm.sbatch
#
# The model and vLLM environment must live on shared storage visible to compute
# nodes (for ARC, /projects or /home, not node-local scratch).
#
# Then point graphharness at the node printed in the job log:
#   python -m graphharness.runner ... --base-url http://<node>:8000/v1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${VENV:?set VENV to a Python 3.12 venv with vllm installed}"
: "${MODEL_PATH:?set MODEL_PATH to the model snapshot directory}"
: "${SERVED_NAME:=local-model}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.60}"
: "${MAX_NUM_SEQS:=8}"
: "${CUDA_MODULE:=cuda13.1/toolkit/13.1.1}"
: "${HF_HOME:=${HOME}/.cache/huggingface}"

module load "${CUDA_MODULE}"
export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}"
export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"
export HF_HOME
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
echo "model=${MODEL_PATH}"
echo "url=http://$(hostname):${PORT}/v1"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvcc --version | tail -1

exec "${VENV}/bin/vllm" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_NAME}" \
  --host "${HOST}" --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --disable-log-requests
