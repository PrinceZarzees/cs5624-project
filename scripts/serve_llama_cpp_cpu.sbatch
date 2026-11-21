#!/bin/bash
#SBATCH --job-name=gh_llama_cpu
#SBATCH --partition=normal_q
#SBATCH --qos=owl_normal_base
#SBATCH --account=psgroup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# CPU fallback server for Qwen2.5-7B-Instruct Q4_K_M (GGUF).
# Useful when B200 GPUs are queued.
#
#   LLAMA_VENV=/path/to/llama-cpp-venv \
#   MODEL_GGUF=/path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
#   sbatch scripts/serve_llama_cpp_cpu.sbatch
#
# Then use --base-url http://<node>:8001/v1 with graphharness.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

: "${LLAMA_VENV:?set LLAMA_VENV to a venv with llama-cpp-python[server]}"
: "${MODEL_GGUF:?set MODEL_GGUF to the .gguf file}"
: "${MODEL_ALIAS:=qwen2.5-7b}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8001}"
: "${N_CTX:=8192}"
: "${N_THREADS:=64}"
: "${N_THREADS_BATCH:=96}"
: "${CHAT_FORMAT:=chatml}"
export PYTHONUNBUFFERED=1

echo "host=$(hostname) model=${MODEL_GGUF}"
echo "url=http://$(hostname):${PORT}/v1"
nproc

exec "${LLAMA_VENV}/bin/python" -m llama_cpp.server \
  --model "${MODEL_GGUF}" \
  --model_alias "${MODEL_ALIAS}" \
  --host "${HOST}" --port "${PORT}" \
  --n_ctx "${N_CTX}" --n_batch 512 \
  --n_threads "${N_THREADS}" --n_threads_batch "${N_THREADS_BATCH}" \
  --chat_format "${CHAT_FORMAT}"
