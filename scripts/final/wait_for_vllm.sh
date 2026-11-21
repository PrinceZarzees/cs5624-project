#!/bin/bash
# wait_for_vllm.sh HOST PORT TIMEOUT_S
set -u
host=${1:?}; port=${2:?}; timeout=${3:-300}
for i in $(seq 1 "$timeout"); do
  curl -sf "http://${host}:${port}/v1/models" >/dev/null 2>&1 && exit 0
  sleep 1
done
echo "vLLM not ready at ${host}:${port}" >&2
exit 1
