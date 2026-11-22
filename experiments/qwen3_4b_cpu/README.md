# Qwen3-4B CPU fallback

Same model family as the primary run, quantized for CPU while B200 GPUs are
queued.

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- GGUF: `unsloth/Qwen3-4B-Instruct-2507-GGUF`, `Q4_K_M`
- Engine: llama-cpp-python 0.3.37, ChatML
- Hardware: 96-core `normal_q` node, no GPU
- Throughput: ~18 tokens/s
- Harness: `baseline_b_v2` (13 rules)
- Split: full `test_small` (150 problems)
- Max agent steps: 8, plus one final-answer turn
- Result: **93.3%** on 150 problems (shortest path 100%, components 100%,
  cycles 100%, diameter 86.7%, max clique 80%)
- Failures: 10/150; mostly hallucinated node ids on tiny diameter graphs and
  six max-clique cases

Command uses `--workers 1` because the CPU server serializes requests.

Reproduce with:

```bash
LLAMA_VENV=/path/to/llama-venv \
MODEL_GGUF=/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
MODEL_ALIAS=qwen3-4b-q4 \
CLIENT_PYTHON=/path/to/python-with-graphharness \
OUT=/path/to/experiments/qwen3_4b_cpu \
sbatch scripts/run_llama_cpp_cpu_pipeline.sbatch
```

Artifacts:

```text
baseline_b_v2/test_small/       predictions, scores, summary, trajectories
```

## Other runs

- Full `baseline_a`: **84.7%** (`baseline_a/test_small/`).
- Targeted diameter recheck of the four previous failures with the 14-rule v2:
  **4/4 correct** (`diameter_rule_check/`). This confirms the
  generator-materialization rule.

## Logs

`logs/` contains the llama.cpp server log and the two full-run logs. The
server was started interactively on owl085; the shared model and venv paths are
listed in `../REPRODUCIBILITY.md`.
