# Qwen2.5-7B CPU fallback

This pilot exists because B200 capacity was unavailable. It is a **quantized,
CPU-only fallback**, not a replacement for the BF16 GPU comparison.

- Model: `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0, 7.6B params)
- Quantization: Q4_K_M GGUF from `bartowski/Qwen2.5-7B-Instruct-GGUF`
- Engine: llama-cpp-python 0.3.37
- Hardware: one 96-core `normal_q` node, no GPU
- Observed throughput: ~12 tokens/s
- Split: `test_small`, 5 problems per task
- Harness: `baseline_b_v2`
- Max agent steps: 8

## Tooling fixes

The first 25-problem run scored 20%. Two harness/tooling bugs caused most of
the loss:

1. Qwen2.5 emitted Python-style triple-quoted strings inside JSON tool calls;
   strict JSON rejected them. Parsing now falls back to `ast.literal_eval`.
2. The model sometimes put a guessed `<answer>` next to an unexecuted code
   block. The agent now runs the code first and only accepts an answer after
   seeing the observation.

The sandbox also echoes a bare final expression, so code like
`list(nx.connected_components(G))` produces visible output without `print(...)`.

## Results

| Run | Problems | Accuracy | Notes |
|---|---:|---:|---|
| Before fixes | 25 | 20% | parser/answer issues |
| After fixes, 10-problem check | 10 | 60% | |
| 25-problem run, old v2 rules | 25 | 68% | diameter 40%, max clique 20% |
| Weak-task check with new v2 rules | 10 | 80% | diameter 80%, max clique 80% |
| Final 25-problem run | 25 | see `baseline_b_v2_final25/summary.txt` | new v2 rules |

The 7B CPU fallback is competitive with the 4B GPU pilot on overall accuracy,
but slower and weaker on the NP-hard clique task before the explicit
`nx.max_weight_clique` rule was added.

## Reproduce

Start the CPU server with `scripts/serve_llama_cpp_cpu.sbatch`, then:

```bash
python -m graphharness.runner \
  --harness baseline_b_v2 --split test_small --limit-per-task 5 \
  --max-steps 8 --workers 1 --model qwen2.5-7b \
  --base-url http://<node>:8001/v1 \
  --out experiments/qwen2_5_7b_cpu_pilot/baseline_b_v2_final25
```
