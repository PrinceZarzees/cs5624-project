# Qwen3-4B pilot

First real-model run for the project.

- Model: `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, 4.02B params)
- Serving: vLLM 0.34.0, Python 3.12, one NVIDIA B200, 8192 context
- Split: `test_small`, 5 problems per task (25 per harness)
- Harnesses: `baseline_a`, `baseline_b`
- Decoding: temperature 0, `max_tokens=512`, up to 8 agent steps

## Results

| Harness | Accuracy | Parsed | Feasible | Tokens |
|---|---:|---:|---:|---:|
| Baseline A | 20% | 100% | 20% | 66,211 |
| Baseline B | 72% | 100% | 84% | 100,305 |

By task:

| Task | Baseline A | Baseline B |
|---|---:|---:|
| shortest_path | 0/5 | 5/5 |
| connected_components | 0/5 | 1/5 |
| cycle_detection | 5/5 | 5/5 |
| diameter | 0/5 | 2/5 |
| max_clique | 0/5 | 5/5 |

## Observation

Baseline A frequently invents node ids (`0,1,2`, `a,b,c`) and does not use the
sandbox. Baseline B follows the tool protocol and solves three tasks, but its
connected-components answers are often hallucinated and its diameter search
sometimes returns a path between a non-diametral pair.

## Reproduce

Start the model with `scripts/serve_vllm.sbatch`, then:

```bash
python -m graphharness.runner \
  --harness baseline_b --split test_small --limit-per-task 5 \
  --max-steps 8 --workers 8 --model qwen3-4b \
  --base-url http://<node>:8000/v1 \
  --out experiments/qwen3_4b_pilot/baseline_b
```

The stored `summary.json`, `scores.jsonl`, `predictions.jsonl`, and
`trajectories/` are the exact artifacts from the pilot.

## One optimization round

`optimizer_1round/` contains a real one-round SkillOpt/GEPA-style pilot with
Qwen3-4B as both actor and critic:

- Initial validation accuracy: 70%.
- Candidate validation accuracy: 40%; rejected.
- Best harness remained `baseline_b`.

The critic's main proposal was to switch to `graph_view: "edge_list"` and add
more verification rules. The edge-list change is exactly the kind of
small-graph overfitting the project is designed to detect: it makes prompts
longer and does not help at scale, and validation accuracy dropped. The
rejected candidate is preserved under `optimizer_1round/candidates/round_01.json`.

Evaluating the frozen best harness on the 25-problem `test_small` pilot gave
68% accuracy, within sampling noise of the 72% Baseline B pilot. A meaningful
optimization result will need more rounds and a larger validation sample.

