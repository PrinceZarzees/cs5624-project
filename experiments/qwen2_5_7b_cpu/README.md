# Qwen2.5-7B CPU full baseline

Full `test_small` run of the 7B CPU fallback, launched as an independent Slurm
job (`1537117`).

- Model: `Qwen/Qwen2.5-7B-Instruct` Q4_K_M GGUF (`bartowski`)
- Engine: llama-cpp-python 0.3.37, ChatML
- Hardware: 48 CPUs / 256 GB on `normal_q`
- Harness: `baseline_b_v2` (13 rules)
- Split: full `test_small`, 150 problems
- Max steps: 8 plus one final-answer turn

## Result

**77.3% accuracy** on 150 problems.

| Task | Accuracy |
|---|---:|
| shortest_path | 86.7% |
| connected_components | 80.0% |
| cycle_detection | 66.7% |
| diameter | 60.0% |
| max_clique | 93.3% |

A second full run with the 14-rule v2 scored **76.0%** (cycles 73.3%, diameter
53.3%, max clique 76.7%), slightly below the 13-rule run. The 7B is slower
(~12 tokens/s) and less accurate than the 4B Qwen3 model on this benchmark
(77.3%/76.0% vs 93.3%/94.0%), so Qwen3-4B-Instruct-2507 remains primary.

Artifacts:

```text
baseline_b_v2/test_small/       predictions, scores, summary, trajectories
logs/                            llama.cpp server log and Slurm output
failure_analysis.txt             task-level failure notes
```

Reproduce with `scripts/run_llama_cpp_cpu_pipeline.sbatch` and
`HARNESS=baseline_b_v2 SPLIT=test_small`.
