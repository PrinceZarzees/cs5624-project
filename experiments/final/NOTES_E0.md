# E0 notes

- Integrity: build_data reproduces the committed manifest; verify_data passes all
  six splits; pytest = 174 passed, 0 skipped.
- Benchmark data present: `data/grapharena_eval/{ga_easy,ga_hard}.jsonl` (400
  records) and `data/graphqa_eval/graphqa_{train,validation,test}.jsonl` (660).
- Actor: Qwen3-4B-Instruct-2507 snapshot
  `cdbee75f17c01a7cc42f958dc650907174af0554`, BF16, vLLM 0.34.0.
- Critic for E3: Qwen2.5-7B-Instruct snapshot
  `a09a35458c702b33eeacc393d103063234e8bc28` (no larger model on disk).

## Smoke test (CPU-Q4 fallback)

`baseline_b_v2`, 25 `test_small` problems (5/task):
accuracy 96.0%, parsed 100%, feasible 96%, tokens 106,913, wall 418s,
~256 tokens/s aggregate, ~16.7 s/problem, one worker.
The GPU smoke is part of E1/E2 when job 1546237 starts; record its throughput
there.
