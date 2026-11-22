# E11 no-tools at 32k

- Actor: Qwen3-4B BF16, `--max-model-len 32768`, 4 workers, 8-step limit.
- Runs: `baseline_a_text` and `baseline_b_v2_text` on test_medium (150) and
  test_large (100), 1 run each.
- No prompt on test_large fits in 8k (max prompt 40,189 tokens); 7/100 also
  exceed 32k and always fail.
- Medium prompts mostly fit in 8k (141/150 for A, 137/150 for v2), so the low
  no-tools medium accuracy is not a context artifact.
- `fit_rows.csv` has per-problem prompt tokens and correctness.
