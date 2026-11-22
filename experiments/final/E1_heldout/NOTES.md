# E1 fresh held-out split

- Split: `test_small_fresh`, seed 56240, 30/task x 5 tasks = 150, |V| 4-50, 0 overlap with the six frozen splits.
- 3 repeats per harness, Qwen3-4B BF16, 8k context, temperature 0.
- McNemar B vs v2 repeat 1: B-only 7, v2-only 7, p=1.000
- McNemar B vs v2 repeat 2: B-only 3, v2-only 7, p=0.344
- McNemar B vs v2 repeat 3: B-only 5, v2-only 10, p=0.302
