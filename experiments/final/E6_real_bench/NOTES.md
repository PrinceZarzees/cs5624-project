# E6 real benchmarks

- GraphArena easy/hard: 100 problems/task (1,000 per split).
- GraphQA test: 50/source task (600 records).
- Harnesses: Baseline A, B, v2; 1 run each, Qwen3-4B BF16, 8k context.
- Every converted record was truth-scored before running (`bad=0`).
- The original 20/task ids are a subset of the new data. v2 on that subset:
  76.5 / 62.5 / 90.4%, versus 74.5 / 61.0 / 88.3% in the frozen run
  (within 2 points, same ranking).
- GraphQA max-flow and node-classification tasks use source-answer truths;
  see the per-task CSV before interpreting those rows.
