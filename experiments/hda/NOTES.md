# HDA notes

- 2026-11-21: H0 blocked because `/mnt/project-files/hda/hda_pipeline/` is not
  present on ARC and no archive exists under `/projects/psgroup/saminrk`.
- Decision from Samin: supply the tested pipeline; use Qwen3-4B-Instruct-2507
  as the HDA for the first run (H6), because gpt-oss-120b is not on disk and
  B200 capacity is fairshare-limited.
- Environment checked: gcc 11.5 (C++17 OK), torch/vLLM venv present, Qwen3-4B
  BF16 snapshot on disk.

## H0 results (2026-11-21)

- `pytest -q tests`: **18 passed**.
- Data: examples 1200, val 400, test_small 400, medium 300, large 300, xl 200,
  stress 100; build 60 s.
- Human C++ reference: **100%** on every task and split; max CPU 0.45 s
  (diameter stress), below one third of the 5 s limit.
- Identifiability at k=10: shortest_path/components/diameter/common_neighbors/
  triangle/mst/bipartite = 1.00; max_flow 0.98; cycle 0.93; reachability 0.86.
- Qwen3-4B Q4 CPU pilot: most submissions fail to compile (the model omits the
  loader template), so best validation accuracy is 0. This is a model-strength
  result, not a pipeline failure. The BF16 k=10 GPU grid is queued as 1550735.
