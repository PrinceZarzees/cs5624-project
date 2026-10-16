# Harness Optimization for Graph Reasoning: Learn Small, Run Big

**CS 5624 — Fall 2026**

## Team
- Bikash Kumar Paul
- Abdullah Al Noman
- Samin Rahman Khan
- Md Zarzees Uddin Shah Chowdhury

## Overview
This project investigates whether an LLM agent harness optimized exclusively on small, computationally cheap graphs can generalize to much larger graphs, or if the learned strategies overfit to small-scale topologies.

See [project_proposal/PROJECT_PLAN.md](project_proposal/PROJECT_PLAN.md) for the full project plan.

## Setup
```bash
pip install -e ".[dev]"      # or: pip install -r requirements.txt
pytest                       # 30 tests, about 1 second
```

## Data (Phase 1)
Build all synthetic splits (takes about 15 seconds, same output on every machine):
```bash
python -m graphharness.build_data
```
This writes `data/<split>.jsonl` (ignored by git) and `data/manifest.json` (committed).
After building, check that your `manifest.json` has no git diff — then you have exactly the same data as everyone else.

| split | graph size (nodes) | problems per task | use |
|---|---|---|---|
| `train` | 4–50 | 60 | harness optimization |
| `val` | 4–50 | 30 | accept/reject harness edits |
| `test_small` | 4–50 | 30 | in-distribution test (small side of the generalization gap) |
| `test_medium` | 100–300 | 30 | test |
| `test_large` | 500–1000 | 20 | test |
| `test_stress` | 2000 | 10 | test |

Sizes and counts are set in `configs/data_splits.json`.

**Tasks** (`src/graphharness/tasks.py`): `shortest_path`, `connected_components`, `cycle_detection`, `diameter` (polynomial) and `max_clique` (NP-hard). Graphs come from Erdős–Rényi, Barabási–Albert and Watts–Strogatz generators, with random node ids so the agent cannot rely on labels like 0..n.

**GraphArena:** the data is [`dataset.zip`](https://drive.google.com/file/d/1NjULkBbM3kS15Q_1YVcPmTQ-DWiNJM9r/view) (~1.4 GB) in the [GraphArena Google Drive folder](https://drive.google.com/drive/folders/1mvJSUTrfOX13wgpkyb3w8s_SJqipnb1c) (linked from [squareRoot3/GraphArena](https://github.com/squareRoot3/GraphArena)). Download, unzip and convert it in one step:
```bash
pip install gdown
python -m graphharness.grapharena_adapter --download
```
If the download fails (Google Drive rate limits, or a network that blocks `drive.google.com`), download `dataset.zip` in a browser and run
`python -m graphharness.grapharena_adapter --zip path/to/dataset.zip` (or `--dataset-dir path/to/unzipped/dataset`).
This converts Distance, Connected, Diameter and MCP into the same format (`data/ga_easy.jsonl`, `data/ga_hard.jsonl`); raw files are kept in `data/grapharena/` (ignored by git).

### Record format
One JSON object per line:
```json
{"id": "shortest_path/train/00000", "source": "synthetic", "task": "shortest_path", "split": "train",
 "graph": {"directed": false, "nodes": [2, 26, 38, ...], "edges": [[2, 38], [38, 26], ...]},
 "params": {"source": 2, "target": 26},
 "question": "Find a shortest path (the fewest edges) from node 2 to node 26. ...",
 "answer": {"length": 2, "path": [2, 38, 26]},
 "meta": {"n_nodes": 5, "n_edges": 4, "family": "ba", ...}}
```
The question does **not** include the graph. How the graph is written into the prompt is part of the harness. `graphharness.encoding.build_prompt(record)` gives a simple default (edge list or adjacency list).

## Scoring answers
The model's final answer is a bracketed list, ideally inside tags: `<answer>[2, 38, 26]</answer>`.
Each answer gets three flags: **parsed** (a list was found), **feasible** (a real path / clique / cycle, no made-up nodes) and **correct** (feasible and optimal).

In Python:
```python
from graphharness import load_split, score_response
from graphharness.encoding import build_prompt

for rec in load_split("val", tasks=["shortest_path"]):
    prompt = build_prompt(rec)            # send to the LLM
    result = score_response(rec, llm_text)
```
From the command line, with `preds.jsonl` holding `{"id": ..., "response": ...}` per line:
```bash
python -m graphharness.evaluate_answer --data data/test_medium.jsonl --predictions preds.jsonl --out scores.jsonl
```
This prints parsed / feasible / accuracy for each task.

## Repository layout
```
src/graphharness/   tasks, generators, data loading, scoring, GraphArena adapter
configs/            data split config
data/               generated JSONL (not in git) + manifest.json
tests/              pytest tests
experiments/        run scripts and results (Phase 2+)
report/             final report (written by the team)
```
