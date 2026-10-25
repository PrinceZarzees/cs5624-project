# CS 5624 Project: Harness Optimization for Graph Reasoning

**Learn Small, Run Big** — Can an LLM agent harness optimized on small graphs generalize to large ones?

## Project Structure

```
cs5624/
├── src/                    # Source code
│   ├── __init__.py
│   ├── graph_generator.py  # Synthetic graph generation (ER, BA, WS)
│   ├── tasks.py            # Task definitions & ground-truth solvers
│   ├── data_loader.py      # JSONL I/O & dataset loading pipeline
│   ├── evaluate_answer.py  # Answer verification & metrics
│   ├── ingest_graphqa.py   # GraphQA benchmark ingestion
│   └── ingest_grapharena.py# GraphArena benchmark ingestion
├── data/                   # Generated & ingested datasets (JSONL)
├── tests/                  # pytest test suite
│   ├── __init__.py
│   └── test_verifiers.py   # Comprehensive verifier tests
├── experiments/            # Experiment logs and results
├── report/                 # Final paper & presentation
├── scripts/                # Slurm job scripts
│   ├── generate_data.sbatch
│   └── run_tests.sbatch
├── .gitignore
└── README.md
```

## Dataset Overview

The benchmark testbed integrates three data sources across varying graph scales:

### 1. Synthetic Graph Benchmark (Controlled Scales)
Generated across three random graph families: Erdős–Rényi ($G(n, p)$), Barabási–Albert, and Watts–Strogatz.

| Scale Split | Node Range ($|V|$) | Graphs | Tasks | Task Types |
| :--- | :--- | :--- | :--- | :--- |
| **Small (Train)** | $N \in [4, 50]$ | 201 | 874 | Shortest Path, Connected Components, Cycle, Max Clique, Hamiltonian Path |
| **Small (Val)** | $N \in [4, 50]$ | 99 | 436 | Shortest Path, Connected Components, Cycle, Max Clique, Hamiltonian Path |
| **Medium (Test 1)** | $N \in [100, 300]$ | 60 | 240 | Shortest Path, Connected Components, Cycle, Max Clique |
| **Large (Test 2)** | $N \in [500, 1000]$ | 30 | 120 | Shortest Path, Connected Components, Cycle, Max Clique |
| **Stress (Test 3)** | $N \approx 2000$ | 15 | 60 | Shortest Path, Connected Components, Cycle, Max Clique |

*Total Synthetic Tasks:* **1,730**

### 2. GraphQA Benchmark (Fatemi et al., ICLR 2024)
Ingested from HuggingFace (`baharef/GraphQA`):

| Split | Graphs | Tasks | Task Types |
| :--- | :--- | :--- | :--- |
| **Train** | 1,000 | 3,000 | Shortest Path, Cycle Check, Reachability |
| **Validation** | 500 | 1,500 | Shortest Path, Cycle Check, Reachability |
| **Test** | 500 | 1,500 | Shortest Path, Cycle Check, Reachability |

*Total GraphQA Tasks:* **6,000** (2,000 graphs)

### 3. GraphArena Benchmark (Tang et al., ICLR 2025)
Ingested from the official GraphArena release on real-world knowledge graphs and social networks:

| Difficulty | Graphs | Tasks | Task Types |
| :--- | :--- | :--- | :--- |
| **Easy** | 1,000 | 1,000 | Distance (Knowledge Graphs), Connected (Social Networks) |
| **Hard** | 1,000 | 1,000 | Distance (Knowledge Graphs), Connected (Social Networks) |

*Total GraphArena Tasks:* **2,000** (2,000 graphs)

**Grand Total Benchmark Tasks:** **9,730**

---

## Tasks & Complexity

| Task | Complexity | Description | Verifier Logic |
| :--- | :--- | :--- | :--- |
| **Shortest Path** | Polynomial | Unweighted BFS shortest path length | Integer comparison ($\pm 0$) |
| **Connected Components** | Polynomial | Number of connected components | Integer comparison |
| **Cycle Detection** | Polynomial | Detect if graph has at least one cycle | Boolean equivalence |
| **Reachability** | Polynomial | Check if path exists between node pair | Boolean equivalence |
| **Max Clique** | NP-hard | Size of maximum clique | Exact comparison for small; lower bound ($\ge$) for large |
| **Hamiltonian Path** | NP-hard | Detect if path visits every node once | Exact backtracking for $\le 20$ nodes; heuristic for large |

---

## Verification & Tests

To execute the full verification suite:
```bash
python3 -m pytest tests/ -v
```
All **50 unit & integration tests** pass, verifying:
- Graph generation across all scale intervals
- Ground truth self-consistency across all task solvers
- GraphQA and GraphArena ingestion and schema conformance
- JSONL I/O roundtrips and text representations
- Batch evaluation and metric computation
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
