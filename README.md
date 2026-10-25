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
