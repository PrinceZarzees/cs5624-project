# CS 5624 Project: Harness Optimization for Graph Reasoning

**Learn Small, Run Big** — Can an LLM agent harness optimized on small graphs generalize to large ones?

## Project Structure

```
cs5624/
├── src/                    # Source code
│   ├── __init__.py
│   ├── graph_generator.py  # Synthetic graph generation (ER, BA, WS)
│   ├── tasks.py            # Task definitions & ground-truth solvers
│   ├── data_loader.py      # JSONL I/O & dataset pipeline
│   └── evaluate_answer.py  # Answer verification & metrics
├── data/                   # Generated datasets (JSONL)
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

## Quick Start (VT ARC Falcon)

### 1. Generate Dataset
```bash
sbatch scripts/generate_data.sbatch
```

### 2. Run Tests
```bash
sbatch scripts/run_tests.sbatch
```

### 3. Check Results
```bash
squeue -u $USER
cat cs5624_datagen_<jobid>.out
cat cs5624_tests_<jobid>.out
```

## Graph Scales

| Scale   | Node Range | Count | Usage         |
|---------|-----------|-------|---------------|
| Small   | 4–50      | 300   | Train (200) / Val (100) |
| Medium  | 100–300   | 60    | Test 1        |
| Large   | 500–1000  | 30    | Test 2        |
| Stress  | 1500–2000 | 15    | Test 3        |

## Tasks

| Task                | Complexity  | Description                          |
|---------------------|-------------|--------------------------------------|
| Shortest Path       | Polynomial  | BFS shortest path length             |
| Connected Components| Polynomial  | Count connected components           |
| Cycle Detection     | Polynomial  | Detect if graph has a cycle          |
| Max Clique          | NP-hard     | Size of maximum clique               |
| Hamiltonian Path    | NP-hard     | Does a Hamiltonian path exist?       |

## Dependencies

- Python 3.9+
- NetworkX 3.x
- NumPy
- pytest
