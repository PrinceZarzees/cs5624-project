# CS 5624: Harness Optimization for Graph Reasoning

This project studies whether a harness optimized on small graphs generalizes
to larger graphs.

See [`project_proposal/PROJECT_PLAN.md`](project_proposal/PROJECT_PLAN.md) for
the full project plan.

## Setup

```bash
python -m pip install -e ".[dev]"
pytest
```

The authoritative Phase 1 implementation is the
[`src/graphharness/`](src/graphharness/) package. The older `src/*.py`
modules are retained only as compatibility code for existing experiment
artifacts; new work should use `graphharness`.

## Synthetic data

Generated JSONL files are intentionally ignored by Git. Build them after
installation:

```bash
python -m graphharness.build_data
```

This writes `data/train.jsonl`, `data/val.jsonl`,
`data/test_small.jsonl`, `data/test_medium.jsonl`,
`data/test_large.jsonl`, and `data/test_stress.jsonl`. The committed
[`data/manifest.json`](data/manifest.json) records the seed, counts, ranges,
and SHA-256 checksums. A fresh build should produce an unchanged manifest.

| Split | Nodes | Problems per task | Purpose |
|---|---:|---:|---|
| `train` | 4–50 | 60 | Harness optimization |
| `val` | 4–50 | 30 | Accept/reject harness edits |
| `test_small` | 4–50 | 30 | In-distribution evaluation |
| `test_medium` | 100–300 | 30 | Scale evaluation |
| `test_large` | 500–1000 | 20 | Scale evaluation |
| `test_stress` | 2000 | 10 | Stress evaluation |

Synthetic tasks are shortest path, connected components, cycle detection,
diameter, and maximum clique. GraphQA additionally uses reachability.
Graph families are Erdős–Rényi, Barabási–Albert, and
Watts–Strogatz. Node IDs are randomized to prevent label shortcuts.

## GraphQA

GraphQA can be downloaded and converted into the same record schema:

```bash
python -m graphharness.graphqa_adapter
```

Use `--raw-dir` to select a cache directory, `--out` to select the output
directory, and `--limit N` for a small local smoke test. The adapter writes
`data/graphqa_train.jsonl`, `data/graphqa_validation.jsonl`, and
`data/graphqa_test.jsonl`. It supports shortest path, cycle detection, and
reachability, and recomputes ground truth with the current solvers.

## GraphArena

GraphArena is distributed as a large Google Drive ZIP. Download it manually
or use the optional `gdown` dependency:

```bash
python -m pip install -e ".[grapharena]"
python -m graphharness.grapharena_adapter --download
```

Alternatively:

```bash
python -m graphharness.grapharena_adapter --zip path/to/dataset.zip
```

The adapter writes `data/ga_easy.jsonl` and `data/ga_hard.jsonl`.
GraphArena pickle files are executable data; only load files obtained from the
official release.

## Records and scoring

Each current-schema record contains `id`, `source`, `task`, `split`, `graph`,
`params`, `question`, `answer`, and `meta`. Load and score records with:

```python
from graphharness import load_split, score_response
from graphharness.encoding import build_prompt

for record in load_split("val", tasks=["shortest_path"]):
    prompt = build_prompt(record)
    result = score_response(record, model_response)
```

For batch scoring, provide a JSONL file containing `{"id": ..., "response": ...}`:

```bash
python -m graphharness.evaluate_answer \
  --data data/test_medium.jsonl \
  --predictions preds.jsonl \
  --out scores.jsonl
```

## Repository layout

```text
src/graphharness/  authoritative Phase 1 package
configs/           reproducible split configuration
data/              manifest plus locally generated JSONL
tests/             unit and integration tests
scripts/           Slurm build and test jobs
experiments/       Phase 2+ experiment outputs
report/            final report
```
