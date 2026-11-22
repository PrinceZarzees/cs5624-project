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

For plots, install the optional dependency with `python -m pip install -e ".[dev,plot]"`.

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
`data/graphqa_test.jsonl`. It supports all 12 GraphQA tasks and recomputes
ground truth with the current solvers where the task is deterministic; node
classification and maximum flow retain the source answer and mark it in
`meta`. See `experiments/BENCHMARK_COVERAGE.md` for the coverage matrix.

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
It converts all 10 GraphArena tasks, including GED/MCS (two-graph records) and
TSP (weighted records). GraphArena pickle files are executable data; only load
files obtained from the official release. See
`experiments/BENCHMARK_COVERAGE.md` for details.

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

## Agent harness and baselines

Phase 2 adds a sandboxed agent loop in `graphharness.agent`. A harness is just
data: a system prompt, a list of rules, a graph rendering, and resource limits.
The model can call `run_python` or answer; each code run gets a fresh Python
process with `G` (NetworkX graph), `params`, and `nodes` defined, plus a 30s
wall-clock timeout and memory/output limits. The graph file never contains the
ground-truth answer.

Three harnesses are built in:

- `baseline_a`: minimal protocol and tools, no strategy.
- `baseline_b`: the original human-written graph-algorithm harness, kept
  unchanged so the first pilot remains reproducible.
- `baseline_b_v2`: the current expert harness, with explicit
  connected-components, diameter-witness, max-clique, stop-after-success and
  no-numeric-node-filtering rules added after failure analysis.

Point the runner at any OpenAI-compatible chat endpoint:

```bash
export GRAPH_HARNESS_MODEL="Qwen/Qwen2.5-72B-Instruct"
export GRAPH_HARNESS_BASE_URL="http://localhost:8000/v1"
export GRAPH_HARNESS_API_KEY="..."   # or OPENAI_API_KEY

python -m graphharness.runner --list-harnesses
python -m graphharness.runner \
  --harness baseline_a --split test_small \
  --out experiments/baseline_a/test_small --workers 4 --cache
```

Each run writes `predictions.jsonl`, `scores.jsonl`, `summary.json`,
`summary.txt`, and one trajectory per problem under `trajectories/`. Runs are
resumable: existing prediction rows are reused unless `--no-resume` is passed.
`--cache` stores model responses in SQLite so repeated optimization and
ablation evaluations do not pay twice.

The sandbox is process-level isolation for runaway model code, not a hardened
multi-tenant security boundary. Do not run code from an adversary. Set
`GRAPH_HARNESS_SANDBOX_DIR` to put temporary sandbox directories on node-local
scratch instead of `/tmp`.

## Model choice for the first runs

The proposal names DeepSeek-V4.1-Flash, GLM, and Qwen as open-weight
candidates, with 7B–14B models suggested for fast local debugging. The first
runs use **`Qwen/Qwen3-4B-Instruct-2507`**:

- Apache-2.0, 4.02B parameters, ~7.6 GB of BF16 weights.
- Smallest Qwen instruct model that is practical for a code/tool agent.
- Served with vLLM over an OpenAI-compatible endpoint.

The machine-readable choice and fallbacks are in
[`configs/models.json`](configs/models.json). On ARC, start the server with:

```bash
VENV=/path/to/vllm-venv \
MODEL_PATH=/path/to/Qwen3-4B-Instruct-2507/snapshot \
sbatch scripts/serve_vllm.sbatch
```

The launcher needs a Python 3.12 environment with `vllm`, a CUDA toolkit with
`nvcc`, and `ninja` on `PATH` (FlashInfer JIT-compiles on first start). The
model and environment must be on shared storage such as `/projects`, not
node-local scratch. `scripts/run_qwen_pipeline.sbatch` starts the server and
runs baselines, optimization and scaling in one reproducible job.

A pilot on 25 `test_small` problems (5 per task) gave:

| Harness | Accuracy | Parsed | Feasible |
|---|---:|---:|---:|
| Baseline A | 20% | 100% | 20% |
| Baseline B | 72% | 100% | 84% |

These are pilot numbers from one lightweight model, not final results. The
artifacts are under `experiments/qwen3_4b_pilot/`. After the tooling fixes, the
full `test_small` results on BF16 are **87.3% (A), 90.0% (B), 93.3% (13-rule
B v2), 94.0% (14-rule B v2)**; see `experiments/RESULTS.md` and
`experiments/qwen3_4b_scaling_full/`.

A one-round optimizer pilot with the same 4B model proposed switching to a full
edge-list prompt, which dropped validation accuracy from 70% to 40% and was
rejected. That is a useful early signal that the validation gate is doing its
job; details are in `experiments/qwen3_4b_pilot/README.md`.

### CPU fallback

When B200 capacity is unavailable, the 7B fallback can run on the CPU
partition using a Q4_K_M GGUF model and llama.cpp:

```bash
LLAMA_VENV=/path/to/llama-venv \
MODEL_GGUF=/path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf \
sbatch scripts/serve_llama_cpp_cpu.sbatch
```

Observed throughput on 96 cores is about 12 tokens/s. The CPU/quantized pilot
is useful for debugging and pipeline validation, not as a replacement for the
BF16 GPU runs. `scripts/run_llama_cpp_cpu_pipeline.sbatch` starts the server and
runs a baseline in one job.

## Small-graph harness optimization

Phase 3 implements a SkillOpt/GEPA-style textual optimizer. It runs the current
harness on a small training batch, sends failure trajectories to a critic
model, validates the proposed prompt/rules on `val`, and keeps the candidate
only if validation accuracy improves.

```bash
python -m graphharness.optimizer \
  --start baseline_b_v2 --train-split train --val-split val \
  --rounds 15 --out experiments/optimizer --workers 4 --cache
```

Outputs under `experiments/optimizer/`:

- `best_harness.json` - frozen `H*_small`
- `history.jsonl` - every accepted/rejected edit and validation score
- `candidates/` and `checkpoints/` - round-by-round harness JSON
- `optimization_summary.json` - final comparison and token totals

## Scaling evaluation and plots

Phase 4 runs one or more harnesses on every graph scale and computes the
generalization gap.

```bash
python -m graphharness.scaling \
  --harnesses baseline_a baseline_b experiments/optimizer/best_harness.json \
  --splits test_small test_medium test_large test_stress \
  --workers 8 --cache --out experiments/scaling

python -m graphharness.plot_results \
  --run "Baseline A=experiments/scaling/baseline_a/*/scores.jsonl" \
  --run "Baseline B=experiments/scaling/baseline_b/*/scores.jsonl" \
  --run "H*_small=experiments/scaling/*candidate*/test_*/scores.jsonl" \
  --out experiments/plots
```

`experiments/scaling/scaling_summary.{json,txt}` reports accuracy, tokens,
latency, and `delta_gen = accuracy(test_small) - accuracy(test_large)` per
harness. `graphharness.analysis` prints size and task tables from any score
files.

## Full report

[`report/EXPERIMENTAL_REPORT.md`](report/EXPERIMENTAL_REPORT.md) contains the consolidated methods and results, including all reviewer-driven experiments.

## Review-driven experiments

`experiments/REVIEW_EXPERIMENTS.md` maps each reviewer comment to a controlled
experiment: cheap rubric proxies (SkillLift-inspired), tool-access ablation,
graph-characteristic stratification, GEPA/EvoSkill baselines, optimization-cost
accounting, specialized-harness routing/decomposition, and RRSI-style
regularization. Two analyses are already run:

```bash
python -m graphharness.characteristics --run "Bv2=experiments/qwen3_4b_scaling_full/scaling/baseline_b_v2/*/scores.jsonl"
python -m graphharness.cost_analysis --run "from_A=experiments/qwen3_4b_opt_a/optimizer" --run "from_Bv2=experiments/qwen3_4b/optimizer"
```

Results are in `experiments/analysis_by_characteristics.md`,
`experiments/optimization_costs.md`, and `experiments/REVIEW_RESULTS.md`.
The C tool backend is in `graphharness.csandbox`; Meta-Harness code evolution is
in `graphharness.meta_harness`; routing/decomposition is in
`graphharness.router`.

## Dataset and run inspection

Before or after a long run, verify the data and inspect failure trajectories:

```bash
python -m graphharness.verify_data
python -m graphharness.inspect_runs experiments/qwen3_4b_pilot/baseline_b \
  --failures 3 --out experiments/qwen3_4b_pilot/failure_analysis.txt
```

`verify_data` checks every stored ground-truth answer, record schema, manifest
counts, and SHA-256 checksums without calling a model. `inspect_runs` summarizes
tool calls, timeouts, failure notes, and code/observation excerpts.

## Rule ablations

Phase 5 rule-by-rule ablation reruns the frozen harness with each rule removed:

```bash
python -m graphharness.ablation \
  --harness experiments/optimizer/best_harness.json \
  --split test_large --workers 8 --cache \
  --out experiments/ablation
```

`ablation_summary.txt` shows the accuracy drop caused by removing each rule.
The report prose itself is intentionally not generated; only structure and
automation live in the repository.

## Repository layout

```text
src/graphharness/  authoritative Phase 1+ package
  agent.py         harness format, tool protocol, baselines, agent loop
  sandbox.py       resource-limited Python execution
  models.py        OpenAI-compatible client, SQLite cache, test models
  runner.py        resumable split runner and run summaries
  optimizer.py     small-graph textual harness optimization
  scaling.py       multi-split scaling runs and generalization gap
  ablation.py      rule-by-rule ablation
  analysis.py      accuracy by size/task from score files
  plot_results.py  scaling and task plots
configs/           reproducible split configuration
data/              manifest plus locally generated JSONL
tests/             unit and integration tests
scripts/           Slurm build, test, experiment and scaling jobs
experiments/       Phase 2+ experiment outputs
report/            final report
```
