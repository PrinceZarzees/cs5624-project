# Reproducibility record

Last updated: 2026-11-20 (US Eastern).

## Code

- Repository HEAD before this work: `fe5d5cd61fbed2712fe7349d0e6766b266f56fd9`
- Working tree: modified/untracked (Phase 2–5 code, scripts, pilots)
- Package version: `graphharness 0.2.0`
- Test command: `python -m pytest -q` → 165 passed, 0 skipped
- Dataset integrity: `python -m graphharness.verify_data` → all six splits OK
- Data seed/config: `configs/data_splits.json` (seed 5624)

Data artifacts are gitignored and rebuild with:

```bash
python -m graphharness.build_data
python -m graphharness.graphqa_adapter
python -m graphharness.grapharena_adapter --download
```

The legacy compatibility datasets can be regenerated with
`src/data_loader.build_full_dataset`, `src/ingest_graphqa.py`, and
`src/ingest_grapharena.py`.

The committed `data/manifest.json` is unchanged by a fresh build.

## Model environments and weights

Model weights and Python environments live on shared ARC project storage, not in
Git:

```text
/projects/psgroup/saminrk/gh-qwen/
```

| Item | Version / snapshot |
|---|---|
| Qwen3-4B-Instruct-2507 | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Qwen2.5-7B-Instruct GGUF Q4_K_M | `bartowski`, snapshot `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031` |
| Qwen3-4B-Instruct-2507 GGUF Q4_K_M | `unsloth`, snapshot `a06e946bb6b655725eafa393f4a9745d460374c9` |
| vLLM environment | Python 3.12, vLLM 0.34.0, PyTorch 2.14.0+cu130 |
| llama.cpp environment | Python 3.12, llama-cpp-python 0.3.37 |
| CUDA toolkit for FlashInfer JIT | 13.1.1 at `/cm/shared/apps/cuda13.1/toolkit/13.1.1` |

Serving scripts:

```text
scripts/serve_vllm.sbatch          GPU / BF16
scripts/serve_llama_cpp_cpu.sbatch CPU / GGUF fallback
scripts/run_qwen_pipeline.sbatch   self-contained baseline + optimize + scaling
```

## Slurm runs

| Job | Purpose | Status |
|---|---|---|
| `1535583` | Qwen3-4B 5-problem baseline smoke | completed |
| `1535647` | Qwen3-4B 25-problem baseline pilot | completed |
| `1535717` | Qwen3-4B 1-round optimizer pilot | completed |
| `1537011` | Qwen3-4B full pipeline (baselines, optimizer, scaling) | completed, 15 min on owl-sr002 |
| `1537117` | Qwen2.5-7B CPU full test_small | completed on owl080 |
| `1537830` | Qwen3-4B optimizer from baseline_a + scaling | completed |
| `1538020` | Qwen3-4B full rule ablation on test_large | completed |
| `1538032` | Qwen3-4B full scaling (14-rule v2) | completed |
| `1538186` | Qwen3-4B full rule ablation on test_stress | completed |
| `1538817` | Qwen3-4B full rule ablation on val | completed |
| `1534890` | interactive CPU node: 4B/7B CPU servers and 7B 14-rule full run | completed |
| `1540814` | Qwen3-4B real GraphArena/GraphQA evaluation (20/task) | completed |

The full-pipeline job wrote to `experiments/qwen3_4b/`: full baselines,
8-round optimizer from `baseline_b_v2`, and a 3-per-task scaling pilot. The
scheduler initially projected a next-day start, but the job backfilled onto
owl-sr002 and completed in 15 minutes.

## CPU fallback

CPU fallbacks: Qwen3-4B Q4_K_M (~18 tokens/s) and Qwen2.5-7B Q4_K_M
(~12 tokens/s). Artifacts are under `experiments/qwen3_4b_cpu/` and
`experiments/qwen2_5_7b_cpu/`. These validate the pipeline and provide
cross-model checks, but are not a substitute for the BF16 GPU comparison.

## Harness changes after the first pilot

The first Qwen3-4B pilot exposed four issues that were fixed and are covered by
tests:

1. The sandbox now echoes a bare final expression, so `list(nx.connected_components(G))`
   is visible without requiring `print(...)`.
2. Tool/proposal parsing accepts Python-literal JSON with triple-quoted strings.
3. If a response contains both code and a guessed `<answer>`, the code is run
   first; the guessed answer is not accepted until the model sees the observation.
4. After the tool-step budget is exhausted, the agent gets one final-answer turn
   with no code execution, so a successful observation is not wasted.

`baseline_b_v2` adds explicit connected-components, diameter-witness,
max-clique, stop-after-success, no-numeric-node-filtering and
generator-materialization rules. Original `baseline_b` is kept unchanged for
the first pilot.

## Disk inventory (shared storage)

```text
/projects/psgroup/saminrk/gh-qwen/venv         ~9.5 GB  (vLLM + torch)
/projects/psgroup/saminrk/gh-qwen/llama_venv   ~178 MB  (llama-cpp-python)
/projects/psgroup/saminrk/gh-qwen/hf/models    ~29 GB   (model blobs)
```

The HF cache keeps multiple snapshots; these are not in Git.

## Real benchmark data

```text
data/grapharena_smoke/   2 per task (40 records); raw zip + unzipped files
data/grapharena_eval/    20 per task (400 records)
data/graphqa_smoke/      2 per task (66 published records)
data/graphqa_eval/       20 per task (660 published records)
```

Converted JSONL and raw benchmark files are gitignored and can be regenerated
with `python -m graphharness.grapharena_adapter --download` and
`python -m graphharness.graphqa_adapter`. Real evaluation job: `1540814`.
