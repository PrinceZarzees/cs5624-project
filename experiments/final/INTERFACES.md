# E0 interface discovery

Generated 2026-11-21 from the repository. Canonical flags are copied from the
actual `--help` output and sbatch files; new experiments must use these names.

## runner
usage: runner.py [-h] [--harness HARNESS] [--split SPLIT]
                 [--data-dir DATA_DIR] [--out OUT] [--tasks [TASKS ...]]
                 [--ids [IDS ...]] [--limit LIMIT]
                 [--limit-per-task LIMIT_PER_TASK] [--workers WORKERS]
                 [--max-steps MAX_STEPS] [--no-resume] [--no-trajectories]
                 [--quiet] [--list-harnesses] [--model MODEL]
                 [--base-url BASE_URL] [--api-key-env API_KEY_ENV]
                 [--temperature TEMPERATURE] [--max-tokens MAX_TOKENS]
                 [--request-timeout REQUEST_TIMEOUT] [--retries RETRIES]
                 [--cache [CACHE]]

Run a harness over a split and save predictions, scores and trajectories.

Example::

    python -m graphharness.runner         --harness baseline_b --split test_medium         --model Qwen/Qwen2.5-72B-Instruct --base-url http://localhost:8000/v1         --out experiments/baseline_b/test_medium --workers 8

The runner is resumable: existing rows in ``predictions.jsonl`` are reused and
only missing problem ids are sent to the model.

options:
  -h, --help            show this help message and exit
  --harness HARNESS     baseline name or path to a harness JSON checkpoint
  --split SPLIT
  --data-dir DATA_DIR
  --out OUT             output directory (default:
                        experiments/<harness>/<split>)
  --tasks [TASKS ...]   only these task names
  --ids [IDS ...]       only these problem ids
  --limit LIMIT         max problems total
  --limit-per-task LIMIT_PER_TASK
                        max problems per task
  --workers WORKERS     parallel model calls
  --max-steps MAX_STEPS
                        override harness max_steps
  --no-resume
  --no-trajectories
  --quiet
  --list-harnesses

model:
  --model MODEL         model name for an OpenAI-compatible endpoint (or
                        GRAPH_HARNESS_MODEL)
  --base-url BASE_URL   e.g. https://api.openai.com/v1 or
                        http://localhost:8000/v1
  --api-key-env API_KEY_ENV
                        environment variable holding the API key (default:
                        GRAPH_HARNESS_API_KEY)
  --temperature TEMPERATURE
  --max-tokens MAX_TOKENS
  --request-timeout REQUEST_TIMEOUT
  --retries RETRIES
  --cache [CACHE]       SQLite response cache path (default off; use --cache
                        with no path for the default)

## optimizer
usage: optimizer.py [-h] [--start START] [--train-split TRAIN_SPLIT]
                    [--val-split VAL_SPLIT] [--data-dir DATA_DIR]
                    [--tasks [TASKS ...]] [--rounds ROUNDS]
                    [--train-per-task TRAIN_PER_TASK]
                    [--val-per-task VAL_PER_TASK] [--max-steps MAX_STEPS]
                    [--workers WORKERS] [--seed SEED] [--min-gain MIN_GAIN]
                    [--out OUT] [--critic-model CRITIC_MODEL]
                    [--critic-base-url CRITIC_BASE_URL]
                    [--critic-temperature CRITIC_TEMPERATURE] [--rrs]
                    [--critic-max-tokens CRITIC_MAX_TOKENS] [--model MODEL]
                    [--base-url BASE_URL] [--api-key-env API_KEY_ENV]
                    [--temperature TEMPERATURE] [--max-tokens MAX_TOKENS]
                    [--request-timeout REQUEST_TIMEOUT] [--retries RETRIES]
                    [--cache [CACHE]]

Textual harness optimization on small graphs (Phase 3).

The loop is deliberately close to SkillOpt/GEPA:

1. Run the current harness on a small batch from ``train``.
2. Collect failure trajectories and a few successes.
3. Ask a critic model for a revised system prompt / rule list.
4. Evaluate the candidate on a held-out ``val`` sample.
5. Keep it if validation accuracy improves (or ties with a shorter prompt).

Only the harness changes; model weights and the task data do not.  Every round
is logged and every candidate is checkpointed, so a run can be inspected or
stopped and resumed from ``best_harness.json``.

options:
  -h, --help            show this help message and exit
  --start START         starting harness name or JSON checkpoint
  --train-split TRAIN_SPLIT
  --val-split VAL_SPLIT
  --data-dir DATA_DIR
  --tasks [TASKS ...]   only optimize on these task names
  --rounds ROUNDS
  --train-per-task TRAIN_PER_TASK
  --val-per-task VAL_PER_TASK
  --max-steps MAX_STEPS
  --workers WORKERS
  --seed SEED
  --min-gain MIN_GAIN
  --out OUT
  --critic-model CRITIC_MODEL
                        separate critic model (default: actor model)
  --critic-base-url CRITIC_BASE_URL
  --critic-temperature CRITIC_TEMPERATURE
  --rrs                 use RRSI-style edit budget, shortcut filter and multi-
                        view validation
  --critic-max-tokens CRITIC_MAX_TOKENS

model:
  --model MODEL         model name for an OpenAI-compatible endpoint (or
                        GRAPH_HARNESS_MODEL)
  --base-url BASE_URL   e.g. https://api.openai.com/v1 or
                        http://localhost:8000/v1
  --api-key-env API_KEY_ENV
                        environment variable holding the API key (default:
                        GRAPH_HARNESS_API_KEY)
  --temperature TEMPERATURE
  --max-tokens MAX_TOKENS
  --request-timeout REQUEST_TIMEOUT
  --retries RETRIES
  --cache [CACHE]       SQLite response cache path (default off; use --cache
                        with no path for the default)

## gepa
usage: gepa.py [-h] [--start START] [--train-split TRAIN_SPLIT]
               [--val-split VAL_SPLIT] [--data-dir DATA_DIR] [--rounds ROUNDS]
               [--candidates-per-round CANDIDATES_PER_ROUND]
               [--train-per-task TRAIN_PER_TASK] [--val-per-task VAL_PER_TASK]
               [--max-steps MAX_STEPS] [--seed SEED] [--out OUT]
               [--critic-model CRITIC_MODEL]
               [--critic-base-url CRITIC_BASE_URL] [--model MODEL]
               [--base-url BASE_URL] [--api-key-env API_KEY_ENV]
               [--temperature TEMPERATURE] [--max-tokens MAX_TOKENS]
               [--request-timeout REQUEST_TIMEOUT] [--retries RETRIES]
               [--cache [CACHE]]

GEPA-style prompt evolution baseline.

This is a lightweight, reproducible baseline: each round asks the critic for
several revised system prompts, evaluates each on a fixed small-graph
validation sample, and keeps the best prompt. Rules are held fixed, which
isolates prompt evolution from skill/rule evolution.

options:
  -h, --help            show this help message and exit
  --start START
  --train-split TRAIN_SPLIT
  --val-split VAL_SPLIT
  --data-dir DATA_DIR
  --rounds ROUNDS
  --candidates-per-round CANDIDATES_PER_ROUND
  --train-per-task TRAIN_PER_TASK
  --val-per-task VAL_PER_TASK
  --max-steps MAX_STEPS
  --seed SEED
  --out OUT
  --critic-model CRITIC_MODEL
  --critic-base-url CRITIC_BASE_URL

model:
  --model MODEL         model name for an OpenAI-compatible endpoint (or
                        GRAPH_HARNESS_MODEL)
  --base-url BASE_URL   e.g. https://api.openai.com/v1 or
                        http://localhost:8000/v1
  --api-key-env API_KEY_ENV
                        environment variable holding the API key (default:
                        GRAPH_HARNESS_API_KEY)
  --temperature TEMPERATURE
  --max-tokens MAX_TOKENS
  --request-timeout REQUEST_TIMEOUT
  --retries RETRIES
  --cache [CACHE]       SQLite response cache path (default off; use --cache
                        with no path for the default)

## meta_harness
usage: meta_harness.py [-h] [--start START] [--train-split TRAIN_SPLIT]
                       [--val-split VAL_SPLIT] [--data-dir DATA_DIR]
                       [--rounds ROUNDS] [--train-per-task TRAIN_PER_TASK]
                       [--val-per-task VAL_PER_TASK] [--max-steps MAX_STEPS]
                       [--seed SEED] [--out OUT] [--critic-model CRITIC_MODEL]
                       [--model MODEL] [--base-url BASE_URL]
                       [--api-key-env API_KEY_ENV] [--temperature TEMPERATURE]
                       [--max-tokens MAX_TOKENS]
                       [--request-timeout REQUEST_TIMEOUT] [--retries RETRIES]
                       [--cache [CACHE]]

Meta-Harness-lite: LLM-driven, code-level harness evolution. Unlike the
SkillOpt/GEPA loops, a candidate may add a Python helper preamble that is
injected into every sandbox run. This makes the tool loop itself evolvable
while keeping model weights frozen.

options:
  -h, --help            show this help message and exit
  --start START
  --train-split TRAIN_SPLIT
  --val-split VAL_SPLIT
  --data-dir DATA_DIR
  --rounds ROUNDS
  --train-per-task TRAIN_PER_TASK
  --val-per-task VAL_PER_TASK
  --max-steps MAX_STEPS
  --seed SEED
  --out OUT
  --critic-model CRITIC_MODEL

model:
  --model MODEL         model name for an OpenAI-compatible endpoint (or
                        GRAPH_HARNESS_MODEL)
  --base-url BASE_URL   e.g. https://api.openai.com/v1 or
                        http://localhost:8000/v1
  --api-key-env API_KEY_ENV
                        environment variable holding the API key (default:
                        GRAPH_HARNESS_API_KEY)
  --temperature TEMPERATURE
  --max-tokens MAX_TOKENS
  --request-timeout REQUEST_TIMEOUT
  --retries RETRIES
  --cache [CACHE]       SQLite response cache path (default off; use --cache
                        with no path for the default)

## router
usage: router.py [-h] --library LIBRARY [--split SPLIT] [--data-dir DATA_DIR]
                 [--mode {routed,decomposed}]
                 [--limit-per-task LIMIT_PER_TASK] [--max-steps MAX_STEPS]
                 --out OUT [--model MODEL] [--base-url BASE_URL]
                 [--api-key-env API_KEY_ENV] [--temperature TEMPERATURE]
                 [--max-tokens MAX_TOKENS] [--request-timeout REQUEST_TIMEOUT]
                 [--retries RETRIES] [--cache [CACHE]]

Specialized-harness routing and decomposition experiments.

The router chooses one of a small library of class-specific harnesses from
cheap graph features (size, density, task, family). The decomposed runner is
only applied to additive tasks (MIS, MVC, max clique, connected components)
where solving components independently is exact.

options:
  -h, --help            show this help message and exit
  --library LIBRARY     directory of harness JSON files
  --split SPLIT
  --data-dir DATA_DIR
  --mode {routed,decomposed}
  --limit-per-task LIMIT_PER_TASK
  --max-steps MAX_STEPS
  --out OUT

model:
  --model MODEL         model name for an OpenAI-compatible endpoint (or
                        GRAPH_HARNESS_MODEL)
  --base-url BASE_URL   e.g. https://api.openai.com/v1 or
                        http://localhost:8000/v1
  --api-key-env API_KEY_ENV
                        environment variable holding the API key (default:
                        GRAPH_HARNESS_API_KEY)
  --temperature TEMPERATURE
  --max-tokens MAX_TOKENS
  --request-timeout REQUEST_TIMEOUT
  --retries RETRIES
  --cache [CACHE]       SQLite response cache path (default off; use --cache
                        with no path for the default)

## rubric_proxy
usage: rubric_proxy.py [-h] --run RUN [--out OUT] [--md MD] [--model MODEL]
                       [--base-url BASE_URL] [--api-key-env API_KEY_ENV]
                       [--temperature TEMPERATURE] [--max-tokens MAX_TOKENS]
                       [--request-timeout REQUEST_TIMEOUT] [--retries RETRIES]
                       [--cache [CACHE]]

SkillLift-style cheap proxy for ranking candidate harnesses.

The judge reads only a candidate harness (prompt + rules) and scores it on a
dense rubric. Those scores are compared with the validation accuracy that was
actually measured during optimization, so no full large-graph rollouts are
needed to test the proxy.

    python -m graphharness.rubric_proxy         --run experiments/qwen3_4b_opt_a/optimizer         --run experiments/qwen3_4b/optimizer         --model qwen3-4b --base-url http://127.0.0.1:8000/v1         --out experiments/review_rubric_proxy/results.json

options:
  -h, --help            show this help message and exit
  --run RUN             optimizer output dir
  --out OUT
  --md MD

model:
  --model MODEL         model name for an OpenAI-compatible endpoint (or
                        GRAPH_HARNESS_MODEL)
  --base-url BASE_URL   e.g. https://api.openai.com/v1 or
                        http://localhost:8000/v1
  --api-key-env API_KEY_ENV
                        environment variable holding the API key (default:
                        GRAPH_HARNESS_API_KEY)
  --temperature TEMPERATURE
  --max-tokens MAX_TOKENS
  --request-timeout REQUEST_TIMEOUT
  --retries RETRIES
  --cache [CACHE]       SQLite response cache path (default off; use --cache
                        with no path for the default)

## characteristics
usage: characteristics.py [-h] --run LABEL=PATH [--data-dir DATA_DIR]
                          [--out OUT] [--csv CSV]

Accuracy stratified by graph characteristics.

This is the analysis for the review asking where optimized harnesses generalize:
density, topology/family, task, and scale.

    python -m graphharness.characteristics         --run "A=experiments/qwen3_4b_scaling_full/scaling/baseline_a/*/scores.jsonl"         --run "Bv2=experiments/qwen3_4b_scaling_full/scaling/baseline_b_v2/*/scores.jsonl"         --out experiments/analysis_by_characteristics.md

options:
  -h, --help           show this help message and exit
  --run LABEL=PATH     label and comma-separated score path(s)/globs;
                       repeatable
  --data-dir DATA_DIR
  --out OUT
  --csv CSV

## cost_analysis
usage: cost_analysis.py [-h] --run LABEL=PATH
                        [--input-cost-per-m INPUT_COST_PER_M]
                        [--output-cost-per-m OUTPUT_COST_PER_M] [--out OUT]

Optimization-cost accounting from optimizer history/summary files.

    python -m graphharness.cost_analysis         --run "from_A=experiments/qwen3_4b_opt_a/optimizer"         --run "from_Bv2=experiments/qwen3_4b/optimizer"         --out experiments/optimization_costs.md

Reports validation trajectory, cumulative tokens, wall time, and optional
per-million-token cost estimates. The token counts come from the real
optimizer runs; no model calls are made here.

options:
  -h, --help            show this help message and exit
  --run LABEL=PATH
  --input-cost-per-m INPUT_COST_PER_M
  --output-cost-per-m OUTPUT_COST_PER_M
  --out OUT

## build_data
usage: build_data.py [-h] [--config CONFIG] [--out OUT]
                     [--splits [SPLITS ...]] [--tasks [TASKS ...]]

Build the synthetic data splits (reproducible from one seed).

    python -m graphharness.build_data                      # all splits, default config
    python -m graphharness.build_data --splits train val   # only some splits
    python -m graphharness.build_data --config configs/data_splits.json --out data

Writes ``<out>/<split>.jsonl`` for each split and ``<out>/manifest.json``
with counts and a checksum per file, so teammates can confirm they built
exactly the same data.

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --out OUT
  --splits [SPLITS ...]
                        only build these splits
  --tasks [TASKS ...]   only build these tasks

## grapharena_adapter
usage: grapharena_adapter.py [-h] (--download | --zip ZIP |
                             --dataset-dir DATASET_DIR) [--out OUT]
                             [--raw-dir RAW_DIR] [--limit LIMIT]

Convert GraphArena problems into our common JSONL format.

GraphArena's processed data is a set of pickle files (``Distance_easy.pkl``,
``MCP_hard.pkl``, ...) inside ``dataset.zip``, which lives in a Google Drive
folder linked from https://github.com/squareRoot3/GraphArena. There are three
ways to get it:

    # 1. download + unzip + convert in one step (needs `pip install gdown`)
    python -m graphharness.grapharena_adapter --download

    # 2. you downloaded dataset.zip by hand in a browser
    python -m graphharness.grapharena_adapter --zip ~/Downloads/dataset.zip

    # 3. you already unzipped it
    python -m graphharness.grapharena_adapter --dataset-dir path/to/dataset

Raw files go to ``data/grapharena/`` (ignored by git) and the converted
problems to ``data/ga_easy.jsonl`` and ``data/ga_hard.jsonl``.

All ten GraphArena tasks are converted:

    Distance  -> shortest_path
    Connected -> connected_components
    Diameter  -> diameter
    MCP       -> max_clique
    Neighbor  -> common_neighbors
    MIS       -> max_independent_set
    MVC       -> min_vertex_cover
    GED       -> graph_edit_distance
    MCS       -> max_common_subgraph
    TSP       -> traveling_salesman

GED/MCS use two-graph records (``graphs``) and TSP uses weighted edges.
Ground truth is recomputed with our solvers where feasible; for hard GED/MCS/TSP
instances whose exact solver is outside the small-instance limit, the source
answer/path is retained and marked in ``meta``. GraphArena's own answer is
always kept in ``meta``.

Note: GraphArena's "hard" graphs have at most 50 nodes, so they are still
"small" for this project. Larger test graphs come from ``build_data``.

Security note: pickle files can run code when loaded. Only load the official
GraphArena files.

options:
  -h, --help            show this help message and exit
  --download            download dataset.zip from Google Drive
  --zip ZIP             path to a dataset.zip you downloaded yourself
  --dataset-dir DATASET_DIR
                        folder with already unzipped GraphArena *.pkl files
  --out OUT
  --raw-dir RAW_DIR     where to put the zip and unzipped files (default:
                        <out>/grapharena)
  --limit LIMIT         max problems per task (default: all)

## graphqa_adapter
usage: graphqa_adapter.py [-h] [--raw-dir RAW_DIR] [--out OUT] [--limit LIMIT]

Convert GraphQA JSON files into the graphharness JSONL schema. GraphQA
publishes one JSON array per task and split. This adapter supports all twelve
GraphQA tasks. It parses the edge list and query from each question and
recomputes ground truth with the same solvers where the task is deterministic.
GraphQA's node-classification and maximum-flow answers are retained from the
source dataset because their exact generation semantics are not encoded in the
question; ``meta["ground_truth_source"]`` records that distinction.

options:
  -h, --help         show this help message and exit
  --raw-dir RAW_DIR
  --out OUT
  --limit LIMIT

## stats
/home/saminrk/workspace/choto/.venv/bin/python: No module named graphharness.stats
(no CLI; read source)

## evoskill
/home/saminrk/workspace/choto/.venv/bin/python: No module named graphharness.evoskill
(no CLI; read source)

## scripts/generate_data.sbatch
2:#SBATCH --job-name=cs5624_datagen
3:#SBATCH --partition=a30_normal_q
4:#SBATCH --account=psgroup
5:#SBATCH --nodes=1
6:#SBATCH --ntasks=1
7:#SBATCH --cpus-per-task=4
8:#SBATCH --mem=16G
9:#SBATCH --gres=gpu:1
10:#SBATCH --time=01:00:00
11:#SBATCH --output=%x_%j.out
12:#SBATCH --error=%x_%j.err
16:echo "Job ID: ${SLURM_JOB_ID}"
25:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

## scripts/run_harness.sbatch
2:#SBATCH --job-name=cs5624_harness
3:#SBATCH --partition=a30_normal_q
4:#SBATCH --account=psgroup
5:#SBATCH --nodes=1
6:#SBATCH --ntasks=1
7:#SBATCH --cpus-per-task=8
8:#SBATCH --mem=32G
9:#SBATCH --gres=gpu:1
10:#SBATCH --time=12:00:00
11:#SBATCH --output=%x_%j.out
12:#SBATCH --error=%x_%j.err
21:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
23:: "${HARNESS:=baseline_b}"
24:: "${SPLIT:=test_small}"
25:: "${MODEL:?set MODEL to the model name served by the endpoint}"
26:: "${BASE_URL:=${GRAPH_HARNESS_BASE_URL:-http://127.0.0.1:8000/v1}}"
27:: "${WORKERS:=8}"
28:: "${OUT:=experiments/${HARNESS}/${SPLIT}}"
29:: "${LIMIT_PER_TASK:=}"
32:export GRAPH_HARNESS_MODEL="${MODEL}"
33:export GRAPH_HARNESS_BASE_URL="${BASE_URL}"
35:ARGS=(--harness "${HARNESS}" --split "${SPLIT}" --out "${OUT}"
36:      --model "${MODEL}" --base-url "${BASE_URL}" --workers "${WORKERS}" --cache)
37:if [[ -n "${LIMIT_PER_TASK}" ]]; then
38:  ARGS+=(--limit-per-task "${LIMIT_PER_TASK}")
41:echo "harness=${HARNESS} split=${SPLIT} model=${MODEL} workers=${WORKERS}"
42:echo "out=${OUT}"
43:python3 -m graphharness.runner "${ARGS[@]}"

## scripts/run_llama_cpp_cpu_pipeline.sbatch
2:#SBATCH --job-name=gh_llama_pipeline
3:#SBATCH --partition=normal_q
4:#SBATCH --qos=owl_normal_base
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=96
9:#SBATCH --mem=512G
10:#SBATCH --time=08:00:00
11:#SBATCH --output=%x_%j.out
12:#SBATCH --error=%x_%j.err
24:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
26:: "${LLAMA_VENV:?set LLAMA_VENV}"
27:: "${MODEL_GGUF:?set MODEL_GGUF}"
28:: "${CLIENT_PYTHON:?set CLIENT_PYTHON}"
29:: "${OUT:?set OUT}"
30:: "${MODEL_ALIAS:=qwen2.5-7b}"
31:: "${PORT:=8001}"
32:: "${N_CTX:=8192}"
33:: "${N_THREADS:=64}"
34:: "${N_THREADS_BATCH:=96}"
35:: "${CHAT_FORMAT:=chatml}"
36:: "${SPLIT:=test_small}"
37:: "${HARNESS:=baseline_b_v2}"
38:: "${LIMIT_PER_TASK:=}"
39:: "${MAX_STEPS:=8}"
42:LOGS="$OUT/logs"
43:mkdir -p "$LOGS"
45:export PYTHONPATH="$REPO/src"
47:export GRAPH_HARNESS_SANDBOX_DIR="$OUT/sandbox"
49:echo "job=${SLURM_JOB_ID} host=$(hostname) model=$MODEL_GGUF"
50:"$LLAMA_VENV/bin/python" -m llama_cpp.server \
51:  --model "$MODEL_GGUF" --model_alias "$MODEL_ALIAS" \
52:  --host 127.0.0.1 --port "$PORT" --n_ctx "$N_CTX" --n_batch 512 \
53:  --n_threads "$N_THREADS" --n_threads_batch "$N_THREADS_BATCH" \
54:  --chat_format "$CHAT_FORMAT" \
55:  > "$LOGS/llama_server.log" 2>&1 &
59:  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
62:  if ! kill -0 "$SERVER_PID" 2>/dev/null; then break; fi
66:  tail -100 "$LOGS/llama_server.log"
67:  kill "$SERVER_PID" 2>/dev/null || true
71:ARGS=(--harness "$HARNESS" --split "$SPLIT" --max-steps "$MAX_STEPS" --workers 1
72:      --model "$MODEL_ALIAS" --base-url "http://127.0.0.1:${PORT}/v1"
73:      --max-tokens 512 --request-timeout 300 --out "$OUT/$HARNESS/$SPLIT")
74:if [[ -n "$LIMIT_PER_TASK" ]]; then ARGS+=(--limit-per-task "$LIMIT_PER_TASK"); fi
76:"$CLIENT_PYTHON" -m graphharness.runner "${ARGS[@]}"
78:kill "$SERVER_PID" 2>/dev/null || true
79:wait "$SERVER_PID" 2>/dev/null || true
81:  cp "${SLURM_SUBMIT_DIR:-.}/gh_llama_pipeline_${SLURM_JOB_ID}.${ext}" "$LOGS/" 2>/dev/null || true

## scripts/run_meta_harness.sbatch
2:#SBATCH --job-name=gh_meta
3:#SBATCH --partition=b200_normal_q
4:#SBATCH --qos=owl_b200_normal_short
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=8
9:#SBATCH --mem=64G
10:#SBATCH --gres=gpu:1
11:#SBATCH --time=01:30:00
12:#SBATCH --output=%x_%j.out
13:#SBATCH --error=%x_%j.err
15:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
16:: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
17:ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/review_meta_harness; mkdir -p "$OUT/logs"
18:export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
19:export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
21:export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
22:export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
23:"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 \
25:  --disable-log-requests > "$OUT/logs/vllm_server.log" 2>&1 &
27:for i in $(seq 1 180); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 2; done
28:curl -sf http://127.0.0.1:8000/health >/dev/null || { tail -100 "$OUT/logs/vllm_server.log"; exit 1; }
30:"$CLIENT_PYTHON" -m graphharness.meta_harness --start baseline_a --rounds 3 \
31:  --train-per-task 4 --val-per-task 4 --out "$OUT" $MODEL_ARGS
32:"$CLIENT_PYTHON" -m graphharness.runner --harness "$OUT/best_harness.json" \
34:  --out "$OUT/eval_best" $MODEL_ARGS
35:kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true

## scripts/run_optimizer.sbatch
2:#SBATCH --job-name=cs5624_opt
3:#SBATCH --partition=a30_normal_q
4:#SBATCH --account=psgroup
5:#SBATCH --nodes=1
6:#SBATCH --ntasks=1
7:#SBATCH --cpus-per-task=8
8:#SBATCH --mem=32G
9:#SBATCH --gres=gpu:1
10:#SBATCH --time=24:00:00
11:#SBATCH --output=%x_%j.out
12:#SBATCH --error=%x_%j.err
18:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
20:: "${START:=baseline_b}"
21:: "${ROUNDS:=15}"
22:: "${MODEL:?set MODEL to the model name served by the endpoint}"
23:: "${BASE_URL:=${GRAPH_HARNESS_BASE_URL:-http://127.0.0.1:8000/v1}}"
24:: "${WORKERS:=8}"
25:: "${TRAIN_PER_TASK:=6}"
26:: "${VAL_PER_TASK:=5}"
27:: "${OUT:=experiments/optimizer}"
30:export GRAPH_HARNESS_MODEL="${MODEL}"
31:export GRAPH_HARNESS_BASE_URL="${BASE_URL}"
33:echo "start=${START} rounds=${ROUNDS} model=${MODEL} out=${OUT}"
35:  --start "${START}" --rounds "${ROUNDS}" --out "${OUT}" \
36:  --model "${MODEL}" --base-url "${BASE_URL}" --workers "${WORKERS}" \
37:  --train-per-task "${TRAIN_PER_TASK}" --val-per-task "${VAL_PER_TASK}" --cache

## scripts/run_qwen_pipeline.sbatch
2:#SBATCH --job-name=gh_qwen_pipeline
3:#SBATCH --partition=b200_normal_q
4:#SBATCH --qos=owl_b200_normal_base
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=8
9:#SBATCH --mem=64G
10:#SBATCH --gres=gpu:1
11:#SBATCH --time=04:00:00
12:#SBATCH --output=%x_%j.out
13:#SBATCH --error=%x_%j.err
36:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
38:: "${VENV:?set VENV to a vLLM environment}"
39:: "${MODEL_PATH:?set MODEL_PATH to a model snapshot}"
40:: "${CLIENT_PYTHON:?set CLIENT_PYTHON to a Python with graphharness installed}"
41:: "${OUT:?set OUT to a shared output directory}"
42:: "${SERVED_NAME:=local-model}"
43:: "${HOST:=127.0.0.1}"
44:: "${PORT:=8000}"
45:: "${MAX_MODEL_LEN:=8192}"
46:: "${GPU_MEMORY_UTILIZATION:=0.60}"
47:: "${MAX_NUM_SEQS:=8}"
48:: "${CUDA_MODULE:=cuda13.1/toolkit/13.1.1}"
49:: "${STAGES:=baseline optimize scaling}"
50:: "${START_HARNESS:=baseline_b_v2}"
51:: "${ROUNDS:=8}"
52:: "${TRAIN_PER_TASK:=6}"
53:: "${VAL_PER_TASK:=8}"
54:: "${SCALING_SPLITS:=test_small test_medium test_large test_stress}"
55:: "${SCALING_LIMIT_PER_TASK:=}"
56:: "${LIMIT_PER_TASK:=}"
57:: "${BEST_EVAL_LIMIT_PER_TASK:=}"
58:: "${ABLATION_HARNESS:=baseline_b_v2}"
59:: "${ABLATION_SPLIT:=test_large}"
60:: "${ABLATION_LIMIT_PER_TASK:=}"
61:: "${BENCH_SPECS:=ga_easy:data/grapharena_eval ga_hard:data/grapharena_eval graphqa_test:data/graphqa_eval}"
62:: "${BENCH_HARNESS:=baseline_b_v2}"
63:: "${BENCH_LIMIT_PER_TASK:=20}"
64:: "${HF_HOME:=${HOME}/.cache/huggingface}"
67:LOGS="$OUT/logs"
68:mkdir -p "$LOGS"
71:export GRAPH_HARNESS_SANDBOX_DIR="$OUT/sandbox"
73:export PYTHONPATH="$REPO/src"
75:module load "$CUDA_MODULE"
76:export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}"
77:export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
81:echo "job=${SLURM_JOB_ID} host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
82:echo "model=$MODEL_PATH served_name=$SERVED_NAME"
83:echo "stages=$STAGES out=$OUT"
87:"$VENV/bin/vllm" serve "$MODEL_PATH" \
88:  --served-model-name "$SERVED_NAME" \
89:  --host "$HOST" --port "$PORT" \
90:  --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" \
91:  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
92:  --max-num-seqs "$MAX_NUM_SEQS" --disable-log-requests \
93:  > "$LOGS/vllm_server.log" 2>&1 &
97:  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
100:  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
106:  tail -100 "$LOGS/vllm_server.log"
107:  kill "$SERVER_PID" 2>/dev/null || true
110:BASE="http://127.0.0.1:${PORT}/v1"
111:MODEL_ARGS=(--model "$SERVED_NAME" --base-url "$BASE")
123:has_stage() { [[ " $STAGES " == *" $1 "* ]]; }
127:  if [[ -n "$LIMIT_PER_TASK" ]]; then LIMIT_ARGS=(--limit-per-task "$LIMIT_PER_TASK"); fi
129:    run_stage "baseline $HARNESS" \
130:      "$CLIENT_PYTHON" -m graphharness.runner \
131:        --harness "$HARNESS" --split test_small "${LIMIT_ARGS[@]}" \
133:        --out "$OUT/baseline/$HARNESS" "${MODEL_ARGS[@]}" \
138:BEST="$OUT/optimizer/best_harness.json"
141:    "$CLIENT_PYTHON" -m graphharness.optimizer \
142:      --start "$START_HARNESS" --train-split train --val-split val \
143:      --rounds "$ROUNDS" --train-per-task "$TRAIN_PER_TASK" \
144:      --val-per-task "$VAL_PER_TASK" --max-steps 8 --workers 8 \
146:      --critic-model "$SERVED_NAME" --critic-max-tokens 1800 \
148:      --out "$OUT/optimizer" --cache "$OUT/cache.sqlite" "${MODEL_ARGS[@]}"
149:  if [[ -f "$BEST" ]]; then
150:    BEST_ARGS=(--harness "$BEST" --split test_small --max-steps 8 --workers 8
152:               --out "$OUT/optimizer/eval_best_test_small")
153:    if [[ -n "$BEST_EVAL_LIMIT_PER_TASK" ]]; then
154:      BEST_ARGS+=(--limit-per-task "$BEST_EVAL_LIMIT_PER_TASK")
157:      "$CLIENT_PYTHON" -m graphharness.runner "${BEST_ARGS[@]}" "${MODEL_ARGS[@]}" \
164:  if [[ -f "$BEST" ]]; then HARNESSES+=("$BEST"); fi
165:  read -r -a SPLIT_ARRAY <<< "$SCALING_SPLITS"
166:  SCALE_ARGS=(--harnesses "${HARNESSES[@]}" --splits "${SPLIT_ARRAY[@]}"
167:              --max-steps 8 --workers 8 --out "$OUT/scaling")
168:  if [[ -n "$SCALING_LIMIT_PER_TASK" ]]; then
169:    SCALE_ARGS+=(--limit-per-task "$SCALING_LIMIT_PER_TASK")
172:    "$CLIENT_PYTHON" -m graphharness.scaling "${SCALE_ARGS[@]}" "${MODEL_ARGS[@]}" \
173:      --max-tokens 768 --request-timeout 180 --cache "$OUT/cache.sqlite"
177:  ABL_ARGS=(--harness "$ABLATION_HARNESS" --split "$ABLATION_SPLIT"
178:            --max-steps 8 --workers 8 --out "$OUT/ablation")
179:  if [[ -n "$ABLATION_LIMIT_PER_TASK" ]]; then
180:    ABL_ARGS+=(--limit-per-task "$ABLATION_LIMIT_PER_TASK")
183:    "$CLIENT_PYTHON" -m graphharness.ablation "${ABL_ARGS[@]}" "${MODEL_ARGS[@]}" \
184:      --max-tokens 768 --request-timeout 180 --cache "$OUT/cache.sqlite"
188:  for spec in $BENCH_SPECS; do
190:    run_stage "bench $BENCH_HARNESS / $split" \
191:      "$CLIENT_PYTHON" -m graphharness.runner \
192:        --harness "$BENCH_HARNESS" --split "$split" --data-dir "$data_dir" \
193:        --limit-per-task "$BENCH_LIMIT_PER_TASK" --max-steps 8 --workers 8 \
194:        --out "$OUT/bench/$split" "${MODEL_ARGS[@]}" \
195:        --max-tokens 768 --request-timeout 180 --cache "$OUT/cache.sqlite"
199:kill "$SERVER_PID" 2>/dev/null || true
200:wait "$SERVER_PID" 2>/dev/null || true
201:cp "$LOGS/vllm_server.log" "$LOGS/vllm_server_${SLURM_JOB_ID}.log" 2>/dev/null || true
203:  cp "${SLURM_SUBMIT_DIR:-.}/gh_qwen_pipeline_${SLURM_JOB_ID}.${ext}" "$LOGS/" 2>/dev/null || true

## scripts/run_review_c.sbatch
2:#SBATCH --job-name=gh_cbackend
3:#SBATCH --partition=b200_normal_q
4:#SBATCH --qos=owl_b200_normal_short
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=8
9:#SBATCH --mem=64G
10:#SBATCH --gres=gpu:1
11:#SBATCH --time=01:00:00
12:#SBATCH --output=%x_%j.out
13:#SBATCH --error=%x_%j.err
15:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
16:: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
17:ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/review_c_backend; mkdir -p "$OUT/logs"
18:export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
19:export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
21:export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
22:export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
23:"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 \
25:  --disable-log-requests > "$OUT/logs/vllm_server.log" 2>&1 &
27:for i in $(seq 1 180); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 2; done
28:curl -sf http://127.0.0.1:8000/health >/dev/null || { tail -100 "$OUT/logs/vllm_server.log"; exit 1; }
30:"$CLIENT_PYTHON" -m graphharness.runner --harness "$OUT/harnesses/baseline_b_v2_c.json" \
32:  --out "$OUT/gpu_test_small" $MODEL_ARGS
33:kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true

## scripts/run_review_optimizers.sbatch
2:#SBATCH --job-name=gh_review_opt
3:#SBATCH --partition=b200_normal_q
4:#SBATCH --qos=owl_b200_normal_short
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=8
9:#SBATCH --mem=64G
10:#SBATCH --gres=gpu:1
11:#SBATCH --time=04:00:00
12:#SBATCH --output=%x_%j.out
13:#SBATCH --error=%x_%j.err
15:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
16:: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
19:LOGS=$OUT/logs; mkdir -p "$LOGS"
20:export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
21:: "${PORT:=8000}"
22:export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
24:export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
25:export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
26:"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port "$PORT" \
28:  --disable-log-requests > "$LOGS/vllm_server.log" 2>&1 &
31:  curl -sf http://127.0.0.1:${PORT}/health >/dev/null 2>&1 && { echo ready; break; }
32:  kill -0 $PID 2>/dev/null || break
35:curl -sf http://127.0.0.1:${PORT}/health >/dev/null || { tail -100 "$LOGS/vllm_server.log"; exit 1; }
36:MODEL_ARGS="--model qwen3-4b --base-url http://127.0.0.1:${PORT}/v1 --max-tokens 768 --request-timeout 180"
40:run "$CLIENT_PYTHON" -m graphharness.optimizer --start baseline_a --rrs --rounds 5 \
41:  --train-per-task 4 --val-per-task 3 --out "$OUT/rrsi_a" --cache "$OUT/cache.sqlite" $MODEL_ARGS
42:run "$CLIENT_PYTHON" -m graphharness.optimizer --start baseline_b_v2 --rrs --rounds 5 \
43:  --train-per-task 4 --val-per-task 3 --out "$OUT/rrsi_bv2" --cache "$OUT/cache.sqlite" $MODEL_ARGS
46:run "$CLIENT_PYTHON" -m graphharness.gepa --start baseline_a --rounds 3 --candidates-per-round 2 \
47:  --train-per-task 4 --val-per-task 4 --out "$OUT/gepa_a" --cache "$OUT/cache.sqlite" $MODEL_ARGS
50:for spec in "rrsi_a:$OUT/rrsi_a/best_harness.json" "rrsi_bv2:$OUT/rrsi_bv2/best_harness.json" "gepa_a:$OUT/gepa_a/best_harness.json"; do
52:  run "$CLIENT_PYTHON" -m graphharness.runner --harness "$harness" --split test_small \
54:    --out "$OUT/eval_$name" $MODEL_ARGS
58:run "$CLIENT_PYTHON" -m graphharness.router --library experiments/review_router/harnesses \
60:  --out "$OUT/router_routed" $MODEL_ARGS
61:run "$CLIENT_PYTHON" -m graphharness.router --library experiments/review_router/harnesses \
63:  --out "$OUT/router_decomposed" $MODEL_ARGS
65:kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true

## scripts/run_review_router.sbatch
2:#SBATCH --job-name=gh_router
3:#SBATCH --partition=b200_normal_q
4:#SBATCH --qos=owl_b200_normal_short
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=8
9:#SBATCH --mem=64G
10:#SBATCH --gres=gpu:1
11:#SBATCH --time=01:00:00
12:#SBATCH --output=%x_%j.out
13:#SBATCH --error=%x_%j.err
15:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
16:: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
17:ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/review_router; mkdir -p "$OUT/logs"
18:export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
19:export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
21:export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
22:export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
23:"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8001 \
25:  --disable-log-requests > "$OUT/logs/vllm_server.log" 2>&1 &
27:for i in $(seq 1 180); do curl -sf http://127.0.0.1:8001/health >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 2; done
28:curl -sf http://127.0.0.1:8001/health >/dev/null || { tail -100 "$OUT/logs/vllm_server.log"; exit 1; }
30:"$CLIENT_PYTHON" -m graphharness.runner --harness experiments/review_router/harnesses/baseline_b_v2.json \
32:  --no-resume --no-trajectories --out "$OUT/global" $M
33:"$CLIENT_PYTHON" -m graphharness.router --library experiments/review_router/harnesses \
35:  --out "$OUT/routed" $M
36:"$CLIENT_PYTHON" -m graphharness.router --library experiments/review_router/harnesses \
38:  --out "$OUT/decomposed" $M
39:kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true

## scripts/run_review_rrsi2.sbatch
2:#SBATCH --job-name=gh_rrsi2
3:#SBATCH --partition=b200_normal_q
4:#SBATCH --qos=owl_b200_normal_short
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=8
9:#SBATCH --mem=64G
10:#SBATCH --gres=gpu:1
11:#SBATCH --time=01:30:00
12:#SBATCH --output=%x_%j.out
13:#SBATCH --error=%x_%j.err
15:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
16:: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
17:ROOT=/projects/psgroup/saminrk/gh-qwen; OUT=experiments/review_optimizers2; mkdir -p "$OUT/logs"
18:export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
19:export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
21:export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
22:export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
23:"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 \
25:  --disable-log-requests > "$OUT/logs/vllm_server.log" 2>&1 &
27:for i in $(seq 1 180); do curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break; kill -0 $PID 2>/dev/null || break; sleep 2; done
28:curl -sf http://127.0.0.1:8000/health >/dev/null || { tail -100 "$OUT/logs/vllm_server.log"; exit 1; }
32:  "$CLIENT_PYTHON" -m graphharness.optimizer --start "$start" --rrs --rounds 3 \
33:    --train-per-task 4 --val-per-task 3 --out "$OUT/$name" --cache "$OUT/cache.sqlite" $MODEL_ARGS
34:  "$CLIENT_PYTHON" -m graphharness.runner --harness "$OUT/$name/best_harness.json" \
36:    --out "$OUT/eval_$name" $MODEL_ARGS
38:kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true

## scripts/run_review_tool_access.sbatch
2:#SBATCH --job-name=gh_review_tools
3:#SBATCH --partition=b200_normal_q
4:#SBATCH --qos=owl_b200_normal_short
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=8
9:#SBATCH --mem=64G
10:#SBATCH --gres=gpu:1
11:#SBATCH --time=02:00:00
12:#SBATCH --output=%x_%j.out
13:#SBATCH --error=%x_%j.err
15:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
16:: "${VENV:?}"; : "${MODEL_PATH:?}"; : "${CLIENT_PYTHON:?}"
19:LOGS=$OUT/logs; mkdir -p "$LOGS"
20:export HF_HOME=$ROOT/hf PYTHONUNBUFFERED=1 GRAPH_HARNESS_API_KEY=EMPTY
21:export GRAPH_HARNESS_SANDBOX_DIR=$OUT/sandbox PYTHONPATH="$PWD/src"
23:export CUDA_HOME=${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}
24:export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH" TOKENIZERS_PARALLELISM=false
25:"$VENV/bin/vllm" serve "$MODEL_PATH" --served-model-name qwen3-4b --host 127.0.0.1 --port 8000 \
27:  --disable-log-requests > "$LOGS/vllm_server.log" 2>&1 &
31:  kill -0 $PID 2>/dev/null || break
34:curl -sf http://127.0.0.1:8000/health >/dev/null || { tail -100 "$LOGS/vllm_server.log"; exit 1; }
39:    "$CLIENT_PYTHON" -m graphharness.runner --harness "$OUT/harnesses/$cond.json" \
42:      --max-tokens 768 --request-timeout 180 --cache "$OUT/cache.sqlite" \
43:      --out "$OUT/runs/$cond/$split"
46:kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true

## scripts/run_scaling.sbatch
2:#SBATCH --job-name=cs5624_scale
3:#SBATCH --partition=a30_normal_q
4:#SBATCH --account=psgroup
5:#SBATCH --nodes=1
6:#SBATCH --ntasks=1
7:#SBATCH --cpus-per-task=8
8:#SBATCH --mem=32G
9:#SBATCH --gres=gpu:1
10:#SBATCH --time=24:00:00
11:#SBATCH --output=%x_%j.out
12:#SBATCH --error=%x_%j.err
19:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
21:: "${HARNESSES:=baseline_a baseline_b}"
22:: "${SPLITS:=test_small test_medium test_large test_stress}"
23:: "${MODEL:?set MODEL to the model name served by the endpoint}"
24:: "${BASE_URL:=${GRAPH_HARNESS_BASE_URL:-http://127.0.0.1:8000/v1}}"
25:: "${WORKERS:=8}"
26:: "${LIMIT_PER_TASK:=}"
27:: "${OUT:=experiments/scaling}"
30:export GRAPH_HARNESS_MODEL="${MODEL}"
31:export GRAPH_HARNESS_BASE_URL="${BASE_URL}"
33:read -r -a HARNESS_ARRAY <<< "${HARNESSES}"
34:read -r -a SPLIT_ARRAY <<< "${SPLITS}"
35:ARGS=(--harnesses "${HARNESS_ARRAY[@]}" --splits "${SPLIT_ARRAY[@]}"
36:      --out "${OUT}" --model "${MODEL}" --base-url "${BASE_URL}"
37:      --workers "${WORKERS}" --cache)
38:if [[ -n "${LIMIT_PER_TASK}" ]]; then
39:  ARGS+=(--limit-per-task "${LIMIT_PER_TASK}")
42:echo "harnesses=${HARNESSES} splits=${SPLITS} model=${MODEL}"
43:python3 -m graphharness.scaling "${ARGS[@]}"

## scripts/run_tests.sbatch
2:#SBATCH --job-name=cs5624_tests
3:#SBATCH --partition=a30_normal_q
4:#SBATCH --account=psgroup
5:#SBATCH --nodes=1
6:#SBATCH --ntasks=1
7:#SBATCH --cpus-per-task=4
8:#SBATCH --mem=16G
9:#SBATCH --gres=gpu:1
10:#SBATCH --time=00:30:00
11:#SBATCH --output=%x_%j.out
12:#SBATCH --error=%x_%j.err
16:echo "Job ID: ${SLURM_JOB_ID}"
25:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
28:export PATH="${HOME}/.local/bin:${PATH}"

## scripts/serve_llama_cpp_cpu.sbatch
2:#SBATCH --job-name=gh_llama_cpu
3:#SBATCH --partition=normal_q
4:#SBATCH --qos=owl_normal_base
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=96
9:#SBATCH --mem=512G
10:#SBATCH --time=08:00:00
11:#SBATCH --output=%x_%j.out
12:#SBATCH --error=%x_%j.err
24:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
26:: "${LLAMA_VENV:?set LLAMA_VENV to a venv with llama-cpp-python[server]}"
27:: "${MODEL_GGUF:?set MODEL_GGUF to the .gguf file}"
28:: "${MODEL_ALIAS:=qwen2.5-7b}"
29:: "${HOST:=0.0.0.0}"
30:: "${PORT:=8001}"
31:: "${N_CTX:=8192}"
32:: "${N_THREADS:=64}"
33:: "${N_THREADS_BATCH:=96}"
34:: "${CHAT_FORMAT:=chatml}"
37:echo "host=$(hostname) model=${MODEL_GGUF}"
38:echo "url=http://$(hostname):${PORT}/v1"
41:exec "${LLAMA_VENV}/bin/python" -m llama_cpp.server \
42:  --model "${MODEL_GGUF}" \
43:  --model_alias "${MODEL_ALIAS}" \
44:  --host "${HOST}" --port "${PORT}" \
45:  --n_ctx "${N_CTX}" --n_batch 512 \
46:  --n_threads "${N_THREADS}" --n_threads_batch "${N_THREADS_BATCH}" \
47:  --chat_format "${CHAT_FORMAT}"

## scripts/serve_vllm.sbatch
2:#SBATCH --job-name=gh_vllm
3:#SBATCH --partition=b200_normal_q
4:#SBATCH --qos=owl_b200_normal_base
5:#SBATCH --account=psgroup
6:#SBATCH --nodes=1
7:#SBATCH --ntasks=1
8:#SBATCH --cpus-per-task=8
9:#SBATCH --mem=64G
10:#SBATCH --gres=gpu:1
11:#SBATCH --time=08:00:00
12:#SBATCH --output=%x_%j.out
13:#SBATCH --error=%x_%j.err
28:cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
30:: "${VENV:?set VENV to a Python 3.12 venv with vllm installed}"
31:: "${MODEL_PATH:?set MODEL_PATH to the model snapshot directory}"
32:: "${SERVED_NAME:=local-model}"
33:: "${HOST:=0.0.0.0}"
34:: "${PORT:=8000}"
35:: "${MAX_MODEL_LEN:=8192}"
36:: "${GPU_MEMORY_UTILIZATION:=0.60}"
37:: "${MAX_NUM_SEQS:=8}"
38:: "${CUDA_MODULE:=cuda13.1/toolkit/13.1.1}"
39:: "${HF_HOME:=${HOME}/.cache/huggingface}"
41:module load "${CUDA_MODULE}"
42:export CUDA_HOME="${CUDA_HOME:-/cm/shared/apps/cuda13.1/toolkit/13.1.1}"
43:export PATH="${VENV}/bin:${CUDA_HOME}/bin:${PATH}"
49:echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none}"
50:echo "model=${MODEL_PATH}"
51:echo "url=http://$(hostname):${PORT}/v1"
55:exec "${VENV}/bin/vllm" serve "${MODEL_PATH}" \
56:  --served-model-name "${SERVED_NAME}" \
57:  --host "${HOST}" --port "${PORT}" \
59:  --max-model-len "${MAX_MODEL_LEN}" \
60:  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
61:  --max-num-seqs "${MAX_NUM_SEQS}" \


## Resolved interface notes (2026-11-21)

- Harness registry: `graphharness.agent.HARNESSES` contains `baseline_a`,
  `baseline_b`, `baseline_b_v2`; `get_harness(name_or_json_path)` loads a saved
  harness. Router library JSONs are under `experiments/review_router/harnesses/`.
- Runner model endpoint flags: `--model NAME --base-url URL --api-key-env VAR`.
- Per-run summary file: `<out>/summary.json` and `<out>/summary.txt`.
- Runner concurrency: `--workers N`; runs are resumable with `predictions.jsonl`.
- Existing jobs use `--partition=b200_normal_q`, `--account=psgroup`,
  `--qos=owl_b200_normal_short`; CPU jobs use `normal_q` + `owl_normal_base`.
- `optimizer.py` and `gepa.py` support `--critic-model` and
  `--critic-base-url`; `meta_harness.py` supports `--critic-model` but not a
  separate critic URL (needs [CREATE] for a separate endpoint).
- `build_data` has no `--seed`/`--split-name`; E1 uses a separate config file
  with a new seed/split name, preserving the default manifest.
- Adapters use `--limit N` per task (GraphArena per difficulty; GraphQA per
  source split), not `--per-task`.
- E3 critic: no gpt-oss/14B on disk; use Qwen2.5-7B-Instruct BF16 as the
  minimum stronger critic, as allowed by the runbook.
