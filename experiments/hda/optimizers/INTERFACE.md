# HDA optimizer interface (O0)

## Prompt constants (`hda/prompts.py`) and use in `hda/agent.py`

| Constant | Use |
|---|---|
| `SYSTEM` | `sys_msg = SYSTEM.replace("{template}", TEMPLATE)` |
| `STATEMENT_GIVEN` | `.format(statement=task.statement)` |
| `STATEMENT_HIDDEN` | raw string; appended to the hidden user prompt |
| `EXAMPLES` | `.format(k, total, blocks)` |
| `ITERATION` | `.format(it, n_iter, feedback, best_block, statement, steps)` |
| `FEEDBACK` / `FEEDBACK_Q` | `.format(...)` in submission feedback |
| `JUDGE` | hidden-statement judge prompt |
| `STRATEGY_NOTE` | appended to the system prompt as `Strategy notes:` |
| `HIDDEN_NOTE` | appended to the hidden user prompt |
| `ITERATION_NOTE` | appended after the formatted iteration block |

`HDA_PROMPT_FILE` is read when `hda.prompts` is imported. A candidate has exactly
`strategy`, `hidden_procedure`, `iteration_note`; empty fields are a no-op.

## Frozen HDA config (`configs/opt_hda.json`)

- Model: Qwen3-4B-Instruct-2507 Q4_K_M (llama.cpp), base URL `127.0.0.1:8002/v1`
- k=5, n_iter=3, steps=8, temperature=0.2, max_tokens=4096, context 32768
- data `data/hda`, optimization seed 200
- Test evaluation off: pass `--test-splits` with no values (argparse `nargs="*"` → `[]`)

## Counters

- HDA tokens: `final.json["tokens"]` (`prompt`, `completion`)
- HDA LLM calls: count `"kind": "reply"` events in `log.jsonl`
- Wall time: `final.json["duration_s"]` or the rollout timer
- Proposer tokens/calls: recorded by each optimizer's ledger

## Existing optimizer code

`src/graphharness/optimizer.py`, `gepa.py`, `meta_harness.py` (NetworkX work)
provide proposal parsing, critic endpoint flags, validators and token
accounting. We reused the proposal/JSON parsing ideas but implemented the HDA
loops in `hda/opt/` because the HDA rollout and scoring interfaces differ.

## val_opt

`val_opt` has 400 records (40/task, 4–50 nodes), seed offset 12. A graph-hash
check over `examples`, `val`, `test_*` and `val_opt` reports **0 overlap**.
Optimizer score = accuracy of the rollout's `best_solution.cpp` on `val_opt`
(relabeling check, same penalty as the agent).
