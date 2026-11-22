# Optimizer setup

- HDA model: Qwen3-4B-Instruct-2507 Q4_K_M on CPU, 32k context.
- Proposer: same Qwen3-4B Q4 for now (no gpt-oss/14B+ on disk; GPU queue
  blocked). Recorded as a limitation; all four optimizers use the same
  proposer.
- Frozen HDA config: `hda_pipeline/configs/opt_hda.json`.
- Editable fields: `strategy` (3,000 chars), `hidden_procedure` (1,500),
  `iteration_note` (800).
- Budget: 240 uncached HDA rollouts + 40 re-check rollouts per optimizer;
  proposer ≤ 60 calls.
- Seeds: optimization 100–199 (reflection) and 200–202 (acceptance); final
  comparison 0–2.
- `val_opt`: new 400-record split, 0 graph overlap with all other splits.

GPU status at setup: B200 nodes were reserved; only the Q4 CPU server was
available. O3–O5 are therefore pending a GPU/throughput decision.
