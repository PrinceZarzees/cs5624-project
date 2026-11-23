# Optimizer setup

- HDA model: Qwen3-4B-Instruct-2507, BF16 served by vLLM on Falcon a30 for
  O3-O5. The CPU dry runs used the Q4_K_M llama.cpp build of the same
  checkpoint. All five final prompts use the BF16 GPU run.
- Proposer: the same Qwen3-4B-Instruct-2507 checkpoint (BF16 on GPU, Q4 on the
  CPU dry runs). No gpt-oss/14B+ model was on disk at run time. Recorded as a
  limitation; all four optimizers use the same proposer.
- Frozen HDA config: `hda_pipeline/configs/opt_hda.json`.
- Editable fields: `strategy` (3,000 chars), `hidden_procedure` (1,500),
  `iteration_note` (800).
- Budget: 240 uncached HDA rollouts + 40 re-check rollouts per optimizer;
  proposer ≤ 60 calls.
- Seeds: optimization 100–199 (reflection) and 200–202 (acceptance); final
  comparison 0–2.
- `val_opt`: new 400-record split, 0 graph overlap with all other splits.

O3-O5 completed on Falcon a30 on 2026-11-22. Baseline 0.20 GPU-h, GEPA 1.01,
SkillOpt 1.11, Meta-Harness 0.88, EvoSkill 0.77, final grid 0.91; 4.87 GPU-h
total. All four optimizers accepted zero edits.
