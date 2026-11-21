# E9 cost assumptions

- Token prices: $0.50/M input, $1.50/M output.
- Optimizer logs store total tokens only; assume 80% prompt / 20% completion.
- GPU-hour price: $4.00/B200-hour (cloud reference; ARC is a class allocation).
- Counterfactual protocol: 20 train + 50 val problems per round, 10 rounds.
- `est_gpu_hours` uses 0.8M tokens per B200-hour, measured from the optimizer pilots.
- Counterfactual is arithmetic from measured per-problem tokens, not a new run.
