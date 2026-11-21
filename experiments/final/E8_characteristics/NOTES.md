# E8 characteristics

- Rows: 1350 scored problems (A/B/v2 x 4 scaling splits).
- `family_x_size.csv`: family x node-size bin per harness, Wilson CI, n, small-n flag.
- `regression.csv`: logistic `correct ~ log(n) + 10*density + task + harness`.
- `regression_family.csv`: same with family instead of task.
- family and task are perfectly nested by construction: `tree` only occurs
  with cycle_detection and `mixed` only with connected_components, so a model
  containing both is not identified. Report one at a time.
- Reference levels: task=shortest_path / family=er, harness=baseline_a.

| term | coef | 95% CI | odds ratio |
|---|---:|---|---:|
| intercept | 4.761 | [3.797, 5.726] | 116.920 |
| log_n | -0.456 | [-0.591, -0.321] | 0.634 |
| density | -0.274 | [-0.442, -0.107] | 0.760 |
| task[connected_components] | 0.256 | [-0.427, 0.940] | 1.292 |
| task[cycle_detection] | -1.092 | [-1.647, -0.537] | 0.335 |
| task[diameter] | -0.908 | [-1.470, -0.346] | 0.403 |
| task[max_clique] | -1.317 | [-1.855, -0.779] | 0.268 |
| harness[baseline_b] | 0.533 | [0.152, 0.913] | 1.703 |
| harness[baseline_b_v2] | 0.645 | [0.256, 1.033] | 1.906 |
