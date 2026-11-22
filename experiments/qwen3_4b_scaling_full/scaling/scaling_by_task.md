# Scaling accuracy by task (Qwen3-4B BF16)

## baseline_a

| split | shortest_path | connected_components | cycle_detection | diameter | max_clique |
|---|---|---|---|---|---|
| test_small | 76.7% | 100.0% | 70.0% | 96.7% | 93.3% |
| test_medium | 83.3% | 100.0% | 63.3% | 76.7% | 80.0% |
| test_large | 100.0% | 100.0% | 45.0% | 85.0% | 75.0% |
| test_stress | 80.0% | 100.0% | 70.0% | 10.0% | 50.0% |

## baseline_b

| split | shortest_path | connected_components | cycle_detection | diameter | max_clique |
|---|---|---|---|---|---|
| test_small | 90.0% | 83.3% | 100.0% | 96.7% | 100.0% |
| test_medium | 100.0% | 96.7% | 76.7% | 100.0% | 80.0% |
| test_large | 100.0% | 80.0% | 85.0% | 90.0% | 75.0% |
| test_stress | 80.0% | 40.0% | 80.0% | 40.0% | 60.0% |

## baseline_b_v2

| split | shortest_path | connected_components | cycle_detection | diameter | max_clique |
|---|---|---|---|---|---|
| test_small | 100.0% | 100.0% | 100.0% | 90.0% | 80.0% |
| test_medium | 100.0% | 100.0% | 93.3% | 93.3% | 63.3% |
| test_large | 100.0% | 100.0% | 95.0% | 90.0% | 40.0% |
| test_stress | 80.0% | 100.0% | 90.0% | 20.0% | 80.0% |

