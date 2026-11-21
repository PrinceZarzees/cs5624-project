# Accuracy by graph characteristics

## Baseline A  (n=450, accuracy=80.9%)

### Density

| density_bin | n | mean nodes | accuracy | feasible | parsed |
|---|---:|---:|---:|---:|---:|
| dense (.10-.30) | 73 | 24 | 87.7% | 87.7% | 100.0% |
| medium (.03-.10) | 74 | 59 | 86.5% | 91.9% | 100.0% |
| sparse (.01-.03) | 100 | 197 | 78.0% | 84.0% | 99.0% |
| very dense (>.30) | 21 | 8 | 81.0% | 81.0% | 100.0% |
| very sparse (<.01) | 182 | 994 | 77.5% | 83.0% | 95.6% |

### Topology / family

| family | n | mean nodes | accuracy | feasible | parsed |
|---|---:|---:|---:|---:|---:|
| ba | 95 | 477 | 74.7% | 85.3% | 98.9% |
| er | 85 | 494 | 85.9% | 89.4% | 95.3% |
| mixed | 90 | 442 | 100.0% | 100.0% | 100.0% |
| tree | 90 | 459 | 62.2% | 65.6% | 98.9% |
| ws | 90 | 426 | 82.2% | 86.7% | 96.7% |

### Task

| task | n | mean nodes | accuracy | feasible | parsed |
|---|---:|---:|---:|---:|---:|
| connected_components | 90 | 442 | 100.0% | 100.0% | 100.0% |
| cycle_detection | 90 | 459 | 62.2% | 65.6% | 98.9% |
| diameter | 90 | 450 | 77.8% | 87.8% | 94.4% |
| max_clique | 90 | 480 | 80.0% | 88.9% | 100.0% |
| shortest_path | 90 | 466 | 84.4% | 84.4% | 96.7% |

## Baseline B  (n=450, accuracy=87.3%)

### Density

| density_bin | n | mean nodes | accuracy | feasible | parsed |
|---|---:|---:|---:|---:|---:|
| dense (.10-.30) | 73 | 24 | 94.5% | 94.5% | 100.0% |
| medium (.03-.10) | 74 | 59 | 91.9% | 91.9% | 100.0% |
| sparse (.01-.03) | 100 | 197 | 92.0% | 97.0% | 100.0% |
| very dense (>.30) | 21 | 8 | 95.2% | 95.2% | 100.0% |
| very sparse (<.01) | 182 | 994 | 79.1% | 84.6% | 94.0% |

### Topology / family

| family | n | mean nodes | accuracy | feasible | parsed |
|---|---:|---:|---:|---:|---:|
| ba | 95 | 477 | 90.5% | 93.7% | 98.9% |
| er | 85 | 494 | 85.9% | 90.6% | 96.5% |
| mixed | 90 | 442 | 82.2% | 84.4% | 96.7% |
| tree | 90 | 459 | 86.7% | 92.2% | 98.9% |
| ws | 90 | 426 | 91.1% | 92.2% | 96.7% |

### Task

| task | n | mean nodes | accuracy | feasible | parsed |
|---|---:|---:|---:|---:|---:|
| connected_components | 90 | 442 | 82.2% | 84.4% | 96.7% |
| cycle_detection | 90 | 459 | 86.7% | 92.2% | 98.9% |
| diameter | 90 | 450 | 90.0% | 90.0% | 96.7% |
| max_clique | 90 | 480 | 83.3% | 92.2% | 97.8% |
| shortest_path | 90 | 466 | 94.4% | 94.4% | 97.8% |

## Baseline B v2  (n=450, accuracy=88.4%)

### Density

| density_bin | n | mean nodes | accuracy | feasible | parsed |
|---|---:|---:|---:|---:|---:|
| dense (.10-.30) | 73 | 24 | 93.2% | 97.3% | 100.0% |
| medium (.03-.10) | 74 | 59 | 91.9% | 95.9% | 100.0% |
| sparse (.01-.03) | 100 | 197 | 91.0% | 97.0% | 100.0% |
| very dense (>.30) | 21 | 8 | 90.5% | 90.5% | 100.0% |
| very sparse (<.01) | 182 | 994 | 83.5% | 91.8% | 95.1% |

### Topology / family

| family | n | mean nodes | accuracy | feasible | parsed |
|---|---:|---:|---:|---:|---:|
| ba | 95 | 477 | 80.0% | 92.6% | 98.9% |
| er | 85 | 494 | 81.2% | 91.8% | 95.3% |
| mixed | 90 | 442 | 100.0% | 100.0% | 100.0% |
| tree | 90 | 459 | 95.6% | 95.6% | 100.0% |
| ws | 90 | 426 | 85.6% | 92.2% | 95.6% |

### Task

| task | n | mean nodes | accuracy | feasible | parsed |
|---|---:|---:|---:|---:|---:|
| connected_components | 90 | 442 | 100.0% | 100.0% | 100.0% |
| cycle_detection | 90 | 459 | 95.6% | 95.6% | 100.0% |
| diameter | 90 | 450 | 83.3% | 87.8% | 93.3% |
| max_clique | 90 | 480 | 65.6% | 91.1% | 98.9% |
| shortest_path | 90 | 466 | 97.8% | 97.8% | 97.8% |

