# Optimization cost accounting

| run | rounds | initial val | best val | wall time | tokens | est. cost |
|---|---:|---:|---:|---:|---:|---:|
| from_baseline_a | 11 | 87.5% | 87.5% | 486s | 4,006,987 | $2.80 |
| from_baseline_b_v2 | 9 | 97.5% | 97.5% | 383s | 3,526,321 | $2.47 |

## from_baseline_a
| round | role | train | val | best val | accepted | round tokens | cumulative |
|---:|---|---:|---:|---:|---|---:|---:|
| 0 | initial |  | 87.5% | 87.5% | True | 137,042 | 137,042 |
| 1 |  | 80.0% | 70.0% | 87.5% | False | 392,496 | 529,538 |
| 2 |  | 83.3% | 60.0% | 87.5% | False | 377,058 | 906,596 |
| 3 |  | 93.3% | 62.5% | 87.5% | False | 325,611 | 1,232,207 |
| 4 |  | 90.0% | 62.5% | 87.5% | False | 318,775 | 1,550,982 |
| 5 |  | 93.3% | 72.5% | 87.5% | False | 358,505 | 1,909,487 |
| 6 |  | 76.7% | 37.5% | 87.5% | False | 552,404 | 2,461,891 |
| 7 |  | 90.0% | 52.5% | 87.5% | False | 431,838 | 2,893,729 |
| 8 |  | 83.3% | 42.5% | 87.5% | False | 383,482 | 3,277,211 |
| 9 |  | 83.3% | 70.0% | 87.5% | False | 404,824 | 3,682,035 |
| 10 |  | 93.3% | 72.5% | 87.5% | False | 324,952 | 4,006,987 |

## from_baseline_b_v2
| round | role | train | val | best val | accepted | round tokens | cumulative |
|---:|---|---:|---:|---:|---|---:|---:|
| 0 | initial |  | 97.5% | 97.5% | True | 201,734 | 201,734 |
| 1 |  | 93.3% | 72.5% | 97.5% | False | 338,214 | 539,948 |
| 2 |  | 90.0% | 67.5% | 97.5% | False | 451,520 | 991,468 |
| 3 |  | 96.7% | 45.0% | 97.5% | False | 407,050 | 1,398,518 |
| 4 |  | 100.0% | 72.5% | 97.5% | False | 406,399 | 1,804,917 |
| 5 |  | 86.7% | 75.0% | 97.5% | False | 421,630 | 2,226,547 |
| 6 |  | 93.3% | 62.5% | 97.5% | False | 448,576 | 2,675,123 |
| 7 |  | 90.0% | 70.0% | 97.5% | False | 440,336 | 3,115,459 |
| 8 |  | 93.3% | 67.5% | 97.5% | False | 410,862 | 3,526,321 |

Cost estimate uses configurable per-million input/output rates; the optimizer history stores total tokens, so the split is an assumption.
