# Benchmark coverage

Updated after completing the missing task implementations.

## Proposal-defined tasks

| Task | Kind | Status |
|---|---|---|
| Shortest path | P | implemented + tested |
| Connected components | P | implemented + tested |
| Cycle detection | P | implemented + tested |
| Max clique | NP | implemented + tested |
| Diameter | P | implemented + tested |
| Reachability | P | implemented + tested |
| Hamiltonian path | NP | implemented + tested in `graphharness.tasks` |

The proposal's "Max Clique **or** Hamiltonian Path" requirement is satisfied
twice: both are now available in the authoritative package.

## GraphArena: 10 of 10 tasks

| GraphArena task | graphharness task | Record shape |
|---|---|---|
| Distance | `shortest_path` | single graph |
| Connected | `connected_components` | single graph |
| Diameter | `diameter` | single graph |
| MCP | `max_clique` | single graph |
| Neighbor | `common_neighbors` | single graph |
| MIS | `max_independent_set` | single graph |
| MVC | `min_vertex_cover` | single graph |
| GED | `graph_edit_distance` | two graphs + node labels |
| MCS | `max_common_subgraph` | two graphs |
| TSP | `traveling_salesman` | weighted graph |

Real-data smoke: `data/grapharena_smoke/ga_easy.jsonl` and `ga_hard.jsonl`,
2 problems per task (40 records). All stored truths score correctly. A larger
20-per-task conversion is in `data/grapharena_eval/` (400 records), also fully
verified.

Hard GED/MCS/TSP instances outside the exact small-instance limits retain the
source answer/path and mark `meta["ground_truth_source"]` accordingly.

## GraphQA: 12 of 12 tasks

| GraphQA task | graphharness task | Answer type |
|---|---|---|
| shortest_path | `shortest_path` | node list |
| cycle_check | `cycle_detection` | node list |
| reachability | `reachability` | node list |
| connected_nodes | `connected_nodes` | direct-neighbor list |
| disconnected_nodes | `disconnected_nodes` | non-neighbor list |
| node_count | `node_count` | number |
| node_degree | `node_degree` | number |
| edge_count | `edge_count` | number |
| edge_existence | `edge_existence` | yes/no |
| triangle_counting | `triangle_counting` | number |
| node_classification | `node_classification` | label string |
| maximum_flow | `maximum_flow` | number |

Real-data smoke: `data/graphqa_smoke/` (2 problems per task, 66 records across
the published splits); all stored truths score correctly. A 20-per-task
conversion is in `data/graphqa_eval/` (660 records), also fully verified.

Node classification and maximum flow use the GraphQA source answer because the
question text does not expose enough information to recompute the generator's
semantics; `meta["ground_truth_source"]` is `graphqa_source` for those records.
All other GraphQA records are recomputed by the graphharness solvers.

## Execution status

- Synthetic experiments: all reported scaling/optimizer/ablation results.
- Real GraphArena: 400 converted records (20 per task per difficulty).
- Real GraphQA: 660 converted records (20 per task per published split).
- Real benchmark model evaluation: Slurm job `1540814` completed
  `baseline_b_v2` on `ga_easy`, `ga_hard`, and `graphqa_test` with 20 problems
  per task. Accuracy: GraphArena easy 74.5%, hard 61.0%, GraphQA test 88.3%.
  Artifacts: `experiments/qwen3_4b_real_bench/`.
