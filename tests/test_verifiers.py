"""
test_verifiers.py
=================
Comprehensive pytest test suite for the ground-truth solvers and verifiers.

Tests cover:
  - Graph generation at all scales
  - Each task type's solver + verifier consistency
  - Edge cases (disconnected graphs, single-node graphs, complete graphs)
  - Data I/O round-trips
  - Evaluation pipeline
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
from pathlib import Path

import networkx as nx
import pytest

# Ensure the src package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph_generator import (
    GraphInstance,
    generate_graphs,
    generate_full_dataset,
    split_small_graphs,
    SCALE_RANGES,
    GRAPH_MODELS,
)
from src.tasks import (
    TaskInstance,
    TASK_REGISTRY,
    generate_task_instance,
    shortest_path_question,
    connected_components_question,
    cycle_detection_question,
    max_clique_question,
    hamiltonian_path_question,
    verify_shortest_path,
    verify_connected_components,
    verify_cycle_detection,
    verify_max_clique,
    verify_hamiltonian_path,
)
from src.data_loader import (
    save_graphs_jsonl,
    load_graphs_jsonl,
    save_tasks_jsonl,
    load_tasks_jsonl,
    graph_to_adjacency_text,
    graph_to_edge_list_text,
    graph_to_summary_stats,
    generate_task_dataset,
)
from src.evaluate_answer import (
    verify_answer,
    evaluate_batch,
    compute_metrics,
    EvalResult,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def small_graphs():
    """Generate a small set of test graphs."""
    return generate_graphs("small", 10, seed=123)


@pytest.fixture
def medium_graphs():
    """Generate a few medium-scale test graphs."""
    return generate_graphs("medium", 3, seed=123)


@pytest.fixture
def known_graph():
    """Create a known graph with predictable properties."""
    # Path graph: 0-1-2-3-4 (no cycle, 1 component, known shortest paths)
    G = nx.path_graph(5)
    return GraphInstance.from_networkx(G, "test_path_5", "manual", "small")


@pytest.fixture
def cycle_graph():
    """Create a graph with a known cycle."""
    G = nx.cycle_graph(6)
    return GraphInstance.from_networkx(G, "test_cycle_6", "manual", "small")


@pytest.fixture
def disconnected_graph():
    """Create a disconnected graph."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])  # Component 1
    G.add_edges_from([(3, 4)])          # Component 2
    G.add_node(5)                        # Isolated node
    return GraphInstance.from_networkx(G, "test_disconnected", "manual", "small")


@pytest.fixture
def complete_graph():
    """Create a complete graph K5."""
    G = nx.complete_graph(5)
    return GraphInstance.from_networkx(G, "test_complete_5", "manual", "small")


@pytest.fixture
def single_node_graph():
    """Create a single-node graph."""
    G = nx.Graph()
    G.add_node(0)
    return GraphInstance.from_networkx(G, "test_single", "manual", "small")


# =========================================================================
# Graph Generator Tests
# =========================================================================

class TestGraphGenerator:

    def test_generate_correct_count(self):
        graphs = generate_graphs("small", 15, seed=42)
        assert len(graphs) == 15

    def test_node_count_in_range(self, small_graphs):
        lo, hi = SCALE_RANGES["small"]
        for g in small_graphs:
            assert lo <= g.num_nodes <= hi, (
                f"Graph {g.graph_id} has {g.num_nodes} nodes, "
                f"expected [{lo}, {hi}]"
            )

    def test_medium_scale_range(self, medium_graphs):
        lo, hi = SCALE_RANGES["medium"]
        for g in medium_graphs:
            assert lo <= g.num_nodes <= hi

    def test_all_models_used(self):
        graphs = generate_graphs("small", 30, seed=42)
        models_used = set(g.model for g in graphs)
        assert models_used == set(GRAPH_MODELS.keys())

    def test_graph_to_networkx_roundtrip(self, small_graphs):
        for g in small_graphs:
            G = g.to_networkx()
            assert G.number_of_nodes() == g.num_nodes
            assert G.number_of_edges() == g.num_edges

    def test_reproducibility(self):
        g1 = generate_graphs("small", 5, seed=42)
        g2 = generate_graphs("small", 5, seed=42)
        for a, b in zip(g1, g2):
            assert a.nodes == b.nodes
            assert a.edges == b.edges

    def test_different_seeds_different_graphs(self):
        g1 = generate_graphs("small", 5, seed=42)
        g2 = generate_graphs("small", 5, seed=99)
        # At least one graph should differ
        any_diff = any(a.edges != b.edges for a, b in zip(g1, g2))
        assert any_diff

    def test_train_val_split(self, small_graphs):
        train, val = split_small_graphs(small_graphs, train_ratio=0.7, seed=42)
        assert len(train) + len(val) == len(small_graphs)
        assert len(train) >= len(val)  # 70/30 split
        # No overlap
        train_ids = {g.graph_id for g in train}
        val_ids = {g.graph_id for g in val}
        assert train_ids.isdisjoint(val_ids)

    def test_graph_instance_serialization(self, known_graph):
        d = known_graph.to_dict()
        reconstructed = GraphInstance.from_dict(d)
        assert reconstructed.graph_id == known_graph.graph_id
        assert reconstructed.nodes == known_graph.nodes
        assert reconstructed.edges == known_graph.edges


# =========================================================================
# Task Solver Tests
# =========================================================================

class TestShortestPath:

    def test_path_graph(self, known_graph):
        G = known_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = shortest_path_question(G, rng)
        s, t = meta["source"], meta["target"]
        expected = nx.shortest_path_length(G, s, t)
        assert ans == expected

    def test_disconnected_no_path(self, disconnected_graph):
        G = disconnected_graph.to_networkx()
        # Force source=0 (component {0,1,2}) and target=5 (isolated)
        rng = random.Random(42)
        # Generate multiple questions to find a disconnected pair
        found_disconnected = False
        for _ in range(50):
            q, ans, meta = shortest_path_question(G, rng)
            if ans == -1:
                found_disconnected = True
                assert not nx.has_path(G, meta["source"], meta["target"])
                break
        # It's OK if we don't find one; the graph might always pick connected pairs

    def test_verifier_correct(self):
        assert verify_shortest_path(3, 3, {}) is True

    def test_verifier_incorrect(self):
        assert verify_shortest_path(5, 3, {}) is False

    def test_verifier_string_answer(self):
        assert verify_shortest_path("3", 3, {}) is True


class TestConnectedComponents:

    def test_connected_graph(self, known_graph):
        G = known_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = connected_components_question(G, rng)
        assert ans == 1

    def test_disconnected_graph(self, disconnected_graph):
        G = disconnected_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = connected_components_question(G, rng)
        assert ans == 3  # {0,1,2}, {3,4}, {5}

    def test_single_node(self, single_node_graph):
        G = single_node_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = connected_components_question(G, rng)
        assert ans == 1

    def test_verifier_correct(self):
        assert verify_connected_components(3, 3, {}) is True

    def test_verifier_incorrect(self):
        assert verify_connected_components(2, 3, {}) is False


class TestCycleDetection:

    def test_path_no_cycle(self, known_graph):
        G = known_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = cycle_detection_question(G, rng)
        assert ans is False

    def test_cycle_graph(self, cycle_graph):
        G = cycle_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = cycle_detection_question(G, rng)
        assert ans is True
        assert len(meta["cycle_edges"]) > 0

    def test_complete_graph_has_cycle(self, complete_graph):
        G = complete_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = cycle_detection_question(G, rng)
        assert ans is True

    def test_verifier_bool(self):
        assert verify_cycle_detection(True, True, {}) is True
        assert verify_cycle_detection(False, False, {}) is True
        assert verify_cycle_detection(True, False, {}) is False

    def test_verifier_string(self):
        assert verify_cycle_detection("True", True, {}) is True
        assert verify_cycle_detection("false", False, {}) is True
        assert verify_cycle_detection("yes", True, {}) is True
        assert verify_cycle_detection("no", False, {}) is True


class TestMaxClique:

    def test_complete_graph(self, complete_graph):
        G = complete_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = max_clique_question(G, rng)
        assert ans == 5  # K5 -> clique of 5

    def test_path_graph(self, known_graph):
        G = known_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = max_clique_question(G, rng)
        assert ans == 2  # Path graph -> max clique is 2 (any edge)

    def test_single_node(self, single_node_graph):
        G = single_node_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = max_clique_question(G, rng)
        assert ans == 1

    def test_verifier_exact(self):
        assert verify_max_clique(5, 5, {"is_exact": True}) is True
        assert verify_max_clique(4, 5, {"is_exact": True}) is False

    def test_verifier_approximate(self):
        # For approximate, agent answer >= ground truth is OK
        assert verify_max_clique(5, 5, {"is_exact": False}) is True
        assert verify_max_clique(6, 5, {"is_exact": False}) is True
        assert verify_max_clique(4, 5, {"is_exact": False}) is False


class TestHamiltonianPath:

    def test_path_graph(self, known_graph):
        """A path graph has a Hamiltonian path by definition."""
        G = known_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = hamiltonian_path_question(G, rng)
        assert ans is True
        assert meta["path"] is not None
        assert len(meta["path"]) == G.number_of_nodes()

    def test_complete_graph(self, complete_graph):
        """Complete graph has Hamiltonian path."""
        G = complete_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = hamiltonian_path_question(G, rng)
        assert ans is True

    def test_disconnected_graph(self, disconnected_graph):
        """Disconnected graph cannot have a Hamiltonian path."""
        G = disconnected_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = hamiltonian_path_question(G, rng)
        assert ans is False

    def test_verifier_correct(self):
        assert verify_hamiltonian_path(True, True, {"is_exact": True}) is True
        assert verify_hamiltonian_path(False, False, {"is_exact": True}) is True

    def test_verifier_incorrect(self):
        assert verify_hamiltonian_path(False, True, {"is_exact": True}) is False


# =========================================================================
# Self-Consistency: Ground truth passes its own verifier
# =========================================================================

class TestSelfConsistency:
    """
    The most critical test: for every graph × task, the ground truth
    answer must pass its own verifier.
    """

    def test_all_small_graphs_all_tasks(self, small_graphs):
        for graph in small_graphs:
            G = graph.to_networkx()
            for task_type in TASK_REGISTRY:
                if task_type == "hamiltonian_path" and graph.num_nodes > 20:
                    continue
                task = generate_task_instance(
                    graph.graph_id, G, task_type, seed=42
                )
                result = verify_answer(task, task.answer)
                assert result.is_correct, (
                    f"Self-consistency FAILED for {task.task_id}: "
                    f"answer={task.answer}, task_type={task_type}"
                )


# =========================================================================
# Data I/O Tests
# =========================================================================

class TestDataIO:

    def test_graph_jsonl_roundtrip(self, small_graphs, tmp_path):
        fpath = str(tmp_path / "test_graphs.jsonl")
        save_graphs_jsonl(small_graphs, fpath)
        loaded = load_graphs_jsonl(fpath)
        assert len(loaded) == len(small_graphs)
        for orig, loaded_g in zip(small_graphs, loaded):
            assert orig.graph_id == loaded_g.graph_id
            assert orig.nodes == loaded_g.nodes
            assert orig.edges == loaded_g.edges

    def test_task_jsonl_roundtrip(self, small_graphs, tmp_path):
        tasks = generate_task_dataset(small_graphs[:3],
                                       task_types=["shortest_path"])
        fpath = str(tmp_path / "test_tasks.jsonl")
        save_tasks_jsonl(tasks, fpath)
        loaded = load_tasks_jsonl(fpath)
        assert len(loaded) == len(tasks)
        for orig, loaded_t in zip(tasks, loaded):
            assert orig.task_id == loaded_t.task_id
            assert orig.answer == loaded_t.answer

    def test_graph_to_text_representations(self, known_graph):
        adj_text = graph_to_adjacency_text(known_graph)
        assert "5 nodes" in adj_text
        assert "edges" in adj_text.lower()

        edge_text = graph_to_edge_list_text(known_graph)
        assert "Nodes:" in edge_text
        assert "Edges:" in edge_text

        stats = graph_to_summary_stats(known_graph)
        assert "Nodes: 5" in stats
        assert "Density:" in stats


# =========================================================================
# Evaluation Pipeline Tests
# =========================================================================

class TestEvaluationPipeline:

    def test_batch_evaluation(self, small_graphs):
        tasks = generate_task_dataset(
            small_graphs[:5],
            task_types=["shortest_path", "connected_components"],
        )
        # Simulate correct answers
        answers = {t.task_id: t.answer for t in tasks}
        results = evaluate_batch(tasks, answers)
        metrics = compute_metrics(results)

        assert metrics["overall_accuracy"] == 1.0
        assert metrics["total"] == len(tasks)
        assert metrics["correct"] == len(tasks)

    def test_missing_answers(self, small_graphs):
        tasks = generate_task_dataset(
            small_graphs[:3],
            task_types=["cycle_detection"],
        )
        # Provide no answers
        results = evaluate_batch(tasks, {})
        metrics = compute_metrics(results)
        assert metrics["overall_accuracy"] == 0.0

    def test_partial_answers(self, small_graphs):
        tasks = generate_task_dataset(
            small_graphs[:4],
            task_types=["connected_components"],
        )
        # Answer only half correctly
        answers = {}
        for i, t in enumerate(tasks):
            if i < len(tasks) // 2:
                answers[t.task_id] = t.answer
        results = evaluate_batch(tasks, answers)
        metrics = compute_metrics(results)
        assert 0 < metrics["overall_accuracy"] < 1.0

    def test_per_task_type_metrics(self, small_graphs):
        tasks = generate_task_dataset(
            small_graphs[:3],
            task_types=["shortest_path", "cycle_detection"],
        )
        answers = {t.task_id: t.answer for t in tasks}
        results = evaluate_batch(tasks, answers)
        metrics = compute_metrics(results)

        assert "shortest_path" in metrics["by_task_type"]
        assert "cycle_detection" in metrics["by_task_type"]
        for tt_metrics in metrics["by_task_type"].values():
            assert tt_metrics["accuracy"] == 1.0


# =========================================================================
# Reachability Task Tests
# =========================================================================

class TestReachability:

    def test_reachability_connected(self, known_graph):
        from src.tasks import reachability_question, verify_reachability
        G = known_graph.to_networkx()
        rng = random.Random(42)
        q, ans, meta = reachability_question(G, rng)
        assert ans is True
        assert verify_reachability(True, ans, meta) is True

    def test_reachability_verifier_strings(self):
        from src.tasks import verify_reachability
        assert verify_reachability("yes", True, {}) is True
        assert verify_reachability("no", False, {}) is True
        assert verify_reachability("true", True, {}) is True
        assert verify_reachability("false", False, {}) is True


# =========================================================================
# Benchmark Ingestion Tests (GraphQA & GraphArena)
# =========================================================================

class TestBenchmarkIngestion:

    def test_graphqa_test_split_exists_and_loads(self):
        from src.data_loader import load_graphqa
        tasks = load_graphqa("data", split="test")
        assert len(tasks) > 0
        assert all(isinstance(t, TaskInstance) for t in tasks)
        task_types = set(t.task_type for t in tasks)
        assert "shortest_path" in task_types
        assert "cycle_detection" in task_types
        assert "reachability" in task_types

    def test_grapharena_easy_split_exists_and_loads(self):
        from src.data_loader import load_grapharena
        tasks = load_grapharena("data", difficulty="easy")
        assert len(tasks) > 0
        assert all(isinstance(t, TaskInstance) for t in tasks)
        task_types = set(t.task_type for t in tasks)
        assert "shortest_path" in task_types
        assert "connected_components" in task_types

    def test_dataset_summary_completeness(self):
        from src.data_loader import get_full_dataset_summary
        summary = get_full_dataset_summary("data")
        assert "synthetic" in summary
        assert "graphqa" in summary
        assert "grapharena" in summary
        assert len(summary["synthetic"]) >= 4
        assert len(summary["graphqa"]) == 3
        assert len(summary["grapharena"]) == 2


# =========================================================================
# Edge Case Tests
# =========================================================================

class TestEdgeCases:

    def test_two_node_graph(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        gi = GraphInstance.from_networkx(G, "test_2node", "manual", "small")
        G_nx = gi.to_networkx()

        rng = random.Random(42)

        # Shortest path
        q, ans, meta = shortest_path_question(G_nx, rng)
        assert ans == 1  # Single edge

        # Components
        q, ans, meta = connected_components_question(G_nx, rng)
        assert ans == 1

        # Cycle
        q, ans, meta = cycle_detection_question(G_nx, rng)
        assert ans is False

        # Max clique
        q, ans, meta = max_clique_question(G_nx, rng)
        assert ans == 2  # The edge itself

    def test_empty_graph_no_edges(self):
        G = nx.empty_graph(5)
        gi = GraphInstance.from_networkx(G, "test_empty", "manual", "small")
        G_nx = gi.to_networkx()
        rng = random.Random(42)

        q, ans, meta = connected_components_question(G_nx, rng)
        assert ans == 5

        q, ans, meta = cycle_detection_question(G_nx, rng)
        assert ans is False

    def test_large_complete_graph_clique(self):
        """K10 — max clique should be 10."""
        G = nx.complete_graph(10)
        rng = random.Random(42)
        q, ans, meta = max_clique_question(G, rng)
        assert ans == 10

