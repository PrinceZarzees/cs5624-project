import json

import pytest

from graphharness import build_data
from graphharness.ablation import run_ablation
from graphharness.agent import BASELINE_A, BASELINE_B
from graphharness.analysis import (
    accuracy_by_size,
    accuracy_by_task,
    format_size_table,
    size_bucket,
)
from graphharness.data_loader import load_jsonl
from graphharness.models import CallableModel
from graphharness.plot_results import plot
from graphharness.scaling import _dedupe_harnesses, generalize_gaps, run_scaling


def fake_scores():
    return [
        {"id": "a", "task": "shortest_path", "n_nodes": 10, "parsed": True, "feasible": True, "correct": True},
        {"id": "b", "task": "shortest_path", "n_nodes": 120, "parsed": True, "feasible": False, "correct": False},
        {"id": "c", "task": "max_clique", "n_nodes": 800, "parsed": False, "feasible": False, "correct": False},
    ]


def truth_text(record):
    a = record["answer"]
    nodes = {
        "shortest_path": a.get("path"),
        "connected_components": a.get("representatives"),
    }[record["task"]]
    return "<answer>[" + ", ".join(str(v) for v in nodes) + "]</answer>"


def build_tiny(tmp_path, splits=("test_small", "test_large")):
    cfg = {
        "seed": 19,
        "tasks": ["shortest_path", "connected_components"],
        "splits": {
            "test_small": {"n_range": [4, 8], "per_task": 2},
            "test_large": {"n_range": [100, 120], "per_task": 2},
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "data"
    build_data.main(["--config", str(cfg_path), "--out", str(out)])
    return out, {s: load_jsonl(out / f"{s}.jsonl") for s in splits}


def oracle(records):
    by_message = {BASELINE_A.user_message(r): truth_text(r) for r in records}

    def answer(messages):
        text = messages[-1]["content"]
        return by_message.get(text, "<answer>[]</answer>")

    return CallableModel(answer, name="oracle")


def test_size_buckets_and_tables():
    assert size_bucket(50) == "4-50"
    assert size_bucket(51) == "51-99"
    assert size_bucket(2000) == "1000+"
    scores = fake_scores()
    sizes = accuracy_by_size(scores)
    assert [r["size_bin"] for r in sizes] == ["4-50", "100-300", "500-1000"]
    assert sizes[0]["accuracy"] == 1.0
    assert accuracy_by_task(scores)[0]["task"] == "max_clique"
    assert "accuracy" in format_size_table(sizes)


def test_generalization_gap():
    runs = [
        {"harness": "h", "split": "test_small", "accuracy": 0.9},
        {"harness": "h", "split": "test_large", "accuracy": 0.4},
        {"harness": "other", "split": "test_small", "accuracy": 0.5},
    ]
    gaps = generalize_gaps(runs)
    assert gaps == [{
        "harness": "h", "small_split": "test_small", "large_split": "test_large",
        "small_accuracy": 0.9, "large_accuracy": 0.4, "delta_gen": pytest.approx(0.5),
    }]


def test_scaling_dedupes_harness_names():
    from graphharness.agent import BASELINE_B
    a = BASELINE_B.copy(name="same")
    b = BASELINE_B.copy(name="same", rules=[])
    kept, skipped = _dedupe_harnesses([a, b])
    assert len(kept) == 1
    assert skipped and "same" in skipped[0]


def test_run_scaling_writes_summary(tmp_path):
    data_dir, records = build_tiny(tmp_path)
    model = oracle(records["test_small"] + records["test_large"])
    out = tmp_path / "scaling"
    payload = run_scaling(
        ["baseline_a"], ["test_small", "test_large"], model, out,
        data_dir=data_dir, limit_per_task=1, quiet=True,
    )
    assert len(payload["runs"]) == 2
    assert payload["runs"][0]["accuracy"] == 1.0
    assert payload["generalization_gaps"][0]["delta_gen"] == 0.0
    assert (out / "scaling_summary.json").exists()
    assert (out / "scaling_summary.txt").exists()


def test_ablation_writes_rule_deltas(tmp_path):
    data_dir, records = build_tiny(tmp_path, splits=("test_small",))
    model = oracle(records["test_small"])
    out = tmp_path / "ablation"
    payload = run_ablation(
        BASELINE_B, model, "test_small", out, data_dir=data_dir,
        limit_per_task=1, rule_indices=[0, 1], quiet=True,
    )
    assert payload["full_accuracy"] == 1.0
    assert len(payload["rows"]) == 3
    assert all(row["delta"] == 0.0 for row in payload["rows"][1:])
    assert (out / "ablation_summary.txt").exists()


def test_plot_writes_files(tmp_path):
    pytest.importorskip("matplotlib")
    data_dir, records = build_tiny(tmp_path, splits=("test_small",))
    model = oracle(records["test_small"])
    out = tmp_path / "run"
    from graphharness.runner import run_experiment
    run_experiment(BASELINE_A, model, split="test_small", out_dir=out,
                   data_dir=data_dir, quiet=True)
    written = plot([("Baseline A", [str(out / "scores.jsonl")])], tmp_path / "plots")
    assert (tmp_path / "plots" / "accuracy_vs_size.png") in written
    assert (tmp_path / "plots" / "accuracy_by_task.png") in written
