import json
import random

from graphharness import build_data
from graphharness.agent import BASELINE_A
from graphharness.data_loader import load_jsonl
from graphharness.models import CallableModel
from graphharness.runner import evaluate_records, run_experiment


def truth_text(record):
    a = record["answer"]
    nodes = {
        "shortest_path": a.get("path"), "diameter": a.get("path"),
        "connected_components": a.get("representatives"),
        "cycle_detection": a.get("cycle"), "max_clique": a.get("clique"),
        "reachability": a.get("path"),
    }[record["task"]]
    return "<answer>[" + ", ".join(str(v) for v in nodes) + "]</answer>"


def build_tiny(tmp_path, per_task=2):
    cfg = {
        "seed": 3,
        "tasks": ["shortest_path", "connected_components"],
        "splits": {"test": {"n_range": [4, 8], "per_task": per_task}},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "data"
    build_data.main(["--config", str(cfg_path), "--out", str(out)])
    return out


def oracle_for(records):
    by_message = {BASELINE_A.user_message(r): truth_text(r) for r in records}

    def answer(messages):
        text = messages[-1]["content"]
        if text in by_message:
            return by_message[text]
        raise AssertionError("prompt did not match a known problem")

    return CallableModel(answer, name="oracle")


def test_evaluate_records_scores_correct_answers(tmp_path):
    rng = random.Random(0)
    records = [build_data.make_record("shortest_path", "test", 0, 6, rng)]
    results = evaluate_records(BASELINE_A, oracle_for(records), records)
    assert len(results) == 1
    assert results[0].score["correct"]
    assert results[0].agent is not None
    assert results[0].agent.total_tokens == 0


def test_run_experiment_writes_artifacts_and_resumes(tmp_path):
    data_dir = build_tiny(tmp_path)
    records = load_jsonl(data_dir / "test.jsonl")
    model = oracle_for(records)
    out = tmp_path / "run"
    summary = run_experiment(
        BASELINE_A, model, split="test", out_dir=out, data_dir=data_dir,
        quiet=True,
    )
    assert summary["n"] == len(records)
    assert summary["overall"]["accuracy"] == 1.0
    assert summary["errors"] == 0
    assert len(list((out / "trajectories").glob("*.json"))) == len(records)
    assert len(load_jsonl(out / "predictions.jsonl")) == len(records)
    assert len(load_jsonl(out / "scores.jsonl")) == len(records)
    assert (out / "summary.txt").exists()

    calls_before = len(model.calls)
    run_experiment(BASELINE_A, model, split="test", out_dir=out, data_dir=data_dir,
                   quiet=True, resume=True)
    assert len(model.calls) == calls_before, "resume should not call the model again"


def test_no_resume_reruns(tmp_path):
    data_dir = build_tiny(tmp_path, per_task=1)
    records = load_jsonl(data_dir / "test.jsonl")
    model = oracle_for(records)
    out = tmp_path / "run"
    run_experiment(BASELINE_A, model, split="test", out_dir=out, data_dir=data_dir, quiet=True)
    calls_before = len(model.calls)
    run_experiment(BASELINE_A, model, split="test", out_dir=out, data_dir=data_dir,
                   quiet=True, resume=False)
    assert len(model.calls) == calls_before + len(records)


def test_model_errors_become_wrong_answers(tmp_path):
    data_dir = build_tiny(tmp_path, per_task=1)
    records = load_jsonl(data_dir / "test.jsonl")[:1]

    def broken(messages):
        raise RuntimeError("endpoint down")

    model = CallableModel(broken)
    results = evaluate_records(BASELINE_A, model, records)
    assert results[0].score["correct"] is False
    assert "endpoint down" in results[0].error


def test_ids_filter_runs_only_selected_problems(tmp_path):
    data_dir = build_tiny(tmp_path, per_task=2)
    records = load_jsonl(data_dir / "test.jsonl")
    chosen = records[0]["id"]
    out = tmp_path / "run"
    summary = run_experiment(
        BASELINE_A, oracle_for(records), split="test", out_dir=out,
        data_dir=data_dir, ids=[chosen], quiet=True,
    )
    assert summary["n"] == 1
    assert load_jsonl(out / "predictions.jsonl")[0]["id"] == chosen


def test_reusing_output_dir_with_another_harness_is_rejected(tmp_path):
    from graphharness.agent import BASELINE_B
    import pytest

    data_dir = build_tiny(tmp_path, per_task=1)
    records = load_jsonl(data_dir / "test.jsonl")
    out = tmp_path / "run"
    run_experiment(BASELINE_A, oracle_for(records), split="test", out_dir=out,
                   data_dir=data_dir, quiet=True)
    with pytest.raises(SystemExit, match="different harness_hash"):
        run_experiment(BASELINE_B, oracle_for(records), split="test", out_dir=out,
                       data_dir=data_dir, quiet=True)