import json

from graphharness import build_data
from graphharness.agent import BASELINE_A
from graphharness.data_loader import load_jsonl
from graphharness.inspect_runs import format_report, summarize_run, trajectory_stats
from graphharness.models import CallableModel
from graphharness.runner import run_experiment


def build_tiny(tmp_path):
    cfg = {
        "seed": 31,
        "tasks": ["shortest_path", "connected_components"],
        "splits": {"test": {"n_range": [4, 8], "per_task": 1}},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    out = tmp_path / "data"
    build_data.main(["--config", str(path), "--out", str(out)])
    return out


def answer_for(records):
    by_message = {}
    for r in records:
        a = r["answer"]
        nodes = a.get("path") if r["task"] == "shortest_path" else a.get("representatives")
        by_message[BASELINE_A.user_message(r)] = (
            "<answer>[" + ", ".join(str(v) for v in nodes) + "]</answer>"
        )

    def fn(messages):
        return by_message[messages[-1]["content"]]

    return CallableModel(fn, name="oracle")


def test_inspect_run_summary(tmp_path):
    data = build_tiny(tmp_path)
    records = load_jsonl(data / "test.jsonl")
    run = tmp_path / "run"
    run_experiment(BASELINE_A, answer_for(records), split="test", out_dir=run,
                   data_dir=data, quiet=True)
    payload = summarize_run(run)
    assert payload["overall"]["n"] == 2
    assert payload["overall"]["accuracy"] == 1.0
    assert payload["failures"] == []
    assert set(payload["overall"]["tasks"]) == {"shortest_path", "connected_components"}
    assert "accuracy" in format_report(payload)


def test_trajectory_stats_empty():
    stats = trajectory_stats(None)
    assert stats["tool_calls"] == 0
    assert stats["steps"] == 0
