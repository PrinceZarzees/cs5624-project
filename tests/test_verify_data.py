import json

from graphharness import build_data
from graphharness.verify_data import truth_text, verify_all, verify_split


def build_tiny(tmp_path):
    cfg = {
        "seed": 23,
        "tasks": ["shortest_path", "connected_components", "cycle_detection",
                  "diameter", "max_clique"],
        "splits": {"test": {"n_range": [4, 10], "per_task": 2}},
    }
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(cfg))
    out = tmp_path / "data"
    build_data.main(["--config", str(path), "--out", str(out)])
    return out


def test_truth_text_and_verify_split(tmp_path):
    data = build_tiny(tmp_path)
    result = verify_split("test", data)
    assert result["count"] == 10
    assert result["errors"] == []
    assert result["min_nodes"] >= 4
    assert result["max_nodes"] <= 10


def test_verify_all_checks_manifest_and_detects_corruption(tmp_path):
    data = build_tiny(tmp_path)
    ok = verify_all(data)
    assert ok["ok"] is True
    assert ok["manifest_errors"] == []

    # Corrupt one record's answer and confirm verification fails.
    path = data / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["answer"] = {"length": 999, "path": []}
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    bad = verify_all(data)
    assert bad["ok"] is False
    assert bad["splits"][0]["errors"]
    assert any("sha256" in error for error in bad["manifest_errors"])


def test_truth_text_unknown_task():
    try:
        truth_text({"task": "nope", "answer": {}})
    except KeyError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected KeyError")
