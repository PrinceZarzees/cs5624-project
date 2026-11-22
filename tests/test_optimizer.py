import json
import random

from graphharness import build_data
from graphharness.agent import BASELINE_A, BASELINE_B
from graphharness.data_loader import load_jsonl
from graphharness.models import CallableModel
from graphharness.optimizer import (
    OptimizerConfig,
    extract_proposal,
    optimize,
    sample_records,
    validate_proposal,
)


def truth_text(record):
    a = record["answer"]
    nodes = {
        "shortest_path": a.get("path"),
        "connected_components": a.get("representatives"),
    }[record["task"]]
    return "<answer>[" + ", ".join(str(v) for v in nodes) + "]</answer>"


def build_train_val(tmp_path, per_task=2):
    cfg = {
        "seed": 11,
        "tasks": ["shortest_path", "connected_components"],
        "splits": {
            "train": {"n_range": [4, 8], "per_task": per_task},
            "val": {"n_range": [4, 8], "per_task": per_task},
        },
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "data"
    build_data.main(["--config", str(cfg_path), "--out", str(out)])
    return load_jsonl(out / "train.jsonl"), load_jsonl(out / "val.jsonl")


def test_extract_proposal_from_fence_and_prose():
    text = 'thinking...\n```json\n{"rules": ["a"], "system_prompt": "x"}\n```'
    assert extract_proposal(text)["rules"] == ["a"]
    text = 'prefix {"rules": ["b"], "system_prompt": "y", "reason": "r"} suffix'
    assert extract_proposal(text)["rules"] == ["b"]
    assert extract_proposal("no json here") is None


def test_validate_proposal_rejects_bad_edits():
    ok, reason = validate_proposal(
        {"system_prompt": "be careful", "rules": ["use nx"]}, BASELINE_B,
        OptimizerConfig(),
    )
    assert ok is not None and ok.rules == ["use nx"]

    bad, reason = validate_proposal(
        {"system_prompt": "x", "rules": ["r"] * 17}, BASELINE_B, OptimizerConfig()
    )
    assert bad is None and "too many" in reason

    bad, reason = validate_proposal(
        {"system_prompt": "x", "rules": ["r"], "graph_view": "magic"},
        BASELINE_B, OptimizerConfig(),
    )
    assert bad is None and "graph_view" in reason


def test_sample_records_is_per_task_and_deterministic():
    records = [
        {"id": f"{task}/{i}", "task": task}
        for task in ("a", "b") for i in range(10)
    ]
    a = sample_records(records, 3, random.Random(1))
    b = sample_records(records, 3, random.Random(1))
    assert a == b
    assert len(a) == 6
    assert [r["task"] for r in a].count("a") == 3


def test_optimizer_accepts_a_validated_improvement(tmp_path):
    train, val = build_train_val(tmp_path)
    by_message = {BASELINE_A.user_message(r): truth_text(r) for r in train + val}
    wrong = "<answer>[99999]</answer>"

    def actor(messages):
        system = messages[0]["content"]
        if "EXPERT MODE" not in system:
            return wrong
        return by_message.get(messages[-1]["content"], wrong)

    def critic(messages):
        return json.dumps({
            "system_prompt": "EXPERT MODE: solve graph problems carefully.",
            "rules": ["Use NetworkX to compute the answer, then verify it."],
            "reason": "The baseline never used the graph tool.",
        })

    out = tmp_path / "opt"
    result = optimize(
        BASELINE_A,
        CallableModel(actor, name="actor"),
        CallableModel(critic, name="critic"),
        train, val, out,
        OptimizerConfig(rounds=2, train_per_task=2, val_per_task=2, seed=5),
    )
    assert result.initial_val_accuracy == 0.0
    assert result.best_val_accuracy == 1.0
    assert "EXPERT MODE" in result.best_harness.system_prompt
    assert (out / "best_harness.json").exists()
    assert (out / "history.jsonl").exists()
    assert (out / "optimization_summary.json").exists()
    assert (out / "checkpoints" / "round_01.json").exists()
    rows = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    assert rows[0]["role"] == "initial"
    assert rows[1]["accepted"] is True


def test_optimizer_rrs_accepts_single_rule_edit(tmp_path):
    from graphharness.optimizer import optimize_rrs, OptimizerConfig
    train, val = build_train_val(tmp_path)
    by_message = {BASELINE_A.user_message(r): truth_text(r) for r in train + val}
    wrong = "<answer>[99999]</answer>"

    def actor(messages):
        if "EXPERT RULE" not in messages[0]["content"]:
            return wrong
        return by_message.get(messages[-1]["content"], wrong)

    def critic(messages):
        return json.dumps({
            "system_prompt": BASELINE_A.system_prompt,
            "rules": ["EXPERT RULE: use NetworkX and verify the result."],
            "reason": "add a reusable verification rule",
        })

    out = tmp_path / "rrs"
    result = optimize_rrs(
        BASELINE_A, CallableModel(actor, name="actor"), CallableModel(critic, name="critic"),
        train, val, out,
        OptimizerConfig(rounds=1, train_per_task=2, val_per_task=1, seed=3, rrs=True),
    )
    assert result.best_val_accuracy == 1.0
    assert "EXPERT RULE" in result.best_harness.rules[0]
    assert (out / "best_harness.json").exists()


def test_gepa_accepts_a_prompt_improvement(tmp_path):
    from graphharness.gepa import optimize_gepa, GEPAConfig
    train, val = build_train_val(tmp_path)
    by_message = {BASELINE_A.user_message(r): truth_text(r) for r in train + val}
    wrong = "<answer>[99999]</answer>"

    def actor(messages):
        if "EXPERT MODE" not in messages[0]["content"]:
            return wrong
        return by_message.get(messages[-1]["content"], wrong)

    def critic(messages):
        return json.dumps({
            "system_prompt": "EXPERT MODE: verify every graph answer.",
            "rules": [],
            "reason": "ask for verification",
        })

    summary = optimize_gepa(
        BASELINE_A, CallableModel(actor), CallableModel(critic), train, val,
        tmp_path / "gepa", GEPAConfig(rounds=1, candidates_per_round=1,
                                     train_per_task=2, val_per_task=2, seed=7),
    )
    assert summary["best_val_accuracy"] == 1.0
    assert "EXPERT MODE" in summary["best_harness"]["system_prompt"]
    assert (tmp_path / "gepa" / "best_harness.json").exists()


def test_meta_harness_accepts_helper_preamble(tmp_path):
    from graphharness.meta_harness import MetaConfig, optimize_meta
    train, val = build_train_val(tmp_path)
    by_message = {BASELINE_A.user_message(r): truth_text(r) for r in train + val}
    wrong = "<answer>[99999]</answer>"

    def actor(messages):
        if "HELPER" not in messages[0]["content"]:
            return wrong
        return by_message.get(messages[-1]["content"], wrong)

    def critic(messages):
        return json.dumps({
            "system_prompt": BASELINE_A.system_prompt,
            "rules": [],
            "graph_view": "summary",
            "tool_backend": "python",
            "tool_preamble": "# HELPER\nhelper = 1",
            "reason": "inject a reusable helper",
        })

    summary = optimize_meta(
        BASELINE_A, CallableModel(actor), CallableModel(critic), train, val,
        tmp_path / "meta", MetaConfig(rounds=1, train_per_task=2, val_per_task=2, seed=9),
    )
    assert summary["best_val_accuracy"] == 1.0
    assert "HELPER" in summary["best_harness"]["tool_preamble"]
    assert (tmp_path / "meta" / "best_harness.json").exists()


def test_evoskill_accepts_a_skill(tmp_path):
    from graphharness.evoskill import EvoSkillConfig, optimize_evoskill
    train, val = build_train_val(tmp_path)
    by_message = {BASELINE_A.user_message(r): truth_text(r) for r in train + val}
    wrong = "<answer>[99999]</answer>"

    def actor(messages):
        if "EXPERT SKILL" not in messages[0]["content"]:
            return wrong
        return by_message.get(messages[-1]["content"], wrong)

    def critic(messages):
        return json.dumps({"skill": {"name": "verify", "trigger": "EXPERT SKILL",
                                     "instruction": "verify the result"}, "reason": "test"})

    summary = optimize_evoskill(
        BASELINE_A, CallableModel(actor), CallableModel(critic), train, val,
        tmp_path / "evo", EvoSkillConfig(rounds=1, train_per_task=2, val_per_task=2, seed=4),
    )
    assert summary["best_val_accuracy"] == 1.0
    assert (tmp_path / "evo" / "best_harness.json").exists()
