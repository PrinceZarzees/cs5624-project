import json

from graphharness.agent import (
    BASELINE_A,
    BASELINE_B,
    BASELINE_B_V2,
    Harness,
    graph_summary,
    parse_action,
    run_agent,
)
from graphharness.evaluate_answer import score_response
from graphharness.models import ScriptedModel


def record():
    return {
        "id": "shortest_path/test/00000",
        "task": "shortest_path",
        "split": "test",
        "graph": {"directed": False, "nodes": [1, 2, 3, 4], "edges": [[1, 2], [2, 3], [3, 4]]},
        "params": {"source": 1, "target": 4},
        "question": "Find a shortest path from node 1 to node 4.",
        "answer": {"length": 3, "path": [1, 2, 3, 4]},
        "meta": {"n_nodes": 4},
    }


def test_parse_answer_tag():
    action = parse_action("reasoning...\n<answer>[1, 2, 3]</answer>")
    assert action.kind == "answer"
    assert "1, 2, 3" in action.answer


def test_parse_json_tool_call():
    action = parse_action('{"tool": "run_python", "code": "print(1)"}')
    assert action.kind == "code"
    assert action.code == "print(1)"


def test_parse_json_answer():
    action = parse_action('{"tool": "answer", "answer": [4, 5]}')
    assert action.kind == "answer"
    assert action.answer == "<answer>[4, 5]</answer>"


def test_python_literal_tool_call_is_accepted():
    text = '{"tool": "run_python", "code": """\\nprint(G.number_of_nodes())\\n"""}'
    action = parse_action(text)
    assert action.kind == "code"
    assert "print(G.number_of_nodes())" in action.code


def test_code_is_run_before_a_guessed_answer():
    text = '```python\nprint(1)\n```\n<answer>[9]</answer>'
    action = parse_action(text)
    assert action.kind == "code"


def test_last_action_wins_without_code():
    text = 'thinking... <answer>[9]</answer>'
    assert parse_action(text).kind == "answer"


def test_plain_bracketed_answer_fallback():
    assert parse_action("The path is [1, 2, 3, 4].").kind == "answer"


def test_agent_answers_without_tools():
    model = ScriptedModel(["<answer>[1, 2, 3, 4]</answer>"])
    result = run_agent(BASELINE_A, model, record())
    assert result.stopped_reason == "answer"
    assert result.tool_calls == 0
    assert score_response(record(), result.response)["correct"]


def test_agent_runs_code_then_answers():
    code = "import networkx as nx\nprint(nx.shortest_path(G, params['source'], params['target']))"
    model = ScriptedModel([
        '{"tool": "run_python", "code": ' + repr(code) + '}',
        "<answer>[1, 2, 3, 4]</answer>",
    ])
    result = run_agent(BASELINE_B, model, record())
    assert result.stopped_reason == "answer"
    assert result.tool_calls == 1
    assert "[1, 2, 3, 4]" in result.steps[0]["stdout"]
    assert score_response(record(), result.response)["correct"]


def test_agent_runs_code_before_accepting_same_message_answer():
    model = ScriptedModel([
        "```python\nprint(G.number_of_nodes())\n```\n<answer>[9]</answer>",
        "<answer>[1, 2, 3, 4]</answer>",
    ])
    result = run_agent(BASELINE_A, model, record())
    assert result.tool_calls == 1
    assert result.steps[0]["stdout"].strip() == "4"
    assert score_response(record(), result.response)["correct"]


def test_code_failure_is_sent_back_to_model():
    model = ScriptedModel([
        "```python\nraise RuntimeError('boom')\n```",
        "<answer>[1, 2, 3, 4]</answer>",
    ])
    result = run_agent(BASELINE_A, model, record())
    assert result.steps[0]["ok"] is False
    assert "boom" in result.messages[3]["content"]
    assert score_response(record(), result.response)["correct"]


def test_step_limit_stops_loop():
    model = ScriptedModel(["still thinking"] * 6)
    result = run_agent(BASELINE_A.copy(max_steps=3), model, record())
    assert result.stopped_reason == "max_steps"
    assert len(result.steps) == 4  # three tool turns + one final-answer turn


def test_final_answer_turn_after_max_steps():
    model = ScriptedModel([
        "still thinking", "still thinking", "<answer>[1, 2, 3, 4]</answer>",
    ])
    result = run_agent(BASELINE_A.copy(max_steps=2), model, record())
    assert result.stopped_reason == "final_answer"
    assert len(result.steps) == 3
    assert score_response(record(), result.response)["correct"]


def test_final_turn_accepts_answer_even_with_code():
    model = ScriptedModel([
        "still thinking",
        "```python\nprint(1)\n```\n<answer>[1, 2, 3, 4]</answer>",
    ])
    result = run_agent(BASELINE_A.copy(max_steps=1), model, record())
    assert result.stopped_reason == "final_answer"
    assert score_response(record(), result.response)["correct"]


def test_prompt_does_not_include_ground_truth():
    msg = BASELINE_B.user_message(record())
    assert json.dumps(record()["answer"]) not in msg
    assert "length" not in msg
    # graph summary is structural, not the stored answer object
    assert "path" not in graph_summary(record())


def test_baseline_b_v2_documents_diameter_witness():
    assert any("diameter" in rule.lower() and "pair" in rule.lower()
               for rule in BASELINE_B_V2.rules)


def test_harness_round_trip_and_ablation(tmp_path):
    h = BASELINE_B.copy(name="candidate")
    path = tmp_path / "h.json"
    h.save(path)
    loaded = Harness.load(path)
    assert loaded == h
    assert len(loaded.without_rule(0).rules) == len(h.rules) - 1
    assert "Rules:" in loaded.system_message()
