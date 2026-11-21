import pytest

from graphharness.models import (
    CachingModel,
    ChatResult,
    CallableModel,
    ScriptedModel,
    _parse_chat_response,
)


def test_scripted_model_returns_in_order():
    model = ScriptedModel(["a", "b"])
    assert model.chat([{"role": "user", "content": "x"}]).text == "a"
    assert model.chat([{"role": "user", "content": "x"}]).text == "b"
    with pytest.raises(RuntimeError):
        model.chat([{"role": "user", "content": "x"}])


def test_caching_model_hits_cache(tmp_path):
    calls = {"n": 0}

    def fn(messages):
        calls["n"] += 1
        return f"answer {calls['n']}"

    cache = CachingModel(CallableModel(fn), tmp_path / "cache.sqlite")
    messages = [{"role": "user", "content": "same prompt"}]
    first = cache.chat(messages)
    second = cache.chat(messages)
    assert first.text == second.text == "answer 1"
    assert calls["n"] == 1
    # Decoding settings are part of the key.
    third = cache.chat(messages, temperature=0.7)
    assert third.text == "answer 2"
    assert calls["n"] == 2


def test_parse_chat_response_usage():
    raw = {
        "model": "m",
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }
    result = _parse_chat_response(raw, "requested")
    assert result.text == "hello"
    assert result.total_tokens == 7
    assert result.model == "m"


def test_parse_chat_response_reasoning_fallback():
    raw = {"choices": [{"message": {"content": None, "reasoning_content": "thought"}}]}
    assert _parse_chat_response(raw, "m").text == "thought"


def test_chat_result_total_tokens():
    assert ChatResult("x", prompt_tokens=2, completion_tokens=5).total_tokens == 7
