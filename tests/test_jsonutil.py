import pytest

from graphharness.jsonutil import loads_relaxed


def test_loads_relaxed_accepts_json_and_python_literals():
    assert loads_relaxed('{"tool": "answer", "answer": [1, 2]}')["answer"] == [1, 2]
    value = loads_relaxed('{"tool": "run_python", "code": """print(1)"""}')
    assert value["code"] == "print(1)"


def test_loads_relaxed_rejects_garbage():
    with pytest.raises(Exception):
        loads_relaxed("not json at all")
