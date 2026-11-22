import importlib
import json
from pathlib import Path


def test_empty_override_is_identity(tmp_path, monkeypatch):
    import hda.prompts as p
    monkeypatch.setenv("HDA_PROMPT_FILE", str(tmp_path / "empty.json"))
    (tmp_path / "empty.json").write_text(json.dumps({"strategy": "", "hidden_procedure": "", "iteration_note": ""}))
    p = importlib.reload(p)
    assert p.STRATEGY_NOTE == p.HIDDEN_NOTE == p.ITERATION_NOTE == ""


def test_override_applied(tmp_path, monkeypatch):
    import hda.prompts as p
    f = tmp_path / "c.json"
    f.write_text(json.dumps({"strategy": "Be concise.", "hidden_procedure": "Infer first.",
                             "iteration_note": "Test before submit."}))
    monkeypatch.setenv("HDA_PROMPT_FILE", str(f))
    p = importlib.reload(p)
    assert p.STRATEGY_NOTE == "Be concise."
    assert p.HIDDEN_NOTE == "Infer first."
    assert p.ITERATION_NOTE == "Test before submit."
