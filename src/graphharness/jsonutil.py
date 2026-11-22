"""Relaxed JSON parsing for model output.

Models sometimes emit a JSON-looking object with Python syntax (for example a
triple-quoted code string).  ``json.loads`` rejects that, while
``ast.literal_eval`` accepts it safely.  Use strict JSON first, then fall back.
"""

from __future__ import annotations

import ast
import json
from typing import Any


def loads_relaxed(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise json.JSONDecodeError(str(exc), text, 0) from None
