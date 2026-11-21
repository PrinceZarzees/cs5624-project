"""Model clients for harness experiments.

Only a chat-style interface is required::

    result = model.chat([{"role": "user", "content": "..."}])
    result.text

``OpenAIModel`` talks to any OpenAI-compatible ``/chat/completions`` endpoint
(OpenAI, vLLM, SGLang, DeepSeek, Together, ...).  ``CachingModel`` wraps another
model and stores responses in SQLite, which is important because harness
optimization and scaling runs replay many identical prompts.

The small ``ScriptedModel``/``CallableModel`` clients are for tests and for
offline pipeline checks; they never call the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    raw: dict | None = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@runtime_checkable
class Model(Protocol):
    name: str

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResult:
        ...


class OpenAIModel:
    """Minimal OpenAI-compatible chat client using only the standard library."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_s: float = 180.0,
        retries: int = 3,
        extra_body: dict | None = None,
    ):
        if not model:
            raise ValueError("model name is required")
        self.name = model
        self.base_url = (base_url or os.environ.get("GRAPH_HARNESS_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = (api_key or os.environ.get("GRAPH_HARNESS_API_KEY")
                        or os.environ.get("OPENAI_API_KEY") or "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.retries = retries
        self.extra_body = extra_body or {}

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResult:
        body = {
            "model": self.name,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            **self.extra_body,
            **kwargs,
        }
        payload = json.dumps(body).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None

        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                return _parse_chat_response(raw, self.name)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                last_error = RuntimeError(f"HTTP {exc.code} from {url}: {detail}")
                if exc.code not in (408, 409, 429) and exc.code < 500:
                    raise last_error from None
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else 2.0 ** attempt
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = RuntimeError(f"request to {url} failed: {exc}")
                delay = 2.0 ** attempt
            if attempt < self.retries:
                time.sleep(delay)
        raise last_error or RuntimeError("model request failed")


def _parse_chat_response(raw: dict, requested_model: str) -> ChatResult:
    choices = raw.get("choices") or []
    if not choices:
        raise RuntimeError(f"model response has no choices: {json.dumps(raw)[:500]}")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if text is None:
        # Some reasoning endpoints return only this field when content is empty.
        text = message.get("reasoning_content") or ""
    usage = raw.get("usage") or {}
    return ChatResult(
        text=str(text),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        model=str(raw.get("model") or requested_model),
        raw=raw,
    )


class ScriptedModel:
    """Return a fixed list of responses, one per call. Useful in tests."""

    def __init__(self, responses: list[str], name: str = "scripted"):
        self.responses = list(responses)
        self.name = name
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise RuntimeError("ScriptedModel ran out of responses")
        text = self.responses.pop(0)
        return ChatResult(text=text, model=self.name)


class CallableModel:
    """Adapt a ``messages -> str`` function to the model interface."""

    def __init__(self, fn: Callable[[list[dict]], str], name: str = "callable"):
        self.fn = fn
        self.name = name
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        return ChatResult(text=self.fn(messages), model=self.name)


class CachingModel:
    """SQLite response cache keyed by model, messages and decoding settings."""

    def __init__(self, inner: Model, path: str | Path):
        self.inner = inner
        self.name = inner.name
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                "key TEXT PRIMARY KEY, text TEXT NOT NULL, model TEXT NOT NULL,"
                "prompt_tokens INTEGER NOT NULL, completion_tokens INTEGER NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _key(self, messages: list[dict], kwargs: dict) -> str:
        blob = json.dumps(
            {"model": self.name, "messages": messages, "kwargs": kwargs},
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResult:
        key = self._key(messages, kwargs)
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT text, model, prompt_tokens, completion_tokens "
                "FROM responses WHERE key = ?", (key,),
            ).fetchone()
        if row:
            return ChatResult(
                text=row[0], model=row[1],
                prompt_tokens=row[2], completion_tokens=row[3],
            )
        result = self.inner.chat(messages, **kwargs)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO responses "
                "(key, text, model, prompt_tokens, completion_tokens) VALUES (?, ?, ?, ?, ?)",
                (key, result.text, result.model, result.prompt_tokens, result.completion_tokens),
            )
        return result


def default_cache_path() -> Path:
    return Path(os.environ.get("GRAPH_HARNESS_CACHE", "experiments/cache/responses.sqlite"))
