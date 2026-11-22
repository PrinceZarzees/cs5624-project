"""Shared command-line helpers for experiment, optimizer and ablation runners."""

from __future__ import annotations

import argparse
import os
import sys

from .agent import Harness, get_harness
from .models import CachingModel, Model, OpenAIModel, default_cache_path


def add_model_arguments(parser: argparse.ArgumentParser, default_model: str | None = None) -> None:
    group = parser.add_argument_group("model")
    group.add_argument(
        "--model",
        default=default_model or os.environ.get("GRAPH_HARNESS_MODEL"),
        help="model name for an OpenAI-compatible endpoint (or GRAPH_HARNESS_MODEL)",
    )
    group.add_argument(
        "--base-url",
        default=os.environ.get("GRAPH_HARNESS_BASE_URL"),
        help="e.g. https://api.openai.com/v1 or http://localhost:8000/v1",
    )
    group.add_argument(
        "--api-key-env",
        default="GRAPH_HARNESS_API_KEY",
        help="environment variable holding the API key (default: GRAPH_HARNESS_API_KEY)",
    )
    group.add_argument("--temperature", type=float, default=0.0)
    group.add_argument("--max-tokens", type=int, default=1024)
    group.add_argument("--request-timeout", type=float, default=180.0)
    group.add_argument("--retries", type=int, default=3)
    group.add_argument(
        "--cache",
        nargs="?",
        const=str(default_cache_path()),
        default=None,
        help="SQLite response cache path (default off; use --cache with no path for the default)",
    )


def build_model(args: argparse.Namespace, model_name: str | None = None,
                required: bool = True) -> Model | None:
    name = model_name or getattr(args, "model", None)
    if not name:
        if required:
            print(
                "error: no model given. Pass --model, set GRAPH_HARNESS_MODEL, "
                "or use a local endpoint with --base-url.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return None
    api_key = os.environ.get(getattr(args, "api_key_env", "GRAPH_HARNESS_API_KEY"))
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        base = getattr(args, "base_url", None) or os.environ.get("GRAPH_HARNESS_BASE_URL") or ""
        if "localhost" in base or "127.0.0.1" in base:
            api_key = "EMPTY"
        else:
            print(
                "error: no API key. Set GRAPH_HARNESS_API_KEY (or OPENAI_API_KEY) "
                "or pass --api-key-env with the name of the variable.",
                file=sys.stderr,
            )
            raise SystemExit(2)
    model: Model = OpenAIModel(
        model=name,
        base_url=getattr(args, "base_url", None),
        api_key=api_key,
        temperature=getattr(args, "temperature", 0.0),
        max_tokens=getattr(args, "max_tokens", 1024),
        timeout_s=getattr(args, "request_timeout", 180.0),
        retries=getattr(args, "retries", 3),
    )
    if getattr(args, "cache", None):
        model = CachingModel(model, args.cache)
    return model


def harness_from_arg(value: str) -> Harness:
    try:
        return get_harness(value)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
