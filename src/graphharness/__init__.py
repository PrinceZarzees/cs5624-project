"""Harness Optimization for Graph Reasoning (CS 5624) - data, tasks and scoring."""

__version__ = "0.2.0"

from .agent import BASELINE_A, BASELINE_B, BASELINE_B_V2, HARNESSES, Harness, get_harness, run_agent
from .data_loader import load_jsonl, load_split, save_jsonl, to_graph, to_graphs
from .evaluate_answer import score_response
from .models import CachingModel, ChatResult, OpenAIModel
from .tasks import TASKS, get_task

__all__ = [
    "BASELINE_A",
    "BASELINE_B",
    "BASELINE_B_V2",
    "HARNESSES",
    "Harness",
    "CachingModel",
    "ChatResult",
    "OpenAIModel",
    "TASKS",
    "get_harness",
    "get_task",
    "load_jsonl",
    "load_split",
    "run_agent",
    "save_jsonl",
    "score_response",
    "to_graph",
    "to_graphs",
]
