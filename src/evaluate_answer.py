"""
evaluate_answer.py
==================
Answer evaluation and verification module for the CS 5624 project.

Provides:
  - Single-answer verification against ground truth.
  - Batch evaluation of agent answers across task splits.
  - Accuracy computation and result logging.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .tasks import TASK_REGISTRY, TaskInstance
from .data_loader import load_tasks_jsonl


# ---------------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluating a single answer."""
    task_id: str
    task_type: str
    graph_id: str
    ground_truth: Any
    agent_answer: Any
    is_correct: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core verification
# ---------------------------------------------------------------------------

def verify_answer(
    task: TaskInstance,
    agent_answer: Any,
) -> EvalResult:
    """
    Verify an agent's answer against the ground truth for a task instance.

    Parameters
    ----------
    task : TaskInstance
        The task with ground truth.
    agent_answer : Any
        The agent's answer to verify.

    Returns
    -------
    EvalResult
    """
    task_type = task.task_type

    if task_type not in TASK_REGISTRY:
        return EvalResult(
            task_id=task.task_id,
            task_type=task_type,
            graph_id=task.graph_id,
            ground_truth=task.answer,
            agent_answer=agent_answer,
            is_correct=False,
            error_message=f"Unknown task type: {task_type}",
        )

    verify_fn = TASK_REGISTRY[task_type]["verify_fn"]

    try:
        is_correct = verify_fn(agent_answer, task.answer, task.metadata)
    except Exception as e:
        return EvalResult(
            task_id=task.task_id,
            task_type=task_type,
            graph_id=task.graph_id,
            ground_truth=task.answer,
            agent_answer=agent_answer,
            is_correct=False,
            error_message=f"Verification error: {str(e)}",
        )

    return EvalResult(
        task_id=task.task_id,
        task_type=task_type,
        graph_id=task.graph_id,
        ground_truth=task.answer,
        agent_answer=agent_answer,
        is_correct=is_correct,
    )


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    tasks: List[TaskInstance],
    answers: Dict[str, Any],
) -> List[EvalResult]:
    """
    Evaluate a batch of answers.

    Parameters
    ----------
    tasks : list of TaskInstance
        Task instances with ground truth.
    answers : dict
        Mapping from task_id -> agent answer.

    Returns
    -------
    list of EvalResult
    """
    results = []
    for task in tasks:
        agent_answer = answers.get(task.task_id)
        if agent_answer is None:
            results.append(EvalResult(
                task_id=task.task_id,
                task_type=task.task_type,
                graph_id=task.graph_id,
                ground_truth=task.answer,
                agent_answer=None,
                is_correct=False,
                error_message="No answer provided",
            ))
        else:
            results.append(verify_answer(task, agent_answer))
    return results


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(results: List[EvalResult]) -> Dict[str, Any]:
    """
    Compute accuracy metrics from evaluation results.

    Returns
    -------
    dict with keys:
        - "overall_accuracy": float
        - "by_task_type": dict[str, dict] with accuracy per task type
        - "total": int
        - "correct": int
        - "errors": int
    """
    if not results:
        return {
            "overall_accuracy": 0.0,
            "by_task_type": {},
            "total": 0,
            "correct": 0,
            "errors": 0,
        }

    total = len(results)
    correct = sum(1 for r in results if r.is_correct)
    errors = sum(1 for r in results if r.error_message is not None)

    # Per-task breakdown
    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        by_type[r.task_type]["total"] += 1
        if r.is_correct:
            by_type[r.task_type]["correct"] += 1

    by_type_metrics = {}
    for task_type, counts in by_type.items():
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        by_type_metrics[task_type] = {
            "accuracy": round(acc, 4),
            "correct": counts["correct"],
            "total": counts["total"],
        }

    return {
        "overall_accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "by_task_type": by_type_metrics,
        "total": total,
        "correct": correct,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Save / Load results
# ---------------------------------------------------------------------------

def save_results(results: List[EvalResult], filepath: str) -> None:
    """Save evaluation results to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")


def save_metrics(metrics: Dict[str, Any], filepath: str) -> None:
    """Save computed metrics to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------------
# Self-test: verify ground truth solvers are self-consistent
# ---------------------------------------------------------------------------

def self_test_verifiers(n_graphs: int = 10, seed: int = 42) -> bool:
    """
    Quick self-test: generate small graphs, compute ground truth,
    and verify that the ground truth answers pass their own verifiers.

    Returns True if all tests pass.
    """
    import random
    from .graph_generator import generate_graphs
    from .tasks import generate_task_instance

    print("Running verifier self-test...")
    graphs = generate_graphs("small", n_graphs, seed=seed)
    all_passed = True

    for graph in graphs:
        G = graph.to_networkx()
        for task_type in TASK_REGISTRY:
            if task_type == "hamiltonian_path" and graph.num_nodes > 20:
                continue
            task = generate_task_instance(
                graph.graph_id, G, task_type, seed=seed
            )
            result = verify_answer(task, task.answer)
            if not result.is_correct:
                print(f"  FAIL: {task.task_id} — expected {task.answer}, "
                      f"verification returned False")
                all_passed = False

    if all_passed:
        print("  All verifier self-tests PASSED ✓")
    else:
        print("  Some verifier self-tests FAILED ✗")

    return all_passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    """Run self-test and optionally evaluate a results file."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate answers or run verifier self-tests."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="Run verifier self-consistency test.")
    parser.add_argument("--tasks-file", type=str,
                        help="Path to tasks JSONL file.")
    parser.add_argument("--answers-file", type=str,
                        help="Path to answers JSON file (task_id -> answer).")
    parser.add_argument("--output-dir", type=str, default="experiments/results",
                        help="Output directory for results.")
    args = parser.parse_args()

    if args.self_test:
        passed = self_test_verifiers()
        exit(0 if passed else 1)

    if args.tasks_file and args.answers_file:
        tasks = load_tasks_jsonl(args.tasks_file)
        with open(args.answers_file) as f:
            answers = json.load(f)

        results = evaluate_batch(tasks, answers)
        metrics = compute_metrics(results)

        out = Path(args.output_dir)
        save_results(results, str(out / "eval_results.jsonl"))
        save_metrics(metrics, str(out / "eval_metrics.json"))

        print("\nEvaluation Results:")
        print(f"  Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        for task_type, info in metrics["by_task_type"].items():
            print(f"  {task_type}: {info['accuracy']:.2%} "
                  f"({info['correct']}/{info['total']})")


if __name__ == "__main__":
    main()

