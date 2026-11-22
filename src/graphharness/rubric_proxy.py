"""SkillLift-style cheap proxy for ranking candidate harnesses.

The judge reads only a candidate harness (prompt + rules) and scores it on a
dense rubric. Those scores are compared with the validation accuracy that was
actually measured during optimization, so no full large-graph rollouts are
needed to test the proxy.

    python -m graphharness.rubric_proxy \
        --run experiments/qwen3_4b_opt_a/optimizer \
        --run experiments/qwen3_4b/optimizer \
        --model qwen3-4b --base-url http://127.0.0.1:8000/v1 \
        --out experiments/review_rubric_proxy/results.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .agent import Harness
from .cli_common import add_model_arguments, build_model

RUBRIC = """\
You are ranking LLM graph-reasoning harnesses before running them. A harness is
a system prompt plus rules; it may use a Python/NetworkX sandbox.

Score the harness 1-5 on each dimension:
1. algorithmic_soundness: recommends correct, appropriate graph algorithms.
2. verification: requires checking paths/components/cycles/cliques before answering.
3. scalability: likely to work on 100-2000 node graphs without context blowup or exponential search.
4. no_shortcuts: does not rely on numeric node ids, small-graph ranges, or memorized answers.
5. output_protocol: clear, machine-checkable final-answer format.
6. tool_use: makes effective use of the sandbox when tools are available.
7. generality: not specific to one task, dataset, or node labeling.

Return only JSON:
{"scores": {"algorithmic_soundness": n, ...},
 "overall": 0-100,
 "reason": "short explanation"}
"""


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = np.array(_rank(x)), np.array(_rank(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_accuracy(scores: list[float], truth: list[float], min_gap: float = 0.025) -> float:
    correct = total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if abs(truth[i] - truth[j]) < min_gap:
                continue
            total += 1
            correct += (scores[i] > scores[j]) == (truth[i] > truth[j])
    return correct / total if total else float("nan")


def collect_candidates(run_dirs: list[str]) -> list[dict]:
    out = []
    for run_dir in run_dirs:
        run = Path(run_dir)
        history_path = run / "history.jsonl"
        if not history_path.exists():
            continue
        history = {
            row.get("round"): row
            for row in (json.loads(line) for line in history_path.read_text().splitlines() if line.strip())
        }
        for path in sorted((run / "candidates").glob("round_*.json")):
            round_no = int(re.search(r"round_(\d+)", path.name).group(1))
            row = history.get(round_no, {})
            val = row.get("val_accuracy")
            if val is None:
                continue
            out.append({
                "id": f"{run.parent.name}/{path.stem}",
                "path": str(path),
                "harness": Harness.load(path),
                "val_accuracy": float(val),
                "train_accuracy": row.get("train_accuracy"),
                "accepted": row.get("accepted"),
            })
    return out


def static_features(h: Harness) -> dict:
    prompt = h.system_message()
    rules = "\n".join(h.rules)
    return {
        "prompt_chars": len(prompt),
        "rule_count": len(h.rules),
        "numeric_node_pattern": bool(re.search(r"\bnode\s+\d+\b", rules, re.IGNORECASE)),
        "edge_list_view": int(h.graph_view == "edge_list"),
    }


def judge(model, harness: Harness) -> dict:
    prompt = (
        "## Candidate harness\n"
        f"```json\n{json.dumps(harness.to_dict(), indent=2)}\n```\n\n"
        "Return only the JSON rubric scores."
    )
    result = model.chat([
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": prompt},
    ], temperature=0.0, max_tokens=800)
    text = result.text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"proxy": None, "error": "no JSON", "raw": text[:500]}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"proxy": None, "error": "bad JSON", "raw": text[:500]}
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="optimizer output dir")
    ap.add_argument("--out", default="experiments/review_rubric_proxy/results.json")
    ap.add_argument("--md", default="experiments/review_rubric_proxy/results.md")
    add_model_arguments(ap)
    args = ap.parse_args(argv)
    model = build_model(args)

    candidates = collect_candidates(args.run)
    if not candidates:
        raise SystemExit("no candidates with val_accuracy found")
    for c in candidates:
        c["static"] = static_features(c["harness"])
        c["rubric"] = judge(model, c["harness"])
        scores = (c["rubric"].get("scores") or {}) if isinstance(c["rubric"], dict) else {}
        c["proxy"] = c["rubric"].get("overall") if isinstance(c["rubric"], dict) else None
        if c["proxy"] is None and scores:
            c["proxy"] = 20.0 * sum(float(v) for v in scores.values()) / len(scores)

    usable = [c for c in candidates if c["proxy"] is not None]
    truth = [c["val_accuracy"] for c in usable]
    proxy = [float(c["proxy"]) for c in usable]
    prompt_len = [c["static"]["prompt_chars"] for c in usable]
    rule_count = [c["static"]["rule_count"] for c in usable]
    ranked = sorted(usable, key=lambda c: c["proxy"], reverse=True)
    best_actual = max(usable, key=lambda c: c["val_accuracy"]) if usable else None
    top3 = ranked[:3]
    payload = {
        "n_candidates": len(candidates),
        "n_usable": len(usable),
        "spearman_proxy": spearman(proxy, truth),
        "spearman_prompt_length": spearman([float(x) for x in prompt_len], truth),
        "spearman_rule_count": spearman([float(x) for x in rule_count], truth),
        "pairwise_proxy": pairwise_accuracy(proxy, truth),
        "top3_recall": bool(best_actual and any(c["id"] == best_actual["id"] for c in top3)),
        "best_actual": None if best_actual is None else {
            "id": best_actual["id"], "val_accuracy": best_actual["val_accuracy"],
            "proxy": best_actual["proxy"],
        },
        "top3": [
            {"id": c["id"], "val_accuracy": c["val_accuracy"], "proxy": c["proxy"]}
            for c in top3
        ],
        "candidates": [
            {
                "id": c["id"], "val_accuracy": c["val_accuracy"], "proxy": c["proxy"],
                "rubric": c["rubric"], **c["static"],
            }
            for c in candidates
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Rubric proxy ranking",
        "",
        f"- candidates: {payload['n_usable']}",
        f"- Spearman(proxy, val accuracy): {payload['spearman_proxy']:.3f}",
        f"- Spearman(prompt length, val): {payload['spearman_prompt_length']:.3f}",
        f"- Spearman(rule count, val): {payload['spearman_rule_count']:.3f}",
        f"- pairwise ranking accuracy: {payload['pairwise_proxy']:.1%}",
        f"- best actual candidate in proxy top-3: {payload['top3_recall']}",
        "",
        "| candidate | val accuracy | rubric proxy |",
        "|---|---:|---:|",
    ]
    for c in sorted(candidates, key=lambda c: c["val_accuracy"], reverse=True):
        lines.append(f"| {c['id']} | {c['val_accuracy']:.1%} | {c['proxy']} |")
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in [
        "n_usable", "spearman_proxy", "pairwise_proxy", "top3_recall",
    ]}, indent=2))


if __name__ == "__main__":
    main()
