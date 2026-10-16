"""Read a model's text answer and score it against the exact ground truth.

How an answer is read:
1. If the response has ``<answer> ... </answer>`` tags, only the last tagged
   part is used. Otherwise the whole response is used.
2. The LAST ``[ ... ]`` list in that text is the answer.
3. Items are split on commas and matched to node names in the graph.
   Names that contain commas (possible in GraphArena) are handled too.

Each answer gets three flags, like GraphArena reports:
* parsed   - a bracketed list was found
* feasible - it is a valid object for the task (a real path, a real clique...)
* correct  - it is valid AND matches the optimal / true answer

Command line use (scores a file of model responses):

    python -m graphharness.evaluate_answer \
        --data data/test_medium.jsonl --predictions preds.jsonl --out scores.jsonl

where each line of ``preds.jsonl`` is ``{"id": ..., "response": "..."}``.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

from .data_loader import load_jsonl, save_jsonl, to_graph
from .tasks import get_task

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BRACKET = re.compile(r"\[([^\[\]]*)\]", re.DOTALL)
_STRIP = " \t\n\r'\"`*.;:"


def extract_list(response: str) -> str | None:
    """Return the text inside the final [ ... ] of the answer, or None."""
    if response is None:
        return None
    tagged = _ANSWER_TAG.findall(response)
    text = tagged[-1] if tagged else response
    matches = _BRACKET.findall(text)
    if not matches:
        return None
    return matches[-1]


def _lookup_table(nodes) -> tuple[dict, dict]:
    exact = {str(v): v for v in nodes}
    lower = {}
    for k, v in exact.items():
        lower.setdefault(k.lower(), v)
    return exact, lower


def _match(token: str, exact: dict, lower: dict):
    t = token.strip(_STRIP)
    if t in exact:
        return exact[t]
    if t.lower() in lower:
        return lower[t.lower()]
    # "node 12" -> "12"
    m = re.fullmatch(r"node\s+(.+)", t, re.IGNORECASE)
    if m:
        return _match(m.group(1), exact, lower)
    return None


_MISSING = object()
_MAX_COMMAS_IN_NAME = 4


def to_nodes(inner: str, graph_nodes) -> tuple[list, list[str]]:
    """Turn the inside of a [...] list into graph nodes.

    Returns (nodes, unknown_tokens). Tokens that are not nodes are reported,
    not silently dropped, so a made-up node makes the answer infeasible.
    """
    exact, lower = _lookup_table(graph_nodes)
    pieces = [p for p in inner.split(",")]
    if all(p.strip(_STRIP) == "" for p in pieces):
        return [], []

    nodes, unknown = [], []
    i = 0
    while i < len(pieces):
        if pieces[i].strip(_STRIP) == "":
            i += 1
            continue
        # try the longest run of pieces (up to _MAX_COMMAS_IN_NAME + 1) that
        # forms a known name, since a name may itself contain commas
        found = _MISSING
        for j in range(min(len(pieces), i + _MAX_COMMAS_IN_NAME + 1), i, -1):
            cand = ",".join(pieces[i:j])
            v = _match(cand, exact, lower)
            if v is not None:
                found, i = v, j
                break
        if found is _MISSING:
            unknown.append(pieces[i].strip(_STRIP))
            i += 1
        else:
            nodes.append(found)
    return nodes, unknown


def score_response(record: dict, response: str | None) -> dict:
    """Score one response for one problem record."""
    out = {
        "id": record["id"],
        "task": record["task"],
        "split": record.get("split"),
        "n_nodes": len(record["graph"]["nodes"]),
        "parsed": False,
        "feasible": False,
        "correct": False,
        "value": None,
        "note": "",
        "answer_nodes": None,
    }
    inner = extract_list(response)
    if inner is None:
        out["note"] = "no [ ... ] answer found" if response is not None else "no prediction"
        return out
    out["parsed"] = True

    G = to_graph(record)
    nodes, unknown = to_nodes(inner, G.nodes)
    out["answer_nodes"] = nodes
    if unknown:
        out["note"] = f"unknown node(s): {unknown[:5]}"
        return out

    task = get_task(record["task"])
    check = task.check(G, record["params"], record["answer"], nodes)
    out.update(feasible=check.feasible, correct=check.correct,
               value=check.value, note=check.note)
    return out


def summarize(scores: list[dict]) -> list[dict]:
    """Accuracy per (split, task), plus an overall row per split."""
    groups = defaultdict(list)
    for s in scores:
        groups[(s["split"], s["task"])].append(s)
        groups[(s["split"], "ALL")].append(s)
    rows = []
    for (split, task), items in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] == "ALL", kv[0][1])):
        n = len(items)
        rows.append({
            "split": split,
            "task": task,
            "n": n,
            "parsed": sum(s["parsed"] for s in items) / n,
            "feasible": sum(s["feasible"] for s in items) / n,
            "accuracy": sum(s["correct"] for s in items) / n,
            "mean_nodes": sum(s["n_nodes"] for s in items) / n,
        })
    return rows


def format_table(rows: list[dict]) -> str:
    head = f"{'split':<12} {'task':<22} {'n':>5} {'nodes':>7} {'parsed':>7} {'feasible':>9} {'accuracy':>9}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{str(r['split']):<12} {r['task']:<22} {r['n']:>5} {r['mean_nodes']:>7.0f} "
            f"{r['parsed']:>7.1%} {r['feasible']:>9.1%} {r['accuracy']:>9.1%}"
        )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="problem JSONL file(s)")
    ap.add_argument("--predictions", required=True, help="JSONL with {id, response} per line")
    ap.add_argument("--out", help="write per-problem scores here (JSONL)")
    args = ap.parse_args(argv)

    records = [r for p in args.data for r in load_jsonl(p)]
    preds = {p["id"]: p.get("response") for p in load_jsonl(args.predictions)}
    scores = [score_response(r, preds.get(r["id"])) for r in records]

    missing = sum(r["id"] not in preds for r in records)
    if missing:
        print(f"warning: {missing} problem(s) have no prediction; counted as wrong")
    print(format_table(summarize(scores)))
    if args.out:
        save_jsonl(scores, args.out)
        print(f"wrote {len(scores)} scores to {Path(args.out)}")


if __name__ == "__main__":
    main()
