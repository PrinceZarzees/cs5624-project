"""Validator for HDA prompt candidates (only the three editable fields)."""
from __future__ import annotations

import re

LIMITS = {"strategy": 3000, "hidden_procedure": 1500, "iteration_note": 800}
FIELDS = tuple(LIMITS)

FORBIDDEN = [
    r"shortest\s*path", r"reachab", r"connected\s*component", r"cycle\s*detection",
    r"diameter", r"common\s*neighbo", r"triangle", r"bipartite", r"spanning\s*tree",
    r"\bMST\b", r"max(?:imum)?\s*flow", r"min(?:imum)?\s*cut", r"clique",
    r"independent\s*set", r"vertex\s*cover", r"dijkstra", r"bellman", r"floyd",
    r"warshall", r"kruskal", r"prim", r"bor[uů]vka", r"dinic", r"edmonds",
    r"karp", r"ford", r"fulkerson", r"push[-\s]*relabel", r"preflow", r"tarjan",
    r"kosaraju", r"johnson", r"hopcroft",
]


def _fenced_lines(text: str) -> int:
    total = 0
    for m in re.finditer(r"```.*?```", text, re.S):
        total = max(total, m.group(0).count("\n") + 1)
    return total


def validate_candidate(candidate: dict) -> dict:
    reasons = []
    if not isinstance(candidate, dict):
        return {"ok": False, "reasons": ["candidate is not an object"]}
    for field in FIELDS:
        value = candidate.get(field, "")
        if not isinstance(value, str):
            reasons.append(f"{field}: not a string")
            continue
        if len(value) > LIMITS[field]:
            reasons.append(f"{field}: {len(value)} chars > {LIMITS[field]}")
        if _fenced_lines(value) > 15:
            reasons.append(f"{field}: fenced code block longer than 15 lines")
        if re.search(r"\b\d{3,}\b", value):
            reasons.append(f"{field}: contains a 3+ digit number")
        for pat in FORBIDDEN:
            if re.search(pat, value, re.I):
                reasons.append(f"{field}: names a benchmark task/algorithm ({pat})")
    return {"ok": not reasons, "reasons": reasons}
