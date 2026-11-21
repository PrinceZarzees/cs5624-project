#!/usr/bin/env python3
"""List every AI-DRAFT block in report/ so authors can rewrite them.

Usage: python scripts/final/list_ai_drafts.py
Writes report/AI_DRAFTS.md and prints the same table.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "report"


def blocks(path: Path):
    start = None
    first = ""
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if re.search(r">>>\s*AI-DRAFT", line):
            start = i
            first = line.strip()
        elif re.search(r"<<<\s*AI-DRAFT", line) and start is not None:
            yield start, i, first
            start = None


def main():
    rows = []
    for path in sorted(REPORT.rglob("*")):
        if path.suffix not in {".tex", ".md"} or path.name == "AI_DRAFTS.md":
            continue
        for a, b, first in blocks(path):
            rows.append((path.relative_to(ROOT), a, b, first))
    out = ["# AI draft checklist", "",
           "Every block below was written by the agent and must be rewritten or",
           "explicitly approved by the authors before submission.", "",
           "| file | lines | first line |", "|---|---|---|"]
    for p, a, b, first in rows:
        out.append(f"| `{p}` | {a}-{b} ({b-a+1} lines) | `{first[:100]}` |")
    if not rows:
        out.append("| – | – | no AI-DRAFT blocks found |")
    text = "\n".join(out) + "\n"
    (REPORT / "AI_DRAFTS.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
