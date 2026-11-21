#!/usr/bin/env python3
"""E11 helper: prompt sizes and accuracy for prompts that fit in 8k vs need >8k.

Run with the GPU venv (transformers/tokenizers available).

    python scripts/final/e11_prompt_sizes.py \
        --harness experiments/review_tool_access/harnesses/baseline_b_v2_text.json \
        --split test_large --data-dir data \
        --run experiments/final/E11_notools_longctx/baseline_b_v2_text/test_large \
        --tokenizer /path/to/Qwen3-4B-snapshot \
        --out experiments/final/E11_notools_longctx/fit_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from graphharness.agent import get_harness  # noqa: E402
from graphharness.data_loader import load_split  # noqa: E402

LIMIT = 8192


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--run", required=True, help="run dir containing scores.jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    h = get_harness(args.harness)
    records = load_split(args.split, args.data_dir)
    sizes = {}
    for rec in records:
        messages = [{"role": "system", "content": h.system_message()},
                    {"role": "user", "content": h.user_message(rec)}]
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            n = len(tok(text, add_special_tokens=False)["input_ids"])
        except Exception:
            n = len(tok(h.system_message() + "\n" + h.user_message(rec))["input_ids"])
        sizes[rec["id"]] = n

    scores = {r["id"]: r for r in map(json.loads, open(Path(args.run) / "scores.jsonl")) if r}
    rows = []
    for pid, n in sizes.items():
        s = scores.get(pid, {})
        rows.append({"id": pid, "prompt_tokens": n,
                     "fits_8k": int(n <= LIMIT), "correct": int(bool(s.get("correct"))),
                     "parsed": int(bool(s.get("parsed")))})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "run", "id", "prompt_tokens",
                                          "fits_8k", "correct", "parsed"])
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({"split": args.split, "run": args.run, **r})

    def bucket(fit):
        xs = [r for r in rows if r["fits_8k"] == fit]
        return (len(xs), sum(x["correct"] for x in xs) / len(xs) if xs else None,
                sum(x["parsed"] for x in xs) / len(xs) if xs else None)

    n_fit, acc_fit, parsed_fit = bucket(1)
    n_over, acc_over, parsed_over = bucket(0)
    print(f"{args.split} {args.run}: fits8k n={n_fit} acc={acc_fit} parsed={parsed_fit}; "
          f"over8k n={n_over} acc={acc_over} parsed={parsed_over}; max={max(sizes.values())}")


if __name__ == "__main__":
    main()
