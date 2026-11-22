#!/usr/bin/env python3
"""Build report/hda_optimizer_loop.pdf (standalone loop explanation, no LaTeX).

Requires reportlab and matplotlib's DejaVu fonts.  Example:
    PYTHONPATH=/path/to/reportlab python report/make_optimizer_loop_pdf.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from reportlab.lib import colors
except ImportError:  # local fallback used during the study
    sys.path.insert(0, "/localscratch-nvme/1554120/opencode/pylibs")
    from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report/hda_optimizer_loop.pdf"
try:
    import matplotlib
    FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
except Exception:
    FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DejaVuMono", str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFont(TTFont("DejaVuSans", str(FONT_DIR / "DejaVuSans.ttf")))

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=9.3, leading=12.1,
                      spaceAfter=4.5, alignment=TA_LEFT)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.1, leading=10.2,
                       textColor=colors.HexColor("#444444"))
H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13.2, leading=16,
                    spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#111111"))
TITLE = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=18, leading=21,
                       spaceAfter=2)
SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=10, leading=12.5,
                     textColor=colors.HexColor("#555555"), spaceAfter=9)
CODE = ParagraphStyle("code", fontName="DejaVuMono", fontSize=6.35, leading=7.05,
                      textColor=colors.HexColor("#111111"))
CAP = ParagraphStyle("cap", parent=SMALL, fontSize=7.9, leading=9.8,
                     textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=8)
BULLET = ParagraphStyle("bullet", parent=BODY, leftIndent=11, bulletIndent=2,
                        spaceAfter=2.5)


def P(text, style=BODY):
    return Paragraph(text, style)


def B(text):
    return Paragraph(text, BULLET, bulletText="\u2022")


def code(text, font=6.35):
    st = ParagraphStyle("c", parent=CODE, fontSize=font, leading=font * 1.12)
    return Preformatted(text, st)


def table(data, widths, header=True, font=8.0, align="LEFT", zebra=True):
    body = []
    for i, row in enumerate(data):
        r = []
        for cell in row:
            if isinstance(cell, str):
                r.append(Paragraph(cell, ParagraphStyle("t", parent=BODY, fontSize=font,
                                                         leading=font * 1.22)))
            else:
                r.append(cell)
        body.append(r)
    t = Table(body, colWidths=widths, hAlign=align, repeatRows=1 if header else 0)
    cmds = [("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#333333")),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, colors.HexColor("#333333"))]
    if zebra:
        for i in range(1, len(body)):
            if i % 2 == 0:
                cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f4f4")))
    t.setStyle(TableStyle(cmds))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(0.7 * inch, 0.45 * inch, "HDA prompt-optimizer loop \u2014 22 November 2026")
    canvas.drawRightString(letter[0] - 0.7 * inch, 0.45 * inch, f"{doc.page}")
    canvas.restoreState()


DIAGRAM_LOOP = """\
                    ┌──────────────────────────────────┐
                    │ state S                          │
                    │ GEPA: candidate pool             │
                    │ SkillOpt: single incumbent       │
                    │ Meta-Harness: full history       │
                    │ EvoSkill: skill frontier (top 3) │
                    └───────────────┬──────────────────┘
                                    │ pick parent p
                                    V
                    ┌──────────────────────────────────┐
                    │ proposer (Qwen3-4B)              │
                    │ input: p + rollout traces        │
                    │        + fixed contract          │
                    │ output: new text / JSON edits    │
                    └───────────────┬──────────────────┘
                                    │ raw proposal
                                    V
                    ┌──────────────────────────────────┐
                    │ parse + validate                 │
                    │ lengths, no task/algorithm names │
                    │ no 3+ digit numbers, no solver   │
                    └───────┬──────────────────┬───────┘
                     reject │                  │ pass
                            V                  V
                     log rejection      evaluate child c
                     (0 rollouts)       20 instances, seed 200
                                               │
                                               V
                                       mean val_opt = s(c)
                                               │
                              ┌────────────────┴────────────────┐
                              │ s(c) >= s(p) + 0.02 ?           │
                              └───────┬─────────────────┬───────┘
                                 yes  │                 │ no
                                      V                 V
                              accept: S <- c        keep p
                              log + save            log rejection
                                      │
                                      └────────> next round"""

DIAGRAM_ROLLOUT = """\
candidate c
    │  HDA_PROMPT_FILE=c.json
    V
hda.agent subprocess                      (one per rollout)
    ├─ k = 5 examples
    ├─ iteration 1..3
    │    ├─ LLM reply: action + C++ code
    │    ├─ compile / run on examples
    │    └─ feedback: verdicts + validation accuracy
    ├─ H* = best_solution.cpp
    └─ test splits OFF
    │
    V
compile H*
run H* on val_opt[task]                    (40 graphs, relabeling check)
    │
    V
accuracy(c, task, condition, seed)"""


def build():
    doc = BaseDocTemplate(str(OUT), pagesize=letter,
                          leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                          topMargin=0.62 * inch, bottomMargin=0.62 * inch,
                          title="How the HDA prompt-optimizer loop works",
                          author="CS 5624 project")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame], onPage=footer)])

    s = []
    s += [P("How the HDA prompt-optimizer loop works", TITLE),
          P("Harness Developing Agent (HDA) \u2014 prompt-optimizer baselines", SUB)]

    s += [P("The proposer is an LLM call that writes candidate prompt text. The optimizer "
            "loop wraps that call with real HDA rollouts. A candidate is a JSON object with "
            "three strings \u2014 <font name='DejaVuMono' size='8'>strategy</font>, "
            "<font name='DejaVuMono' size='8'>hidden_procedure</font>, "
            "<font name='DejaVuMono' size='8'>iteration_note</font> \u2014 appended to the "
            "fixed prompt. Empty fields are the unoptimized prompt; the fixed contract, tool "
            "protocol, input format and graph template are never edited.")]

    s += [P("1. The loop", H1)]
    s += [KeepTogether([code(DIAGRAM_LOOP),
                        P("Figure 1. One optimizer round. The proposer sees text and traces; "
                          "every score comes from rollouts of the candidate prompt.", CAP)])]
    for t in [
        "<b>Pick a parent</b> from the optimizer state: a pool (GEPA), one incumbent "
        "(SkillOpt), the best so far (Meta-Harness), or a top-3 frontier (EvoSkill).",
        "<b>Propose.</b> The proposer gets the parent text, rollout traces, the fixed "
        "contract and the validator rules. It returns new text or structured JSON edits.",
        "<b>Validate.</b> Length limits, no fenced solver longer than 15 lines, no 3+ digit "
        "number, and no benchmark task or textbook algorithm name. A rejected proposal is "
        "logged and costs zero rollouts.",
        "<b>Evaluate.</b> A valid child runs on acceptance set A: ten tasks \u00d7 two "
        "conditions \u00d7 seed 200 = 20 rollouts. The score is mean <font name='DejaVuMono' "
        "size='8'>val_opt</font> accuracy.",
        "<b>Accept or reject.</b> Accept if the child beats the parent by 0.02, or ties "
        "within 0.02 with shorter text. Otherwise keep the parent.",
        "<b>Update state and repeat.</b> Accepted candidates enter the pool, replace the "
        "incumbent, or enter the frontier. Every step is appended to "
        "<font name='DejaVuMono' size='8'>history.jsonl</font> and "
        "<font name='DejaVuMono' size='8'>ledger.jsonl</font>.",
    ]:
        s += [B(t)]

    s += [KeepTogether([P("2. Inside one rollout", H1),
                        code(DIAGRAM_ROLLOUT),
                        P("Figure 2. One HDA rollout. The program is written on small "
                          "examples, then scored on a validation split the optimizer may use.", CAP)])]
    s += [P("The HDA gets <i>k</i> = 5 input/output examples for one task and one condition "
            "(<i>given</i> or <i>hidden</i>). It iterates three times; each iteration it "
            "replies with an action and C++ code, compiles and runs on the examples, and "
            "receives verdicts. Its best program H* is <font name='DejaVuMono' size='8'>"
            "best_solution.cpp</font>. Test evaluation is switched off, and a no-test check "
            "greps every log for test events and test split names.")]

    s += [P("3. What the proposer sees", H1)]
    s += [P("For each rollout in the reflection batch, the trace contains:")]
    for t in [
        "task and condition, plus the aggregate <font name='DejaVuMono' size='8'>val_opt</font> score;",
        "per iteration: the statement, actions taken, compile errors, example verdicts, "
        "and the validation feedback the HDA saw;",
        "the policy result and the final C++ code, truncated to 150 lines.",
    ]:
        s += [B(t)]
    s += [P("It never sees per-graph <font name='DejaVuMono' size='8'>val_opt</font> rows or "
            "answers, and never sees any <font name='DejaVuMono' size='8'>test_*</font> split.")]

    s += [P("4. Acceptance rule and the noise problem", H1)]
    s += [P("<font name='DejaVuMono' size='8'>accept if s(child) &gt;= s(parent) + 0.02, or "
            "|s(child) - s(parent)| &lt;= 0.02 and the child text is shorter.</font>")]
    s += [P("The same empty prompt, re-run on the 20-instance acceptance set, does not give "
            "the same score:")]
    s += [table([["Seed-200 run", "Score"],
                 ["baseline", "0.859"],
                 ["SkillOpt rerun", "0.825"],
                 ["Meta-Harness rerun", "0.890"],
                 ["EvoSkill rerun", "0.925"],
                 ["seeds 201/202 rechecks", "0.748 / 0.840 / 0.817 (baseline 0.815)"]],
                [2.0 * inch, 3.3 * inch], font=8.0)]
    s += [Spacer(1, 4),
          P("Run-to-run standard deviation is roughly 0.04\u20130.06, two to three times the "
            "0.02 margin. A 20-instance single-seed acceptance set cannot separate a small "
            "real improvement from sampling noise.")]

    s += [P("5. Where the four optimizers differ", H1)]
    s += [table([["Optimizer", "Parent choice", "Proposal format", "State update"],
                 ["GEPA", "candidate pool", "rewrite one field, round-robin", "add child if accepted"],
                 ["SkillOpt", "single incumbent", "JSON add / modify / delete edits", "replace incumbent if accepted"],
                 ["Meta-Harness", "best so far", "full candidate JSON", "keep every candidate in history"],
                 ["EvoSkill", "top-3 frontier", "add or rewrite one skill", "insert into frontier if +0.02"]],
                [0.95 * inch, 1.25 * inch, 2.35 * inch, 1.75 * inch], font=7.9)]

    table6 = table([["Optimizer", "Proposals", "Rejected", "Valid", "Best child", "Accepted"],
                 ["GEPA", "8", "7", "1", "0.784", "0"],
                 ["SkillOpt", "4", "1", "3", "0.825", "0"],
                 ["Meta-Harness", "4", "4", "0", "\u2014", "0"],
                 ["EvoSkill", "4", "4", "0", "\u2014", "0"]],
                [1.05 * inch, 0.85 * inch, 0.8 * inch, 0.65 * inch, 0.9 * inch, 0.8 * inch], font=8.2)
    s += [KeepTogether([P("6. What happened in this study", H1), table6, Spacer(1, 5),
          P("Sixteen of twenty proposals were rejected by the task-agnostic validator, mostly "
            "for naming <i>shortest path</i>, <i>Dijkstra</i>, <i>reachability</i> or "
            "<i>connected component</i>. The four valid children scored 0.784 and 0.825 "
            "against the 0.830 parent; none reached +0.02. Every optimizer therefore "
            "returned the empty prompt. The five prompt files are byte-identical, so one "
            "final grid serves all five and four redundant grids were not run.")])]

    s += [P("7. Cost of one round", H1)]
    s += [table([["Part", "Cost"],
                 ["Proposer call", "1 call, about 5.3k prompt tokens, seconds"],
                 ["Child evaluation", "20 rollouts, about 1.7M HDA tokens"],
                 ["Wall time, one a30", "about 15\u201320 min"],
                 ["Cache hit", "free (prompt hash + task + condition + seed)"]],
                [1.4 * inch, 4.1 * inch], font=8.2)]
    s += [Spacer(1, 4),
          P("Measured GPU-hours: shared baseline 0.20, GEPA 1.01, SkillOpt 1.11, "
            "Meta-Harness 0.88, EvoSkill 0.77, final grid 0.91; total 4.87 GPU-hours. "
            "The loop is dominated by 20 full HDA runs per valid child, not by the proposer.")]

    s += [P("8. Deviations and limits", H1)]
    for t in [
        "The proposer was the same Qwen3-4B checkpoint as the HDA; no larger model was "
        "available on the cluster.",
        "Our GEPA loop is GEPA-style: no Pareto front over instances, no minibatch-first "
        "evaluation, and no merge step.",
        "Our Meta-Harness proposer received one trace bundle per round rather than the "
        "read-only <font name='DejaVuMono' size='8'>ls/cat/grep</font> tool loop.",
        "EvoSkill evolved only the <font name='DejaVuMono' size='8'>strategy</font> field "
        "and used one diagnosis-plus-skill call per round.",
        "The parent score was the 3-seed baseline 0.8296 while the child was scored on seed "
        "200; correcting the parent to the seed-200 baseline 0.8588 would not change any "
        "decision, because 0.825 and 0.784 are both below it.",
    ]:
        s += [B(t)]

    s += [Spacer(1, 6),
          P("Code: <font name='DejaVuMono' size='8'>hda_pipeline/hda/opt/</font> "
            "(run.py, rollout.py, validate.py, common.py, cost.py). Results: "
            "<font name='DejaVuMono' size='8'>experiments/hda/optimizers/</font>. "
            "Tables and figures: <font name='DejaVuMono' size='8'>report/tables/hda_optimizers/</font> "
            "and <font name='DejaVuMono' size='8'>report/figures/hda_optimizers/</font>.", SMALL)]

    doc.build(s)
    print(OUT)


if __name__ == "__main__":
    build()
