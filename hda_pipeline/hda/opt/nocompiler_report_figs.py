"""Publication-style figures for the compiler-tool ablation.

Outputs (PDF + PNG):
  fig_nocompiler_accuracy  accuracy by split, compiler on vs off
  fig_nocompiler_effects   per-task deltas and win/tie/loss counts
  fig_nocompiler_cost      prompt tokens and failed submissions
"""
from __future__ import annotations
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments/hda/nocompiler_grid_all10"
OUT = ROOT / "report/figures/hda_nocompiler"
SPLITS = ["val_acc", "test_small", "test_medium", "test_large", "test_xl", "test_stress"]
LABELS = ["val", "small", "medium", "large", "xl", "stress"]
TASKS = ["shortest_path", "reachability", "connected_components", "cycle_detection",
         "diameter", "common_neighbors", "triangle_count", "bipartite_check",
         "mst_weight", "max_flow"]
SHORT = {"shortest_path": "shortest", "reachability": "reach",
         "connected_components": "components", "cycle_detection": "cycle",
         "diameter": "diameter", "common_neighbors": "common nb",
         "triangle_count": "triangle", "bipartite_check": "bipartite",
         "mst_weight": "MST", "max_flow": "max flow"}
ON, OFF = "#2b6cb0", "#dd6b20"


def rows():
    return list(csv.DictReader((SRC / "compare_all10.csv").open()))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def mean(rs, key):
    return statistics.mean(f(r[key]) for r in rs)


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def accuracy():
    R = rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85), sharey=True, constrained_layout=True)
    x = range(len(SPLITS)); w = 0.38
    for ax, cond in zip(axes, ("given", "hidden")):
        on = [r for r in R if r["condition"] == cond and r["arm"] == "on"]
        off = [r for r in R if r["condition"] == cond and r["arm"] == "off"]
        ax.bar([i - w/2 for i in x], [mean(on, s) for s in SPLITS], w, color=ON, label="compiler on")
        ax.bar([i + w/2 for i in x], [mean(off, s) for s in SPLITS], w, color=OFF, label="compiler off")
        ax.set_xticks(list(x)); ax.set_xticklabels(LABELS, fontsize=7)
        ax.set_ylim(0.3, 1.02)
        ax.set_title(f"{cond}  ({mean(on,'tokens_prompt')/1000:.1f}k -> "
                     f"{mean(off,'tokens_prompt')/1000:.1f}k prompt tokens)", fontsize=8)
        style(ax)
    axes[0].set_ylabel("accuracy", fontsize=8)
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Compiler-tool ablation: ten tasks, seeds 0-2", fontsize=9)
    save(fig, "fig_nocompiler_accuracy")


def effects():
    R = rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.45, 1]})
    # panel a: per-task delta on test_small
    ax = axes[0]
    entries = []
    for cond in ("given", "hidden"):
        for t in TASKS:
            on = [r for r in R if r["condition"] == cond and r["arm"] == "on" and r["task"] == t]
            off = [r for r in R if r["condition"] == cond and r["arm"] == "off" and r["task"] == t]
            entries.append((cond, t, mean(off, "test_small") - mean(on, "test_small")))
    entries.sort(key=lambda e: e[2])
    y = range(len(entries))
    colors = [("#9ecae1" if e[0] == "given" else "#fdae6b") for e in entries]
    ax.barh(list(y), [e[2] for e in entries], color=colors, edgecolor="none")
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels([f"{SHORT[e[1]]} ({e[0][0]})" for e in entries], fontsize=6.5)
    ax.set_xlabel("compiler-off minus compiler-on, test small", fontsize=7.5)
    ax.set_xlim(-1.05, 0.4)
    style(ax)
    ax.legend(handles=[Patch(color="#9ecae1", label="given"), Patch(color="#fdae6b", label="hidden")],
              frameon=False, fontsize=6.5, loc="lower right")
    ax.set_title("(a) per-task effect", fontsize=8)

    # panel b: win / tie / loss counts per split
    ax = axes[1]
    eps = 1e-9
    wins, ties, losses = [], [], []
    for s in SPLITS:
        w = t = l = 0
        for task in TASKS:
            for cond in ("given", "hidden"):
                for seed in ("0", "1", "2"):
                    on = next(r for r in R if r["task"] == task and r["condition"] == cond
                              and r["seed"] == seed and r["arm"] == "on")
                    off = next(r for r in R if r["task"] == task and r["condition"] == cond
                               and r["seed"] == seed and r["arm"] == "off")
                    d = f(off[s]) - f(on[s])
                    if d > eps: w += 1
                    elif d < -eps: l += 1
                    else: t += 1
        wins.append(w); ties.append(t); losses.append(l)
    ax.barh(range(len(SPLITS)), wins, color="#2f855a", label="off better")
    ax.barh(range(len(SPLITS)), ties, left=wins, color="0.8", label="tie")
    ax.barh(range(len(SPLITS)), losses, left=[a + b for a, b in zip(wins, ties)],
            color="#c53030", label="on better")
    for i, (w, t, l) in enumerate(zip(wins, ties, losses)):
        ax.text(w + t + l + 1, i, f"{w}/{t}/{l}", va="center", fontsize=6)
    ax.set_yticks(range(len(SPLITS))); ax.set_yticklabels(LABELS, fontsize=7)
    ax.set_xlim(0, 72)
    ax.set_xlabel("paired runs (of 60)", fontsize=7.5)
    ax.set_title("(b) win / tie / loss", fontsize=8)
    style(ax)
    fig.legend(handles=[Patch(color="#2f855a", label="off better"),
                        Patch(color="0.8", label="tie"),
                        Patch(color="#c53030", label="on better")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=6.8)
    save(fig, "fig_nocompiler_effects")


def cost():
    R = rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.4, 1]})
    # panel a: prompt tokens per run
    ax = axes[0]
    groups = [("given", "on"), ("given", "off"), ("hidden", "on"), ("hidden", "off")]
    data = [[f(r["tokens_prompt"]) / 1000 for r in R if r["condition"] == c and r["arm"] == a]
            for c, a in groups]
    pos = [1, 1.7, 3, 3.7]
    bp = ax.boxplot(data, positions=pos, widths=0.45, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1))
    for patch, (c, a) in zip(bp["boxes"], groups):
        patch.set_facecolor(ON if a == "on" else OFF); patch.set_alpha(0.85)
    for p, vals in zip(pos, data):
        ax.plot([p] * len(vals), vals, ".", color="0.25", ms=2, alpha=0.6)
    ax.set_xticks(pos); ax.set_xticklabels(["given\non", "given\noff", "hidden\non", "hidden\noff"], fontsize=7)
    ax.set_ylabel("prompt tokens per run (k)", fontsize=8)
    ax.set_title("(a) token cost", fontsize=8)
    style(ax)

    # panel b: failed submissions
    ax = axes[1]
    summary = list(csv.DictReader((SRC / "summary.csv").open()))
    allr = next(r for r in summary if r["condition"] == "all" and r["arm"] == "on")
    offr = next(r for r in summary if r["condition"] == "all" and r["arm"] == "off")
    on_c, off_c = int(allr["compile_fail_submissions"]), int(offr["compile_fail_submissions"])
    on_o, off_o = int(allr["failed_submissions"]) - on_c, int(offr["failed_submissions"]) - off_c
    ax.bar([0], [on_c], color="#c53030", label="compile error")
    ax.bar([0], [on_o], bottom=[on_c], color="0.75", label="other")
    ax.bar([1], [off_c], color="#c53030")
    ax.bar([1], [off_o], bottom=[off_c], color="0.75")
    ax.text(0, on_c + on_o + 0.4, f"{on_c + on_o}", ha="center", fontsize=8)
    ax.text(1, off_c + off_o + 0.4, f"{off_c + off_o}", ha="center", fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["compiler on", "compiler off"], fontsize=7)
    ax.set_ylabel("failed submissions (60 runs)", fontsize=8)
    ax.set_ylim(0, 21)
    ax.set_title("(b) submission failures", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    style(ax)
    save(fig, "fig_nocompiler_cost")


if __name__ == "__main__":
    accuracy(); effects(); cost()
    print("figures ->", OUT)
