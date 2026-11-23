"""Figure and LaTeX table for the compiler-tool ablation."""
from __future__ import annotations
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments/hda/nocompiler_grid_all10/summary.csv"
FIG = ROOT / "report/figures/hda_nocompiler"
TAB = ROOT / "report/tables/hda_nocompiler"
SPLITS = ["val_acc", "test_small", "test_medium", "test_large", "test_xl", "test_stress"]
LABELS = ["val", "small", "medium", "large", "xl", "stress"]


def main():
    rows = list(csv.DictReader(SRC.open()))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    width = 0.38
    for ax, cond in zip(axes, ("given", "hidden")):
        on = next(r for r in rows if r["condition"] == cond and r["arm"] == "on")
        off = next(r for r in rows if r["condition"] == cond and r["arm"] == "off")
        x = range(len(SPLITS))
        ax.bar([i - width / 2 for i in x], [float(on[s]) for s in SPLITS], width,
               label="compiler on", color="C0")
        ax.bar([i + width / 2 for i in x], [float(off[s]) for s in SPLITS], width,
               label="compiler off", color="C1")
        ax.set_xticks(list(x)); ax.set_xticklabels(LABELS, fontsize=7)
        ax.set_ylim(0.3, 1.02)
        ax.set_title(cond, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("accuracy")
    axes[1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Compiler-tool ablation, 10 tasks x 3 seeds", fontsize=9)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_nocompiler.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)

    TAB.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Condition & Arm & val & small & medium & large & xl & stress \\", r"\midrule"]
    for cond in ("given", "hidden"):
        for arm in ("on", "off"):
            r = next(x for x in rows if x["condition"] == cond and x["arm"] == arm)
            lines.append(f"{cond} & {arm} & " + " & ".join(f"{float(r[s]):.3f}" for s in SPLITS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TAB / "tab_nocompiler.tex").write_text("\n".join(lines) + "\n")
    print(FIG / "fig_nocompiler.pdf")
    print(TAB / "tab_nocompiler.tex")


if __name__ == "__main__":
    main()
