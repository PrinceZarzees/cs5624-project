"""O7: optimizer figures (PDF + PNG) under report/figures/hda_optimizers."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OPT = ROOT / "experiments/hda/optimizers"
OUT = ROOT / "report/figures/hda_optimizers"
SPLITS = ["test_small", "test_medium", "test_large", "test_xl", "test_stress"]
BASE = 0.8295833333333333
BASE_SEED_MEANS = [0.85875, 0.76875, 0.86125]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)


def fig_tokens_accept():
    cost = {r["optimizer"]: r for r in csv.DictReader((OPT / "cost.csv").open())}
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    lo = BASE - statistics.pstdev(BASE_SEED_MEANS)
    hi = BASE + statistics.pstdev(BASE_SEED_MEANS)
    ax.axhspan(lo, hi, color="0.85", label="baseline seed spread", zorder=0)
    ax.axhline(BASE, color="0.4", lw=1, ls="--", zorder=1)
    styles = {"gepa": ("C0", "o"), "skillopt": ("C1", "s"),
              "meta_harness": ("C2", "^"), "evoskill": ("C3", "D")}
    for name, (color, marker) in styles.items():
        r = cost.get(name)
        if not r:
            continue
        toks = (int(r["hda_prompt_tokens"] or 0) + int(r["hda_completion_tokens"] or 0)) / 1e6
        ax.plot([0, toks], [BASE, BASE], color=color, marker=marker, ms=5,
                label=f"{name.replace('_',' ')}")
    # valid children that were evaluated but rejected
    rejected = [("skillopt", 0.825), ("gepa", 0.7838)]
    for name, score in rejected:
        r = cost.get(name)
        toks = (int(r["hda_prompt_tokens"] or 0) + int(r["hda_completion_tokens"] or 0)) / 1e6
        ax.plot([toks], [score], marker="x", color=styles[name][0], ms=7)
    ax.set_xlabel("cumulative HDA tokens (millions)")
    ax.set_ylabel("best acceptance score on A")
    ax.set_ylim(0.72, 0.90)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("No accepted edit: score never moves above the parent", fontsize=8)
    save(fig, "fig1_tokens_vs_acceptance")


def fig_accuracy_splits():
    rows = list(csv.DictReader((OPT / "final_summary.csv").open()))
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for cond, color in (("given", "C0"), ("hidden", "C1")):
        ys, es = [], []
        for s in SPLITS:
            per_seed = []
            for seed in ("0", "1", "2"):
                v = [float(r[s]) for r in rows if r["condition"] == cond and r["seed"] == seed and r[s]]
                if v:
                    per_seed.append(statistics.mean(v))
            ys.append(statistics.mean(per_seed))
            es.append(statistics.pstdev(per_seed) if len(per_seed) > 1 else 0.0)
        ax.errorbar(range(len(SPLITS)), ys, yerr=es, marker="o", capsize=3,
                    color=color, label=cond)
    ax.set_xticks(range(len(SPLITS)))
    ax.set_xticklabels([s.replace("test_", "") for s in SPLITS])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False)
    ax.set_title("Unoptimized prompt (all five prompt files identical)", fontsize=8)
    save(fig, "fig2_accuracy_vs_split")


def fig_verdicts():
    files = sorted((OPT / "final").glob("unoptimized/*/final.json"))
    mixes = {}
    for cond in ("given", "hidden"):
        agg = {s: {} for s in SPLITS}
        for f in files:
            d = json.loads(f.read_text())
            if d["condition"] != cond:
                continue
            for s in SPLITS:
                for k, v in ((d.get("test", {}).get(s) or {}).get("verdicts") or {}).items():
                    agg[s][k] = agg[s].get(k, 0) + v
        mixes[cond] = agg
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    keys = ["AC", "WA", "TLE", "RE"]
    colors = {"AC": "C2", "WA": "C1", "TLE": "C0", "RE": "0.5"}
    for ax, cond in zip(axes, ("given", "hidden")):
        bottom = [0] * len(SPLITS)
        for k in keys:
            vals = [mixes[cond][s].get(k, 0) for s in SPLITS]
            ax.bar(range(len(SPLITS)), vals, bottom=bottom, label=k, color=colors[k])
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_xticks(range(len(SPLITS)))
        ax.set_xticklabels([s.replace("test_", "") for s in SPLITS], fontsize=7)
        ax.set_title(cond, fontsize=8)
    axes[0].set_ylabel("verdicts (all tasks, 3 seeds)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Verdict mix by split, unoptimized prompt", fontsize=8)
    save(fig, "fig3_verdict_mix")


if __name__ == "__main__":
    fig_tokens_accept(); fig_accuracy_splits(); fig_verdicts()
    print("figures ->", OUT)
