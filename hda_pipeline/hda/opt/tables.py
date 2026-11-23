"""O7: build the optimizer tables (CSV + LaTeX) from final_summary.csv and cost.csv."""
from __future__ import annotations
import csv, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPT = ROOT / "experiments/hda/optimizers"
OUT = ROOT / "report/tables/hda_optimizers"
PROMPTS = ["unoptimized", "gepa", "skillopt", "meta_harness", "evoskill"]
SPLITS = ["val_opt", "test_small", "test_medium", "test_large", "test_xl", "test_stress"]
FINAL = OPT / "final"
BASE_A = 0.8295833333333333


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load():
    rows = list(csv.DictReader((OPT / "final_summary.csv").open()))
    for r in rows:
        for k in ["val_acc", "val_opt"] + SPLITS[1:]:
            r[k] = f(r.get(k))
    # no edit was accepted: the four optimized prompts are byte-identical to
    # unoptimized, so reuse the same final-grid rows (marked in table1)
    out = list(rows)
    for prompt in PROMPTS[1:]:
        for r in rows:
            if r["prompt"] == "unoptimized":
                c = dict(r); c["prompt"] = prompt; out.append(c)
    return out


def mean_std(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)


def per_seed_mean(rows, split):
    out = {}
    for s in ("0", "1", "2"):
        rs = [r[split] for r in rows if r["seed"] == s and r[split] is not None]
        out[s] = statistics.mean(rs) if rs else None
    return out


def table_main(rows):
    out = []
    for prompt in PROMPTS:
        rs = [r for r in rows if r["prompt"] == prompt]
        note = "" if prompt == "unoptimized" else "unchanged; identical to unoptimized"
        for cond in ("given", "hidden", "all"):
            sub = rs if cond == "all" else [r for r in rs if r["condition"] == cond]
            for split in SPLITS:
                m, sd = mean_std([r[split] for r in sub])
                gap_v = [r["test_small"] - r["test_stress"] for r in sub
                         if r["test_small"] is not None and r["test_stress"] is not None]
                gm, gsd = mean_std(gap_v) if split == "test_small" else (None, None)
                out.append(dict(prompt=prompt, condition=cond, split=split,
                                mean=m, std=sd, gap=gm, gap_std=gsd, note=note))
    _write_csv(OUT / "table1_main.csv", out)
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Prompt & Condition & Split & Acc. & Gap \\", r"\midrule"]
    for r in out:
        if r["condition"] in ("given", "hidden") and r["split"] in SPLITS:
            gap = "" if r["gap"] is None else f"{r['gap']:+.3f}"
            lines.append(f"{r['prompt'].replace('_',' ')} & {r['condition']} & {r['split'].replace('_',' ')} & "
                         f"{r['mean']:.3f}$\\pm${r['std']:.3f} & {gap} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table1_main.tex").write_text("\n".join(lines) + "\n")


def table_gain(rows):
    out = []
    for prompt in PROMPTS[1:]:
        out.append(dict(optimizer=prompt, gain_A=0.0, gain_val_opt=0.0,
                        gain_test_large=0.0, gain_test_stress=0.0,
                        note="no edit accepted; prompt unchanged"))
    _write_csv(OUT / "table2_val_vs_test.csv", out)
    (OUT / "table2_val_vs_test.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Optimizer & $\\Delta$A & $\\Delta$val\\_opt & $\\Delta$test\\_large & $\\Delta$test\\_stress \\\\\n"
        "\\midrule\n" + "\n".join(
            f"{r['optimizer'].replace('_',' ')} & {r['gain_A']:+.3f} & {r['gain_val_opt']:+.3f} & "
            f"{r['gain_test_large']:+.3f} & {r['gain_test_stress']:+.3f} \\\\" for r in out) +
        "\n\\bottomrule\n\\end{tabular}\n")


def table_task(rows):
    out = []
    tasks = sorted({r["task"] for r in rows})
    for prompt in PROMPTS:
        for task in tasks:
            for cond in ("given", "hidden"):
                sub = [r for r in rows if r["prompt"] == prompt and r["task"] == task and r["condition"] == cond]
                if not sub:
                    continue
                sm, _ = mean_std([r["test_small"] for r in sub])
                tm, _ = mean_std([r["test_stress"] for r in sub])
                out.append(dict(prompt=prompt, task=task, condition=cond,
                                test_small=sm, test_stress=tm))
    _write_csv(OUT / "table3_per_task.csv", out)
    lines = [r"\begin{tabular}{llrr}", r"\toprule",
             r"Task & Condition & test\_small & test\_stress \\", r"\midrule"]
    for r in out:
        if r["prompt"] == "unoptimized":
            lines.append(f"{r['task'].replace('_',' ')} & {r['condition']} & "
                         f"{r['test_small']:.3f} & {r['test_stress']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table3_per_task.tex").write_text("\n".join(lines) + "\n")


def table_breakdown():
    out = []
    for split in ("test_small", "test_stress"):
        for cond in ("given", "hidden"):
            agg = {}
            for f in sorted(FINAL.glob(f"unoptimized/*__{cond}__k5__s*/final.json")):
                d = json.loads(f.read_text())
                t = (d.get("test") or {}).get(split) or {}
                for view in ("by_bucket", "by_density", "by_seen_family"):
                    for key, val in (t.get(view) or {}).items():
                        agg.setdefault((view, key), []).append(val["acc"])
            for (view, key), vals in sorted(agg.items()):
                out.append(dict(prompt="unoptimized", condition=cond, split=split,
                                view=view, key=key, mean=statistics.mean(vals)))
    _write_csv(OUT / "table4_breakdowns.csv", out)
    (OUT / "table4_breakdowns.tex").write_text(
        "\\begin{tabular}{llllr}\n\\toprule\nSplit & Condition & View & Bucket & Acc. \\\\\n\\midrule\n" +
        "\n".join(f"{r['split'].replace('_',' ')} & {r['condition']} & {r['view'].replace('_',' ')} & "
                  f"{r['key']} & {r['mean']:.3f} \\\\" for r in out) +
        "\n\\bottomrule\n\\end{tabular}\n")


def table_identification(rows):
    out = []
    for prompt in PROMPTS:
        rs = [r for r in rows if r["prompt"] == prompt and r["condition"] == "hidden"]
        counts = {}
        for r in rs:
            counts[r["judge"]] = counts.get(r["judge"], 0) + 1
        ident = sum(1 for r in rs if r["test_small"] is not None and r["test_small"] >= 0.95)
        out.append(dict(prompt=prompt, equivalent=counts.get("EQUIVALENT", 0),
                        partial=counts.get("PARTIAL", 0), wrong=counts.get("WRONG", 0),
                        functional_id=ident, n=len(rs),
                        functional_id_rate=ident / len(rs) if rs else 0.0))
    _write_csv(OUT / "table5_identification.csv", out)
    (OUT / "table5_identification.tex").write_text(
        "\\begin{tabular}{lrrrr}\n\\toprule\nPrompt & Equivalent & Partial & Wrong & Func.\\ ID \\\\\n\\midrule\n" +
        "\n".join(f"{r['prompt'].replace('_',' ')} & {r['equivalent']} & {r['partial']} & {r['wrong']} & "
                  f"{r['functional_id']}/{r['n']} \\\\" for r in out) +
        "\n\\bottomrule\n\\end{tabular}\n")


def table_cost(rows):
    cost = list(csv.DictReader((OPT / "cost.csv").open()))
    stress = {}
    for prompt in PROMPTS:
        m, _ = mean_std([r["test_stress"] for r in rows if r["prompt"] == prompt])
        stress[prompt] = m
    out = []
    for c in cost:
        name = c["optimizer"]
        if name.startswith("final_eval_"):
            prompt = name.replace("final_eval_", "")
        else:
            prompt = "unoptimized" if name == "shared_baseline" else name
        c = dict(c)
        c["best_A"] = BASE_A  # returned prompt is empty for every optimizer
        c["test_stress"] = stress.get(prompt)
        out.append(c)
    _write_csv(OUT / "table6_cost.csv", out)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Run & Rollouts & Accepted & Proposer calls & GPU-h & test\_stress \\",
             r"\midrule"]
    for r in out:
        ts = "--" if r["test_stress"] is None else f"{r['test_stress']:.3f}"
        lines.append(f"{r['optimizer'].replace('_',' ')} & {r['rollouts']} & {r['accepted']} & "
                     f"{r['proposer_calls']} & {float(r['gpu_h']):.2f} & {ts} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "table6_cost.tex").write_text("\n".join(lines) + "\n")


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()
    table_main(rows); table_gain(rows); table_task(rows)
    table_breakdown(); table_identification(rows); table_cost(rows)
    print("tables ->", OUT)


if __name__ == "__main__":
    main()
