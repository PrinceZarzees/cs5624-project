"""Aggregate run directories into report tables (CSV + LaTeX fragments).

    python -m hda.aggregate --runs runs/hda/main --data data/hda --out report/tables/hda
Produces:
  hda_main.csv          task x condition: val acc, test acc per split (mean +- std over seeds), gap
  hda_by_family.csv     test accuracy by graph family (seen vs unseen) and density bin
  hda_verdicts.csv      verdict mix (AC/WA/TLE/RE/MLE) per split
  hda_curves.csv        validation accuracy per iteration
  hda_identification.csv hidden-statement runs: judge label per iteration-final statement
  hda_constant_baseline.csv majority-answer baseline per task and split
"""
from __future__ import annotations
import argparse, csv, json, statistics
from collections import Counter, defaultdict
from pathlib import Path
from .evaluate import load_split

SPLITS = ["test_small", "test_medium", "test_large", "test_xl", "test_stress"]

def ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return ""
    return f"{100*statistics.mean(xs):.1f} ± {100*statistics.pstdev(xs):.1f}" if len(xs) > 1 else f"{100*xs[0]:.1f}"

def write(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(path.with_suffix(".tex"), "w") as f:
        f.write("\\begin{tabular}{" + "l" * 2 + "r" * (len(header) - 2) + "}\n\\toprule\n")
        f.write(" & ".join(h.replace("_", "\\_") for h in header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(x).replace("_", "\\_").replace("±", "$\\pm$") for x in r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--data", default="data/hda"); ap.add_argument("--out", required=True)
    a = ap.parse_args(); out = Path(a.out)
    finals = [json.load(open(p)) for p in Path(a.runs).glob("*/final.json")]
    g = defaultdict(list)
    for f in finals:
        g[(f["task"], f["condition"])].append(f)
    rows, curves, ident, verd, fam = [], [], [], defaultdict(Counter), defaultdict(list)
    for (t, c), fs in sorted(g.items()):
        acc = lambda sp: [f["test"].get(sp, {}).get("accuracy") for f in fs]
        gap = [f["test"]["test_small"]["accuracy"] - f["test"]["test_xl"]["accuracy"]
               for f in fs if "test_small" in f["test"] and "test_xl" in f["test"]]
        rows.append([t, c, len(fs), ms([f["best_val_acc"] for f in fs])] + [ms(acc(sp)) for sp in SPLITS] + [ms(gap)])
        for f in fs:
            curves.append([t, c, f["seed"]] + [round(x, 3) for x in f["val_curve"]])
            if c == "hidden":
                ident.append([t, f["seed"], f.get("statement_judge", ""), f["final_statement"][:300].replace("\n", " ")])
            for sp, s in f["test"].items():
                verd[(c, sp)].update(s["verdicts"])
                for k, v in s["by_family"].items():
                    fam[(c, sp, k)].append(v["acc"])
    write(out / "hda_main.csv", ["task", "condition", "runs", "val"] + SPLITS + ["gap_small_minus_xl"], rows)
    write(out / "hda_curves.csv", ["task", "condition", "seed"] + [f"it{i+1}" for i in range(max((len(c) - 3 for c in curves), default=0))], curves)
    write(out / "hda_identification.csv", ["task", "seed", "judge", "final_statement"], ident)
    vk = ["AC", "WA", "TLE", "RE", "MLE", "OLE"]
    write(out / "hda_verdicts.csv", ["condition", "split"] + vk,
          [[c, sp] + [verd[(c, sp)].get(k, 0) for k in vk] for (c, sp) in sorted(verd)])
    write(out / "hda_by_family.csv", ["condition", "split", "family", "acc"],
          [[c, sp, k, ms(v)] for (c, sp, k), v in sorted(fam.items())])
    # majority-answer baseline: most common example answer, applied to each test split
    data = Path(a.data); cb = []
    ex = load_split(data, "examples")
    tasks = sorted({r["task"] for r in ex})
    for t in tasks:
        maj = Counter(r["answer"] for r in ex if r["task"] == t).most_common(1)[0][0]
        row = [t, maj]
        for sp in SPLITS:
            if (data / f"{sp}.jsonl").exists():
                rs = load_split(data, sp, t); row.append(f"{100*sum(r['answer']==maj for r in rs)/len(rs):.1f}")
            else:
                row.append("")
        cb.append(row)
    write(out / "hda_constant_baseline.csv", ["task", "majority_answer"] + SPLITS, cb)
    print(f"{len(finals)} runs aggregated into {out}")

if __name__ == "__main__":
    main()
