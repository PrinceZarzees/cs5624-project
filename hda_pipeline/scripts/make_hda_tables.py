#!/usr/bin/env python3
"""Regenerate report/tables/hda/*.{csv,tex} from HDA outputs (CPU only)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # repo root
PIPE = ROOT / "hda_pipeline"
sys.path.insert(0, str(PIPE))
from hda.tasks import TASKS  # noqa: E402

OUT = ROOT / "report/tables/hda"
OUT.mkdir(parents=True, exist_ok=True)


def esc(x):
    s = str(x)
    for ch, rep in [("%", "\\%"), ("&", "\\&"), ("_", "\\_"), ("#", "\\#")]:
        s = s.replace(ch, rep)
    return s


def tex(name, header, rows, align=None):
    align = align or ("l" + "r" * (len(header) - 1))
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(esc(h) for h in header) + r" \\", "\\midrule"]
    for r in rows:
        lines.append(" & ".join(esc(x) for x in r) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def csvw(name, header, rows):
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)


def main():
    # tasks
    rows = []
    for t in TASKS.values():
        q = "none" if t.n_query == 0 else "2 (s,t)" if t.n_query == 2 else str(t.n_query)
        rows.append([t.name, "yes" if t.directed else "no", "yes" if t.weighted else "no",
                     q, t.statement])
    header = ["task", "directed", "weighted", "query args", "statement"]
    csvw("tab_hda_tasks", header, rows)
    tex("tab_hda_tasks", header, rows, align="llllp{6.2cm}")

    # splits
    cfg = json.loads((PIPE / "configs/splits.json").read_text())
    rows = []
    for name, s in cfg["splits"].items():
        rows.append([name, f"{s['n_range'][0]}-{s['n_range'][1]}", s["per_task"],
                     s["families"], f"{s['time_limit_s']:g}"])
    header = ["split", "nodes", "n/task", "fam.", "limit (s)"]
    csvw("tab_hda_splits", header, rows)
    tex("tab_hda_splits", header, rows, align="llrrlr")

    # identifiability
    idp = ROOT / "experiments/hda/identifiability.json"
    if idp.exists():
        d = json.loads(idp.read_text())
        rows = []
        for task, v in d.items():
            conf = v.get("confusers_k=3") or {}
            top = ", ".join(list(conf)[:3]) if conf else "--"
            rows.append([task, v["n_candidates"],
                         f"{v['k=1']:.2f}", f"{v['k=3']:.2f}",
                         f"{v['k=5']:.2f}", f"{v['k=10']:.2f}", top])
        header = ["task", "candidates", "k=1", "k=3", "k=5", "k=10", "top confusers at k=3"]
        csvw("tab_hda_identifiability", header, rows)
        tex("tab_hda_identifiability", header, rows, align="lrrrrrp{4.5cm}")

    # human reference
    hr = ROOT / "experiments/hda/human_reference/human_reference.json"
    if hr.exists():
        d = json.loads(hr.read_text())
        splits = ["val", "test_small", "test_medium", "test_large", "test_xl", "test_stress"]
        rows = []
        for task, v in d.items():
            acc = [f"{v[s]['accuracy']:.0%}" if s in v else "--" for s in splits]
            cpu = [f"{v[s]['max_cpu_s']:.2f}" if s in v else "--" for s in splits]
            rows.append([task] + acc + cpu)
        header = ["task"] + [f"acc {s}" for s in splits] + [f"cpu {s}" for s in splits]
        csvw("tab_hda_human", header, rows)
        tex("tab_hda_human", header, rows,
            align="l" + "r" * (2 * len(splits)))

    print("wrote", OUT)


if __name__ == "__main__":
    main()
