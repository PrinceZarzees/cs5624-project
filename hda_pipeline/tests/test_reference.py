"""Cross-check the C++ reference solvers against NetworkX on random small/medium graphs."""
import random
import pytest
from hda.tasks import TASKS, nx_answer
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
REF_TASKS = [t for t in TASKS if (ROOT / "reference" / f"{t}.cpp").exists()]
from hda.generator import (ALL_FAMILIES, DENSITY_BINS, make_instance, parse_text,
                           reference_answer, ensure_reference_binaries)

ensure_reference_binaries()

@pytest.mark.parametrize("task", REF_TASKS)
def test_reference_matches_networkx(task):
    rng = random.Random(task)
    for i in range(60):
        fam = ALL_FAMILIES[i % len(ALL_FAMILIES)]
        dens = list(DENSITY_BINS)[i % 3]
        n = rng.randint(2, 40) if i < 45 else rng.randint(100, 250)
        text, meta = make_instance(task, fam, n, dens, rng)
        n2, edges, q, d, w = parse_text(text)
        assert n2 == meta["n"]
        assert len(set((e[0], e[1]) if d else tuple(sorted(e[:2])) for e in edges)) == len(edges)
        assert all(e[0] != e[1] for e in edges)
        ref, _ = reference_answer(task, text)
        assert ref == nx_answer(task, n2, edges, q, d, w), (task, fam, n, text[:200])

def test_balance():
    rng = random.Random(1)
    for task in ["cycle_detection", "bipartite_check", "reachability"]:
        ys = [reference_answer(task, make_instance(task, "er", 30, "medium", rng)[0])[0] for _ in range(60)]
        frac = ys.count("YES") / len(ys)
        assert 0.25 < frac < 0.75, (task, frac)
