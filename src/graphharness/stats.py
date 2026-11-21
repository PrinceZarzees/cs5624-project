"""Small statistics helpers used by the final report."""
from __future__ import annotations
import math
import random
from collections import Counter
from typing import Iterable


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: Iterable[float], B: int = 10000, seed: int = 0) -> tuple[float, float]:
    vals = list(values)
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(B):
        sample = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        means.append(sum(sample) / len(sample))
    means.sort()
    return (means[int(0.025 * B)], means[int(0.975 * B) - 1])


def mcnemar(paired_a: list[bool], paired_b: list[bool]) -> dict:
    if len(paired_a) != len(paired_b):
        raise ValueError("paired vectors must have equal length")
    b = sum(a and not c for a, c in zip(paired_a, paired_b))
    c = sum(not a and c for a, c in zip(paired_a, paired_b))
    n = b + c
    if n == 0:
        p = 1.0
    else:
        # exact two-sided binomial test
        k = min(b, c)
        p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"a_only": b, "b_only": c, "p_value": p}


def aggregate_repeats(rows: list[dict]) -> list[dict]:
    """Aggregate long rows by (harness, split) with mean/std/min/max."""
    groups = {}
    for r in rows:
        groups.setdefault((r["harness"], r["split"]), []).append(float(r["accuracy"]))
    out = []
    for (harness, split), vals in sorted(groups.items()):
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals) if len(vals) > 1 else 0.0
        out.append({"harness": harness, "split": split, "repeats": len(vals),
                    "mean": mean, "std": math.sqrt(var), "min": min(vals), "max": max(vals)})
    return out
