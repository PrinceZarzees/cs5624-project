from graphharness.stats import aggregate_repeats, bootstrap_ci, mcnemar, wilson_ci

def test_wilson():
    lo, hi = wilson_ci(8, 10)
    assert 0 <= lo < 0.8 < hi <= 1

def test_bootstrap():
    lo, hi = bootstrap_ci([0, 1, 1, 1])
    assert 0 <= lo <= hi <= 1

def test_mcnemar():
    r = mcnemar([True, True, False], [True, False, True])
    assert r["a_only"] == 1 and r["b_only"] == 1 and 0 <= r["p_value"] <= 1

def test_aggregate():
    rows = [{"harness": "a", "split": "x", "accuracy": 0.5},
            {"harness": "a", "split": "x", "accuracy": 1.0}]
    out = aggregate_repeats(rows)
    assert out[0]["repeats"] == 2 and out[0]["mean"] == 0.75
