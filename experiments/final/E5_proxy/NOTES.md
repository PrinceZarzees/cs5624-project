# E5 reduced proxy experiment

- Full runbook target: 60 candidates, proxy_large/proxy_medium splits, evolved
  rubric, 5-fold CV.
- Reduced run: 17 candidates (A, B, v2, 14 one-rule-removed v2 variants) with
  existing frozen `test_large` labels; no new GPU rollouts.
- The learned static proxy reaches Spearman 0.107 on large-graph labels, versus
  0.553 on small-graph val labels in the earlier R1 run. This is a negative
  transfer result for static text features.
- Target `proxy_large` split (seed 56241) and evolved-rubric experiment remain
  unimplemented; see RESULTS_INDEX.md.
