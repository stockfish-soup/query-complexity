# Majority junta comparison

Reproduce from the repository root after an editable installation:

```bash
python experiments/experiment_majority_junta.py
```

`majority_comparison.csv` compares the explicit period-four rank-one witness
with numerical solutions of the unrestricted count-space PSD ansatz.
Costs use the learning-adversary normalization documented in
[the junta guide](../../junta_learning/README.md).

- `rank_one_cost` is evaluated from the exact rational squared-cost formula.
- `rank_one_residual` checks all count-kernel equalities with floating arithmetic.
- `sdp_*` columns record the solver value, status, equality residual and
  minimum eigenvalue. Empty entries mean the solve was skipped.
- In this run `k=15` is `optimal_inaccurate`, with residual about `6.5e-7`
  and minimum eigenvalue about `-1.1e-6`. Treat that entry as an approximate
  numerical benchmark, not a certified optimum or feasible certificate.

The explicit witness residual stays below `1e-14` through `k=63` in this
run. Neither this finite table nor the observed `4/pi` slope proves an
asymptotic result; the included proof notes establish the logarithmic bound.
