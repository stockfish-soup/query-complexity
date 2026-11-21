# Irrep-prefix hierarchy benchmarks

Reproduce from the repository root after an editable installation:

```bash
python experiments/experiment_irrep_hierarchy.py
```

`symmetric_functions.json` records each example's group order, query orbits,
isotypic degrees and multiplicities, central-character labels, and every
prefix's objective and witness diagnostics. It also includes independently
constructed intersection-free and dense covariant reference SDPs.

All recorded solves returned `optimal` and passed the `1e-6` numerical
feasibility checks. Across these solves, the largest equality residual was
`4.4e-12`, the minimum reconstructed eigenvalue was approximately `-1.93e-8`,
and prefix-support leakage stayed below `3.8e-15`. These are floating-point
measurements, not exact feasibility certificates.

On three-bit majority, the hierarchy improves from `3` to `2`; on EXACT-1
it improves from `3` to `sqrt(7)`. The terminal values match the independent
dense SDPs. The tests additionally compare one terminal problem with a fully
unreduced SDP, and verify that the earlier unbiased majority witness lies
in level 0 of the finite-universe learning hierarchy.

See [the definition and ordering conventions](../../sdp_hierarchy/README.md).
Each level r includes r+1 present types, starting with the trivial query
stabilizer representation. The backend enumerates finite group actions;
these results make no scalability claim for large symmetric groups.
