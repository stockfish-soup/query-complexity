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

See [the definition and ordering conventions](../../docs/guides/irrep-hierarchy.md).
Each level r includes r+1 present types, starting with the trivial query
stabilizer representation. The backend enumerates finite group actions;
these results make no scalability claim for large symmetric groups.

## Triangle finding

```bash
python experiments/experiment_triangle_hierarchy.py
```

`triangle_finding.json` tests the full graph domain, using vertex `S_n`
symmetry and edge-bit queries. Three vertices give 8 inputs/3 queries and
four vertices give 64 inputs/6 queries.

| Vertices | Level 0 | Level 1 | Level 2 | Level 3 | Dense SDP |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 1.732051 | 1.732051 | — | — | 1.732051 |
| 4 | infeasible | infeasible | 3.000000 | 3.000000 | 2.999999 |

For four vertices the edge stabilizer has four present types, with
multiplicities 28, 12, 12, 12. Thus level 0 uses a 28-dimensional PSD block;
the final level uses blocks of sizes 28,12,12,12 instead of a dense 64 by 64
matrix. This is the stabilizer of an edge, not the stabilizer used in the
subset-learning examples.

Both Clarabel and SCS report infeasibility for levels 0 and 1. These are
numerical solver reports, not independently verified exact dual certificates.
Level 2 and the dense reference return `optimal_inaccurate`; both pass the
recorded `1e-6` reconstruction checks. The terminal level uses SCS after a
Clarabel error, returns `optimal`, and agrees at approximately 3. Infeasible
objectives are stored as JSON `null`, with the status retained.

The earlier symmetric-function table remains in `symmetric_functions.json`;
these are additional tests. Larger full graph domains grow as
`2^binom(n,2)` and were not benchmarked here.
