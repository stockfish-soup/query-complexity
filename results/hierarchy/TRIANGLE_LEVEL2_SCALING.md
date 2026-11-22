# Level-2 triangle finding, requested sweep through n=10

```bash
PYTHONPATH=. python experiments/experiment_triangle_level2_scaling.py
```

This is a resource-checked sweep, **not eight completed SDP solves**. It
solves n=3,4 and declines n=5,...,10 before constructing the oracle table.
`triangle_level2_scaling.json` records all rows, including explicit missing
objectives and reasons. No restricted graph promise is substituted for the
full-domain problem.

| n | Graph inputs | Level-2 value | Stored adversary lower bound | Elementary query upper bound |
| --- | ---: | ---: | ---: | ---: |
| 3 | 8 | 1.732051 (saturated level 1) | — | 3 |
| 4 | 64 | approximately 3 | 3.000000 | 6 |
| 5 | 1,024 | not run | 4.063267 | 10 |
| 6 | 32,768 | not run | 5.219912 | 15 |
| 7 | 2,097,152 | not run | 6.231963 | 21 |
| 8 | 268,435,456 | not run | 6.911788 | 28 |
| 9 | 68,719,476,736 | not run | — | 36 |
| 10 | 35,184,372,088,832 | not run | — | 45 |

Stored values come directly from `SERIES['best MOSEK known']` in
`julia/triangle/plot_adversary_evolution.py`. They are restricted-domain
adversary **lower bounds**, not algorithmic upper bounds or known full-domain
optima (except the matching small case). The n=9,10 numbers elsewhere in the
triangle README are extrapolations of a fitted growth law, not computed
benchmarks, and are intentionally excluded. The elementary upper bound is
the number of edges: reading every edge solves triangle finding exactly.
No asymptotic O(n^c) estimate is treated as a numerical finite-n bound.

A feasible level-2 objective upper-bounds the full adversary SDP, while a
restricted-domain adversary optimum lower-bounds it. For n=4 the values
match at approximately 3. The tiny signed difference from the rounded stored
value is within solver error. Inspect `solution.status` and reconstruction
residuals; this run returns `optimal_inaccurate` at n=4 and passes the existing
1e-6 numerical validation. This is not an exact certificate.

## Why the current backend stops

With N=2^binom(n,2) inputs, an edge stabilizer has order
H=2*(n-2)!. Its number m0 of graph orbits is at least ceil(N/H).
The backend explicitly stores the trivial block lift, requiring at least

```text
8 * N^2 * ceil(N/H)^2 bytes
```

in real arithmetic. This lower bound exceeds 62 GB already at n=5, before
higher sectors, decomposition matrices, constraints and solver memory.
It is a guaranteed excessive allocation, not a timed-out solve. The default
0.5 GiB guard is checked before `triangle_finding(n)` is called. It is not
a promise that total memory for an admitted problem stays below that budget.

Reaching n=10 needs a different structural backend: implicit representation
lifts, compressed graph/pair-orbit constraints, and a way to construct those
constraints without enumerating the full graph cube. Merely restricting to
three irrep types does not bound their multiplicities. The existing Julia
restricted-domain solver can study larger selected graph families, but those
results would be for a different domain and would not constitute the
requested full-domain level-2 test.
