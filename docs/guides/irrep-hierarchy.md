# Irrep-prefix SDP hierarchy

Build adversary SDPs for finite binary-oracle problems with a permutation
group action. The input specifies **the action and the oracle task**, not
just an abstract group: a group alone does not determine an SDP.

The backend handles arbitrary finite permutation actions, including any
`S_n` action that fits the explicit enumeration limit. The examples construct
`S_n` using adjacent transpositions. The decomposition is a numerical
Wedderburn decomposition inspired by the existing SymbolicWedderburn triangle
solver; this Python backend does **not** call SymbolicWedderburn.jl.

## Quick start

From the repository root, after `python -m pip install -e '.[dev]'`:

```python
from sdp_hierarchy import Hierarchy, symmetric_evaluation, symmetric_junta

# Evaluate three-bit majority using coordinate queries.
problem = symmetric_evaluation(3, [0, 0, 1, 1])
hierarchy = Hierarchy(problem)
print(hierarchy.block_summary())
for r in range(hierarchy.max_level + 1):
    solution = hierarchy.solve(r)
    print(r, solution['cost'], solution['numerically_feasible'])
# 0: 3; 1: 2

# Identify the hidden support of a three-variable majority in five coordinates.
learning = Hierarchy(symmetric_junta(5, 3, [0, 0, 1, 1]))
print(learning.solve(0)['cost'])
```

For a symmetric Boolean function, `profile[a]` is its output at weight `a`.
The two builders distinguish Boolean **evaluation** from hidden-support
**identification**. Both are tested; they have different adversary problems.

## SDP and normalization

Let `C` be the concept/input set, `Q` the query set, `a(c,q)` the oracle
answer, and `t(c)` the desired target. Solve

```text
minimize tau
X_q >= 0
sum_q X_q[c,c] <= tau                                    for every c
sum_{q: a(c,q) != a(d,q)} X_q[c,d] = 1                    if t(c) != t(d).
```

The objective is the standard filtered adversary cost, with no extra `1/2`.
In particular OR on `n` bits has value `sqrt(n)`. The older
`query_complexity.py` phase-normalized objective is half this value.
Different concepts may have the same target. Identification defaults to a
different target per concept. Indistinguishable oracles with different
targets are rejected before solving.

The group must preserve the answer table under simultaneous concept/query
permutations and preserve target equivalence. Group averaging therefore
preserves feasibility and cannot increase the maximum load. This proves
that the full covariant SDP has the same optimum as the unrestricted SDP.

## Query stabilizers, rather than the full group's irreps

For each query orbit select a representative `q` and its stabilizer
`H_q={g:gq=q}`. Covariance reduces all matrices in that orbit to one
`H_q`-invariant representative. Decompose its concept permutation module:

```text
C^C = direct_sum_lambda (V_lambda tensor C^m_lambda)
X_q = direct_sum_lambda (I_d_lambda tensor B_q,lambda),    B_q,lambda >= 0.
```

Thus the PSD variable size is the **multiplicity** `m_lambda`, not the
isotypic dimension `d_lambda*m_lambda`. Entire equivalent copies are retained
together. Decomposing the full group's concept module and keeping its
trivial irrep would generally fail to give the intersection-free model.

The implementation obtains the isotypic projectors from joint eigenspaces
of central conjugacy-class sums. Within a sector, a generic Hermitian
group-algebra element selects a multiplicity space `U`. The lift is

```text
L_lambda(B) = (d_lambda / |H_q|) sum_h R(h) U B U* R(h)*.
```

Schur averaging makes this equivalent to `I_d tensor B`.
Consequently `trace(L_lambda(B)) = d_lambda*trace(B)`; omitting that factor
would change the objective. Tests check this normalization, PSD, covariance,
and the repeated eigenvalues for a representation with multiplicities > 1.
Complex conjugate irreps are separated, using complex Hermitian variables
where necessary (a stabilizer of an `S_n` action need not itself be `S_m`).

## Levels and ordering

There is no canonical notion of the “first” irreps without choosing an order.
Each stabilizer has its own list of irreducible types **present** in its
concept module:

1. Trivial type first, including its entire multiplicity space.
2. Remaining types sorted lexicographically by their central class-sum
   eigenvalue signatures. These deterministic numerical labels are reported
   by `block_summary()`; they are not Young-diagram/partition names.

**Level `r` keeps the first `r+1` types in each stabilizer list.** The shift
makes level 0 the trivial sector. Shorter lists saturate. The last level is
`max_q(number_of_present_types(q)-1)`. This is a prefix hierarchy, not a
polynomial-degree or “at most r boxes” hierarchy. Abstract irreps of zero
multiplicity create no variable and are omitted.

Supply `sector_order=lambda sector: ...` to order the nontrivial types
otherwise; the trivial sector stays first. Changing the order changes
intermediate levels, while the first and final feasible sets stay fixed.
For example, `sector_order=lambda s: (s.degree, s.label)` prioritizes smaller
irrep degrees, breaking ties by signature.

Adding a type leaves the old feasible solution available by setting the new
PSD block to zero. Hence the optima are nonincreasing. The final prefix
contains every invariant PSD matrix, so its value equals the full SDP.
An infeasible prefix has value `+infinity`; solver failure is not evidence
of mathematical infeasibility.

## Why level 0 is intersection-free

The trivial stabilizer space is spanned by normalized indicators of the
`H_q`-orbits on concepts. If `F_q` is that indicator matrix, level 0 is exactly

```text
X_q = F_q B_q F_q*,     B_q >= 0.
```

Entries depend only on the two **individual** local concept-orbit labels,
with no further pair-intersection parameter. Conversely every PSD matrix
of this intersection-free form is supported in the trivial stabilizer
sector. This proves equality of the two feasible sets, not just equality
of their measured objective values.

For hidden `k`-subsets and a subset query `S`, the stabilizer is
`S_|S| x S_(n-|S|)` and the individual labels are `|A intersect S|`.
Thus level 0 is the count-based intersection-free SDP with a separate
channel per query-size orbit. For symmetric-function evaluation and a
coordinate query `i`, the labels are `(x_i, |x outside i|)`.

The previously imported `junta_learning.majority_sdp` additionally ties the
query-size channels via one common, unbiased count matrix. Its witness
embeds in this level 0; it is a further restriction, not in general the
same numerical optimization. Finite universes also omit unrealizable
Johnson distances (`d <= min(k,n-k)`), whereas the old universe-independent
program enforces every `d=1,...,k`. Tests verify the old witness's embedding.

## Use another permutation action

```python
from sdp_hierarchy import OracleProblem, Hierarchy

# Two concepts, two queries, with an S2 symmetry.
problem = OracleProblem(
    answers=[[0, 1], [1, 0]],
    generators=[([1, 0], [1, 0])],
    targets=[0, 1],
)
solution = Hierarchy(problem).solve(0)
```

Each generator is `(concept_images, query_images)`, using zero-based images.
Pass a binary answer table of shape `(number_of_concepts, number_of_queries)`.
A backend extension point `decomposer=callable` accepts a full stabilizer's
concept permutations and returns `representation.Sector` objects, trivial
first. Each contains the irrep degree, multiplicity, orthonormal isotypic
basis, and column-major multiplicity lift. This interface separates
representation construction from oracle constraints and prefix selection.
It does not itself supply a Julia or partition-based backend.

## Validation and reproduced examples

```bash
python -m pytest tests/test_sdp_hierarchy.py -q
python experiments/experiment_irrep_hierarchy.py
```

The script saves block metadata and every level's costs, solver statuses,
residuals, eigenvalue checks and support leakage to
`results/hierarchy/symmetric_functions.json`.

| Task | Level 0 | Final level | Independent dense SDP |
| --- | ---: | ---: | ---: |
| OR, 3 bits | 1.732051 | 1.732051 | 1.732051 |
| Majority, 3 bits | 3.000000 | 2.000000 | 2.000000 |
| EXACT-1, 3 bits | 3.000000 | 2.645751 | 2.645751 |
| Parity, 3 bits | 3.309401 | 3.000000 | 3.000000 |
| OR, 4 bits | 2.000000 | 2.000000 | 2.000000 |
| OR support learning, n=4, k=2 | 2.000000 | 2.000000 | 2.000000 |
| Majority support learning, n=5, k=3 | 2.121320 | 2.121320 | 2.121320 |

Numerical monotonicity is checked with tolerance. The first evaluation
examples have two types; the learning examples have four. Extra levels need
not strictly improve the objective.

`solve(intersection_free=True)` independently builds orbit indicators and
solves level 0 without the representation lifts. `solve(dense=True)` instead
uses full PSD matrices with explicit stabilizer-invariance equations. Tests
also compare a final level with a completely unreduced SDP with no covariance
restriction. Constant functions, invalid actions, complex stabilizer types,
and enumeration limits are covered.

## Scope and numerical limitations

This is a general **finite, explicit reference pipeline**, not a scalable
implementation for arbitrary large symmetric groups. It enumerates the
combined concept/query group (default cap `max_order=10000`), queries and
concepts, and constructs dense class-sum and lift arrays. Factorial group
order, exponential oracle tables, and SDP size limit practical examples.
Large cases require combinatorial/partition-based representation backends
and compressed constraints, like those used in the triangle project.

Decomposition and solves use floating-point tolerances. Clarabel is the
default, with SCS fallback on a solver error; the actual solver and status
are recorded. `numerically_feasible` checks reconstructed equality, PSD,
load, covariance and prefix-support residuals at `validation_tol` (default
`1e-6`). It is not an exact or interval certificate. Boundary-degenerate
SDPs may need solver tuning or further facial reduction; inspect residuals
and statuses rather than treating a returned objective as a proof.
