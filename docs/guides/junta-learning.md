# Intersection-free SDP optimization for junta learning

These routines learn a hidden support `A` from subset queries `S` with answer
`f_A(S)=h(|A intersect S|)`. Majority uses `h(a)=1[a >= (k+1)/2]` for a
known odd support size `k`. The objective is the learning dual-adversary cost
`max_A sum_S X_S[A,A]`, with `X_S` positive semidefinite and

```text
sum_{S: f_A(S) != f_B(S)} X_S[A,B] = 1       for every A != B.
```

This objective uses the learning SDP convention; do not apply the Python
Boolean-evaluation module's `Adv/2` normalization to the values here.
The intersection-free ansatz depends on local counts, with no additional
`|A intersect B intersect S|` parameter. It is a restriction of the learning
SDP, so its feasible objectives give upper bounds.

## The optimization that led to logarithmic majority cost

The numerical discovery proceeded from the convex count SDP to a nonconvex
rank-one restriction, then to a finite search over Fourier half-sets. The
period-four half-set produced the explicit witness whose cost is proved
`O(log(k+1))`. The SDP is a benchmark; its optimizer is generally not rank
one. The observed leading coefficient `4/pi` is a numerical asymptotic,
not a certified asymptotic for the SDP optimum.

For fixed odd `k`, write `pi[a]=binom(k,a)/2^k`. At Johnson distance
`d=|A minus B|`, construct

```text
T_d[a,b] = Pr(|A intersect S|=a, |B intersect S|=b) / sqrt(pi[a] pi[b])
```

for a uniformly random query `S`. With `m=(k-1)/2`, the reduced SDP is

```text
minimize trace(Z),  Z >= 0
subject to 2 sum_{a<=m, b>m} T_d[a,b] Z[a,b] = 1,  d=1,...,k.
```

Equivalently optimize the cross-cut block `Q` with objective `2 ||Q||_*`
and the same equalities. `majority_sdp.solve(k)` returns `(cost, Z, status)`;
`solve_cross_block(k)` returns `(cost, Q, status)`. Both use Clarabel with an
SCS fallback. The reduction assumes all distances `1,...,k` are possible
(e.g. ambient universe size `n >= 2k`); feasible witnesses also apply in
smaller universes with fewer available distances.

From the repository root, after `python -m pip install -e '.[dev]'`:

```bash
python -m junta_learning.majority_sdp --ks 1 3 5 7
python -m junta_learning.rank_one --ks 1 3 5 7 15 31 63 --exact
python -m junta_learning.rank_one --ks 3 5 7 --exhaustive
python -m junta_learning.compare_rank_one --ks 3 5 7
```

`rank_one.exhaustive_best(k)` searches `binom(k-1,(k-1)/2)` half-sets;
its CLI skips exhaustive searches above 17. Exact rational arithmetic is
used for the period-four cost squared. This exact evaluation of the formula
is distinct from a floating-point solver certificate.

## Recover the explicit rank-one matrices

```python
import numpy as np
from junta_learning import rank_one

k = 7
z = rank_one.count_witness(k)
Z = np.outer(z, z)
print("rank-one cost:", z @ z)
print("exact cost squared:", rank_one.exact_cost_squared(k))
```

For any query `S` and hidden supports `A,B`, put `a=|A intersect S|` and
`b=|B intersect S|`. The full witness is

```text
X_S[A,B] = 2^(-n) z[a] z[b] / sqrt(pi[a] pi[b]).
```

Each `X_S` is PSD of rank at most one; its cost is `||z||^2`. No enumeration
of the ambient `2^n` queries is needed to construct the compact vector.
Tests independently enumerate all concepts and queries for small sizes to
check every original learning equality and diagonal load.

## Independent witness and spectrum audit

`witness_audit` retains the earlier dyadic-current construction, direct
joint-count spectrum checks, original-SDP enumeration, convex optimization,
and optional exact SymPy verification. Its nearest-neighbor construction is
an exploratory alternative; its measured logarithmic growth is not used as
a proof. The later period-four proof is in the notes linked below.

```bash
python -m pip install -e '.[symbolic]'
python -m junta_learning.witness_audit --k 1 3 5 7
```

Without SymPy, disable exact symbolic checks with `--exact-max-k 0`.
The other audits use NumPy, SciPy and CVXPY.

## Unknown support size at most k

`bounded_size` uses a single global PSD matrix indexed by `(r,a)`, where
`1 <= r <= k` and `0 <= a <= r`. It minimizes the maximum binomially weighted
diagonal load across layers and enforces every mixed-size overlap equality.
This retains PSD consistency between cardinality layers rather than solving
each size separately. Even sizes use **tie-positive majority**.

```bash
python -m junta_learning.bounded_size --ks 1 2 3 --details
python -m junta_learning.bounded_size --ks 3 5 --odd-only --details
```

The global PSD dimension is `sum_{r=1}^k (r+1)`; use small cutoffs for numerical
optimization. These finite solves benchmark the bounded-size ansatz. The
analytic `O(log k)` construction and even-to-odd reduction are explained in
[the bounded-size notes](../junta/bounded_size_log_cost.md); numerical
fits alone do not establish that theorem.

## General symmetric juntas and multiple biases

`multibias` optimizes arbitrary symmetric profiles through finite-bias
nuclear-norm SDPs and has specialized EXACT/OR SOCP and exchange routines.
For majority at one unbiased channel, it reproduces the fixed-size SDP above:

```python
from junta_learning import multibias

k = 7
profile = [int(a >= (k+1)//2) for a in range(k+1)]
result = multibias.solve_general_level0_cvxpy(k, profile, [0.5], solver="CLARABEL")
print(result["value"])
```

```bash
python -m junta_learning.multibias exact-half --k 12
python -m junta_learning.multibias belovs-or --k 12
```

See [multi-bias derivation and commands](../junta/multibias.md).
The exchange routine's continuum envelope is scanned and locally refined;
its reported continuum bracket is numerical, not a rigorous global certificate.

## Reproduce the comparison table

```bash
python experiments/experiment_majority_junta.py
```

This writes `results/junta/majority_comparison.csv`, comparing numerical convex
solutions (through `k=15` by default) with explicit rank-one costs and directly
counted feasibility residuals (through `k=63`). Use `--ks`,
`--optimize-max-k`, and `--output` to control the sweep. Blank SDP columns
mean the convex solve was skipped. Inspect solver status and residuals; the
archived `k=15` solve is `optimal_inaccurate` (see the results README). The rank-one column is the explicit
feasible cost, not a claim of global rank-one optimality.

## Proof notes and provenance

- [Which optimization produced the rank-one witness?](../junta/majority_rank_one_optimization_origin.tex)
- [Direct rank-one construction and logarithmic bound](../junta/majority_rank_one_non_h.tex)
- [Existing closed-form majority PDF](../../majority_rank_one_closed_X.pdf)

Imported from the workspace's `proof/latex/majority` (direct count SDP,
rank-one search/comparison, numerical audit), `proof/latex/at_most_k`
(mixed-size SDP), `code/junta` (multi-bias optimizer), and
`transducer/majority_leq_k_log_cost.md` (bounded-size construction).
Original checkouts and proof sources remain untouched. Module names and
imports were adapted to the installable `junta_learning` package, SymPy
loading was made optional, invalid majority sizes are rejected at optimizer
entry points, and `rank_one.count_witness` exposes the compact PSD factor.
