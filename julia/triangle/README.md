# Triangle adversary SDP

Run the commands below from `julia/triangle`. The environment is in its
parent directory; stored results and default output files are in `results/`.
Historical result filenames in these notes refer to that directory.


This directory contains a compact research implementation for lower-bounding
the quantum query complexity of triangle finding with the general adversary
bound. The code focuses on the experiments that were useful for `n=6` and
`n=7`: symmetry-reduced dual SDPs on selected graph orbits, orbit storage,
orbit pruning, greedy orbit addition, and diagnostics for important inputs.

## Files

- `triangle_common.jl`: graph encoding, the triangle predicate, and generic
  orbit routines.
- `triangle_adversary_dual_restricted.jl`: the symmetry-reduced adversary dual
  SDP on a selected union of graph orbits.
- `triangle_cb_degree_sdp.jl`: equation (18) of Gribling--Laurent
  (`arXiv:1901.04921v1`), specialized to triangle finding and reduced with
  SymbolicWedderburn.
- `graph_collision_adversary_dual.jl`: the full general-adversary dual for
  graph collision on a fixed host graph, including a star-graph CLI.
- `run_graph_collision_hosts.jl`: random or exhaustive comparison of
  nonisomorphic host graphs at a fixed number of vertices.
- `run_cs_graph_collision.jl`: checkpointed exact graph-collision runs for the
  square-difference host `CS(n)`.
- `plot_cs_graph_collision.py`: growth and `Adv±/sqrt(n)` plot for those runs.
- `graph_collision_n5_results.toml`: exhaustive MOSEK ranking of all 33
  nonempty unlabeled host graphs on five vertices.
- `graph_collision_n6_results.toml`: exhaustive MOSEK ranking of all 155
  nonempty unlabeled host graphs on six vertices.
- `run_triangle_symmetric.jl`: command-line runner plus orbit selection,
  storage, pruning, greedy search, diagnostics, and the n=7 drawing helper.
- `triangle_orbit_results.toml`: stored useful orbit sets and run history.
- `n7_important_orbits.svg`: drawing of the most important n=7 q=500 orbits.

Removed side files: the old primal, old full-domain dual runner, Shapley
diagnostics, and wrapper scripts. Their useful functionality now lives in
`run_triangle_symmetric.jl`.

## Setup

```bash
julia --project=.. -e 'using Pkg; Pkg.instantiate()'
```

MOSEK is the intended solver. The code still imports Clarabel in the solver
file because earlier experiments used it, but the compact runner uses MOSEK by
default.

## Main commands

Solve the automorphism-threshold restricted dual:

```bash
julia --project=.. run_triangle_symmetric.jl 6 180
julia --project=.. run_triangle_symmetric.jl 7 500
```

Solve an explicit set of graph-orbit representatives:

```bash
julia --project=.. run_triangle_symmetric.jl solve 6 1 32 36 62 64 122 239 255 511 617 633 685 692 921 1881 2013 4061 32768
```

Solve the currently stored useful set for `(n,q)`:

```bash
julia --project=.. run_triangle_symmetric.jl solve 6 --stored 120
```

Backward prune an orbit set while preserving a lower-bound target:

```bash
julia --project=.. run_triangle_symmetric.jl prune 6 120 0.001 4 0.20
```

Forward greedy search for useful orbits:

```bash
julia --project=.. run_triangle_symmetric.jl forward 6 120 5 12 1e-5 true
```

Solve the symmetry-reduced primal and print the `beta` mass by input orbit:

```bash
julia --project=.. run_triangle_symmetric.jl primal 6 50
julia --project=.. run_triangle_symmetric.jl primal 7 50
```

Regenerate the n=7 orbit drawing:

```bash
julia --project=.. run_triangle_symmetric.jl draw-n7
```

Solve the bipartite-plus-one restriction and optionally compare to all graph
orbits:

```bash
julia --project=.. run_triangle_symmetric.jl bipartite-plus-one 5 true
```

Complete-bipartite hosts only, plus one same-side edge:

```bash
julia --project=.. run_triangle_symmetric.jl complete-bipartite-plus-one 6 2 false
```

Greedy families sorted by the number of triangle-completing missing edges:

```bash
julia --project=.. run_triangle_symmetric.jl almost-triangle-forward 6 8 0 0
```

Solve the completely bounded degree feasibility SDP (18) at tensor order `t`:

```bash
julia --project=.. triangle_cb_degree_sdp.jl 3 2 0.3333333333333333 clarabel
```

Search incrementally from `t=0` through `t_max`, stopping at the first
feasible order:

```bash
julia --project=.. triangle_cb_degree_sdp.jl search 3 6 0.3333333333333333 clarabel
```

An optional final argument changes the starting order. For example, this
searches `t=2:6`:

```bash
julia --project=.. triangle_cb_degree_sdp.jl search 3 6 0.3333333333333333 clarabel 2
```

The arguments are `nv`, `t`, optional `epsilon` (default `1/3`), and optional
solver (default `mosek`). From Julia, use
`triangle_cb_degree_feasibility(nv, t; ...)` for one order or
`triangle_cb_degree(nv; t_max=..., ...)` to test successive orders. The latter
also reports `ceil(t/2)`, the query complexity given by Theorem 4.1 when the
first feasible order is found.

For this SDP, the code always selects `s=floor(t/2)`. Thus, with
`m=binomial(nv,2)` edge variables, the unreduced PSD matrix has the smallest
allowed size

```text
(m+1)^floor(t/2) + (m+1)^ceil(t/2).
```

Vertex relabelings act simultaneously on the tensor coefficients, graph-input
constraints, and the two tuple index sets of the PSD slack. The tensor and
input constraints are replaced by orbit representatives, while
SymbolicWedderburn turns the invariant PSD slack into smaller PSD blocks. The
auxiliary `A_s^*(y)` variables from (18) are eliminated exactly: equations (7)
and (8) form equality components of Gram entries, and membership in
`range(A_s^*)` is the corresponding zero-sum condition on every component.
For even `t`, the implementation additionally averages over tensor reversal,
which exchanges the two equal tuple halves and typically cuts the reduced
model roughly in half. Incremental search releases each infeasible JuMP model
and all construction-only orbit tables before proceeding to the next order.

## Graph collision adversary SDP

For a fixed host graph `G`, an input marks a subset of its vertices. Graph
collision returns one exactly when some host edge has both endpoints marked.
Run the symmetry-reduced adversary SDP for the star with `nv` vertices using:

```bash
julia --project=.. graph_collision_adversary_dual.jl star 6 clarabel
```

The final argument is optional and defaults to `mosek`. From Julia, the star
wrapper is:

```julia
include("graph_collision_adversary_dual.jl")
result = star_graph_collision_adversary(6; solver=:clarabel)
```

For another fixed host graph, supply its edges and automorphism generators:

```julia
result = graph_collision_adversary_dual(
    nv,
    host_edges,
    automorphism_generators;
    solver=:mosek,
)
```

The implementation uses every one of the `2^nv` marked-vertex inputs. Host
automorphisms reduce input and opposite-output-pair constraints to orbit
representatives. Since a host automorphism group need not be transitive on
vertices, the code creates one canonical adversary matrix for each query
orbit. Each canonical matrix is reduced with SymbolicWedderburn under the
stabilizer of its query, and the remaining query matrices are transported from
their orbit representative. For a star these query orbits are the center and
the leaves.

Reference Clarabel results for stars are:

| star vertices | reduced adversary value |
| ---: | ---: |
| 2 | `1.4142135620` |
| 3 | `1.7320508084` |
| 4 | `2.0000000229` |
| 5 | `2.2360680291` |
| 6 | `2.4494903017` |

These agree numerically with `sqrt(nv)`. For `nv=2,3,4`, the reduced values
were also checked against independently constructed dense, unreduced
adversary SDPs.

### Comparing host graphs

Run a reproducible random sample of five-vertex host-graph isomorphism classes:

```bash
julia --project=.. run_graph_collision_hosts.jl random 5 10 1 mosek
```

The arguments after `random` are `nv`, sample count, random seed, optional
solver, and optional output TOML path. To exhaust every nonempty unlabeled host
graph, run:

```bash
julia --project=.. run_graph_collision_hosts.jl all 5 mosek graph_collision_n5_results.toml
```

There are 34 unlabeled simple graphs on five vertices, one of which is empty,
so the exhaustive run solves 33 SDPs. The largest values found are:

| rank | host graph | edges | degree sequence | adversary value |
| ---: | --- | ---: | --- | ---: |
| 1 | `K5` | 10 | `[4,4,4,4,4]` | `2.8284271217` |
| 2 | `K5` minus one edge | 9 | `[4,4,4,3,3]` | `2.7151945915` |
| 3 | representative `255` | 8 | `[4,4,3,3,2]` | `2.6737133745` |
| 4 | representative `223` | 7 | `[4,3,3,2,2]` | `2.6218138368` |
| 5 | representative `207` | 6 | `[4,2,2,2,2]` | `2.6204951071` |

Thus `K5` is the maximizer over all five-vertex host graphs, with value
numerically equal to `sqrt(8)`. The complete ranking, edge lists, query orbits,
block sizes, and solver statuses are stored in
`graph_collision_n5_results.toml`.

For six vertices, the analogous exhaustive command is:

```bash
julia --project=.. run_graph_collision_hosts.jl all 6 mosek graph_collision_n6_results.toml
```

There are 156 unlabeled simple graphs on six vertices, hence 155 nonempty
hosts. All 155 SDPs returned `OPTIMAL`. The largest values are:

| rank | host graph | edges | degree sequence | adversary value |
| ---: | --- | ---: | --- | ---: |
| 1 | `K6` | 15 | `[5,5,5,5,5,5]` | `3.1622776684` |
| 2 | `K6` minus one edge | 14 | `[5,5,5,5,4,4]` | `3.0763780668` |
| 3 | representative `8191` | 13 | `[5,5,5,4,4,3]` | `3.0523095243` |
| 4 | representative `6655` | 11 | `[5,5,3,3,3,3]` | `3.0363222085` |
| 5 | representative `7167` | 12 | `[5,5,4,4,3,3]` | `3.0191250420` |

Thus `K6` is the maximizer over all six-vertex host graphs, with value
numerically equal to `sqrt(10)`. Full details are stored in
`graph_collision_n6_results.toml`.

### Square-difference hosts CS(n)

The paper's graph `CS(n)` has vertices `0:n-1`, with an edge when the absolute
difference is a square at most `n/2`. The implementation shifts labels to
`1:n` and uses reflection `u -> n+1-u`, an automorphism of order two. Run and
checkpoint the exact full-domain adversary SDPs with:

```bash
julia --project=.. run_cs_graph_collision.jl 2 9 mosek false none cs_graph_collision_values.csv true
python3 plot_cs_graph_collision.py cs_graph_collision_values.csv cs_graph_collision_growth.svg
```

The first two arguments are the inclusive range of `n`. The CSV is flushed
after every completed instance, and the last argument enables resume mode.
The measured MOSEK values are:

| n | square offsets | Adv± | Adv± / sqrt(n) | input orbits | pair orbits | max block |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | `1` | `1.41421355` | `1.000000` | 3 | 2 | 4 |
| 3 | `1` | `1.73205082` | `1.000000` | 6 | 9 | 8 |
| 4 | `1` | `2.07136226` | `1.035681` | 10 | 34 | 16 |
| 5 | `1` | `2.38254543` | `1.065507` | 20 | 131 | 32 |
| 6 | `1` | `2.63273061` | `1.074808` | 36 | 459 | 64 |
| 7 | `1` | `2.87346933` | `1.086069` | 72 | 1630 | 128 |
| 8 | `1,4` | `3.28862747` | `1.162705` | 136 | 4079 | 256 |
| 9 | `1,4` | `3.49395610` | `1.164652` | 272 | 13055 | 512 |

For `n <= 7`, `CS(n)` is just the path because only square offset one is
allowed. Offset four first appears at `n=8`, producing a discrete jump. A
power fit over all `n=2:9` is about `0.904 n^0.607`, but this is not reliable
evidence of asymptotically super-square-root growth: the local exponent from
`n=8` to `n=9` is only `0.514`, and `Adv±/sqrt(n)` is nearly flat there.
Larger exact instances become expensive because `CS(n)` has only reflection
symmetry; most query stabilizers are trivial, so the largest PSD block remains
the full `2^n` input dimension.

#### Compact promises for CS(n)

For a collision-free input `S` (an independent set of the host), define

```text
c(S) = |{v not in S : S union {v} contains a host edge}|.
```

`run_cs_collision_promises.jl` restricts the negative inputs to selected
reflection-orbits of independent sets and includes their one-bit positive
completions. It can rank every orbit by its standalone adversary value or use
true marginal selection:

```bash
julia --project=.. run_cs_collision_promises.jl 8 20 mosek false 3.2886274707683962 cs_collision_promise_n8.csv
julia --project=.. run_cs_collision_promises.jl marginal 8 8 mosek false 3.2886274707683962 cs_collision_marginal_n8.csv
python3 plot_cs_collision_promises.py cs_collision_promise_growth.svg
```

For `CS(8)`, eight marginally selected negative orbits and their positive
neighbors use 76 of the 256 full inputs and give `3.2697644752`, or `99.426%`
of the full adversary value. Five negative orbits use 53 inputs and already
give `3.2086651506`, or `97.569%`. For `CS(9)`, eight selected negative orbits
use 78 of 512 inputs and give `3.4187304766`, or `97.847%` of the full value.

High `c(S)` is a useful first filter, but it does not completely predict
importance. The marginal search also selects singletons and reflection-fixed
independent sets because they add crossing-pair constraints complementary to
the current promise at low orbit cost.

## Current useful reference runs

| instance | value | inputs | pair orbits | max PSD block | peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `n=6, q=120` | `4.8939156335` | `1010` | `520` | `93` | about `1.1 GB` |
| `n=6, q=180` | `5.0974954348` | `2810` | `3554` | `241` | about `1.5 GB` |
| `n=6, selected 37-orbit q=360 subset` | `5.1453521482` | `6050` | `13797` | `456` | about `6.2 GB` |
| `n=7, q=200` | `5.0888813193` | `730` | `151` | `45` | about `1.3 GB` |
| `n=7, q=500` | `5.9387790857` | `9007` | `7278` | `359` | about `3.5 GB` |

The attempted `n=7, q=750` instance built the model but was killed at MOSEK
optimization. It had `16567` selected inputs, `20882` crossing-pair orbits, and
maximum PSD block size `583`.

## The adversary dual

The triangle function is represented on graphs on `[n]`. There are
`m = binomial(n,2)` possible edges. A graph is encoded by an integer `x+1`,
where the bits of `x` indicate which edges are present. The predicate is

```text
f(x) = 1 iff graph x contains a triangle.
```

The general adversary dual for a Boolean function can be written with one PSD
matrix `X_j` for each query bit / edge `j`, and a scalar `t`:

```text
minimize     t
subject to   sum_{j : x_j != y_j} X_j[x,y] = 1       for f(x) != f(y)
             sum_j X_j[x,x] <= t                     for every input x
             X_j >= 0                                for every edge j.
```

This implementation uses the original function-evaluation adversary dual. It
only constrains opposite-output input pairs. It does not add state-conversion
constraints forcing same-output pairs to zero.

The returned objective value is an adversary lower bound for the selected
partial function. Because the selected domain is a restriction of triangle
finding, this is also a valid lower bound for the full triangle problem.

## Symmetry reduction

The symmetric group `S_n` acts on graph inputs by relabeling vertices. It also
acts transitively on query edges. Therefore all edge matrices `X_j` can be
recovered from one canonical edge matrix `X_e0`, where `e0 = {1,2}`.

Let `H = Stab({1,2})`, the subgroup preserving the canonical edge. The matrix
`X_e0` can be assumed `H`-invariant:

```text
X_e0 = average_{h in H} h * Y * h^{-1}
```

Instead of optimizing over a full dense matrix, the code asks
SymbolicWedderburn for a symmetry-adapted basis of the selected input
representation restricted to `H`. In that basis, an `H`-invariant PSD matrix
decomposes into smaller PSD blocks. Those blocks are the JuMP variables:

```text
B_1 >= 0, B_2 >= 0, ..., B_k >= 0.
```

The solver never creates independent PSD variables for every edge. For an edge
`j`, it transports the canonical edge matrix by a vertex permutation sending
`{1,2}` to edge `j`. Each constraint evaluates entries of these transported
matrices.

Important implementation points:

- `triangle_common.jl` defines graph encodings and full graph permutations for
  small domains.
- `triangle_adversary_dual_restricted.jl` can also expand only explicitly
  selected graph orbits. This is what makes `n=7` possible without enumerating
  all `2^21` graph inputs as a permutation object.
- `run_triangle_symmetric.jl` chooses graph orbit representatives whose orbit
  size is at most `q`. Equivalently, it keeps graphs with automorphism group
  size at least `n! / q`.

## Restricted duals and orbit pruning

The full adversary dual for `n=6` or `n=7` is too large. We therefore restrict
the input domain to a union of `S_n`-orbits of graphs. The SDP solved is exactly
the adversary dual for that restricted partial function.

There are three ways to choose the domain:

1. `q`-threshold: keep graph orbits of size at most `q`, then apply boundary
   pruning rules.
2. Explicit representatives: solve only the graph orbits listed on the command
   line.
3. Stored useful orbits: load a set from `triangle_orbit_results.toml`.

After a solve, the code computes orbit diagnostics:

- `objective_multiplier`: dual multiplier on the diagonal bound for that graph
  orbit.
- `crossing_multiplier_mass`: total absolute multiplier mass of crossing-pair
  constraints incident to that orbit.
- `activity`: how close the diagonal bound is to being tight.
- `importance`: a heuristic combination of the above.

Backward pruning removes low-importance orbits in batches and resolves the SDP.
The removal is accepted only if the new objective value stays above the chosen
relative threshold. This means every accepted value is still a certified
restricted-domain adversary lower bound; the importance score only proposes
candidates.

Forward search starts from a small stored/domain set and tests candidate orbit
additions. Candidate screening favors:

- positive graphs with exactly one triangle;
- negative triangle-free graphs with many missing edges that would complete a
  triangle.

The selected winner is not chosen by the heuristic alone: each screened
candidate is added, the SDP is solved, and the orbit with the largest solved
marginal gain is accepted.

## Primal formulation and beta weights

The implemented primal uses the formulation

```text
maximize     sum_{x,y in D} Gamma[x,y]

subject to   diag(beta) - Gamma o Delta_j >= 0       for every edge j
             Gamma[x,y] = 0                          when f(x) = f(y)
             sum_{x in f^{-1}(1)} beta[x] = 1/2
             sum_{x in f^{-1}(0)} beta[x] = 1/2
             beta[x] >= 0.
```

Here `Delta_j[x,y] = 1` iff inputs `x` and `y` differ on edge `j`.
The variable `Gamma` is free on opposite-output pairs and zero on same-output
pairs. The implementation assumes `S_n`-invariance:

- `beta` is one variable per graph orbit;
- `Gamma` is one free variable per crossing-pair orbit;
- it is enough to impose the PSD constraint for the canonical edge `{1,2}`;
- that canonical-edge PSD matrix is block-diagonalized by the same
  `H = Stab({1,2})` Wedderburn decomposition used in the dual.

The primal is useful for interpreting `beta`, but it is currently more
memory-heavy than the dual because it adds one equality for each canonical-edge
matrix entry. For example, `n=6,q=120` builds about `510k` entry constraints and
was killed at MOSEK optimization on this machine; `n=6,q=50` and `n=7,q=50`
solve comfortably.

## What high automorphism inputs do

An orbit of a graph `G` has size

```text
|Orb(G)| = n! / |Aut(G)|.
```

So small `q` keeps graphs with large automorphism groups. These inputs are
cheap in two different senses:

1. They add few labeled inputs to the restricted domain.
2. They often make the stabilizer representation decompose into smaller
   Wedderburn blocks.

However, high automorphism alone is not enough. With very small `q`, the
domain quickly collapses to only the empty and complete graphs, whose adversary
value is essentially `1`. Strong lower bounds appear when the domain also
contains boundary orbits:

- triangle-free graphs with many triangle-completing missing edges;
- one-triangle positive graphs close to those negative boundary graphs.

For `n=7, q=500`, the most important positive inputs are mostly one-triangle
graphs, and the most important negative inputs are triangle-free graphs with
many ways to create a triangle by adding one edge. The empty and complete
graphs are useful anchors, but they are not the source of the large bound.

As q increases, more boundary families enter. The objective rises, but the PSD
blocks and crossing-pair constraints grow. The practical limit is determined
mostly by the largest Wedderburn block and MOSEK memory, not by the raw number
of orbit representatives alone.

## Bipartite-plus-one restriction

The command

```bash
julia --project=.. run_triangle_symmetric.jl bipartite-plus-one n
```

uses the `S_n`-closure of the following construction:

- negative inputs are bipartite graphs;
- positive inputs are graphs that become bipartite after deleting one edge.

Equivalently, positives are obtained by taking a bipartite graph on a partition
`A union B` and adding one edge inside `A` or inside `B`; if the added edge has
a common neighbor across the partition, it creates a triangle.

For `n=5`, this restriction keeps `25` of the `34` graph orbits and gives

```text
restricted value = 4.0059386097
full value       = 4.0632665013
ratio            = 0.9858911810
```

For `n=6`, the same construction already gives `82` graph orbits, `18082`
selected inputs, `100483` crossing-pair orbits, and largest PSD block `1345`.
That model is too large for this machine in the current dual implementation.

If the negative hosts are restricted further to complete bipartite graphs
`K_{A,B}`, and positives are `K_{A,B}` plus one edge inside `A` or inside `B`,
the instances are tiny. With balance condition
`min(|A|,|B|) >= floor(n/2)-k`, the values are:

```text
n=5, compared to full n=5 value 4.0632665013:
k=0  value=2.9586398003  ratio=0.7281431822  orbits=3  inputs=50
k=1  value=3.6636025504  ratio=0.9016397397  orbits=5  inputs=85
k=2  value=3.6636025358  ratio=0.9016397361  orbits=6  inputs=86

n=6, compared to best current restricted benchmark 5.1453521482:
k=0  value=3.2928189113  ratio=0.6399598738  orbits=2  inputs=70
k=1  value=3.8403110728  ratio=0.7463650616  orbits=5  inputs=190
k=2  value=4.4574455640  ratio=0.8663052471  orbits=7  inputs=256
k=3  value=4.4574455748  ratio=0.8663052492  orbits=8  inputs=257

n=7:
k=0  value=4.1055045977  orbits=3  inputs=350   maxblock=20
k=1  value=4.6872492347  orbits=6  inputs=602   maxblock=34
k=2  value=5.1791948008  orbits=8  inputs=714   maxblock=42
k=3  value=5.1791950039  orbits=9  inputs=715   maxblock=42

n=8:
k=0  value=4.4570269392  orbits=2  inputs=455   maxblock=12
k=1  value=4.8896923436  orbits=5  inputs=1239  maxblock=32
k=2  value=5.4660970537  orbits=8  inputs=1715  maxblock=46
```

The jump from `k=1` to `k=2` for `n=6` adds the star-like complete bipartite
host `K_{1,5}` and its one-edge positive counterpart, which again carries a
large part of the adversary weight.

Another useful family search ranks negative graph orbits by

```text
completing_edges(G) = |{e notin G : G+e contains a triangle}|.
```

At each round it adds the highest-ranked negative orbit and all positive orbit
neighbors obtained by adding one completing edge. For `n=6`, the first eight
rounds give:

```text
round  value        inputs  pair_orbits  maxblock
1      3.691801236     68       6           9
2      4.892707854   1328     338          97
3      4.954884428   1448     431         110
4      4.963194568   2048     953         161
5      5.045457182   3488    2458         270
6      5.143366121   5288    5718         384
7      5.148552744   6128    7643         450
8      5.174055266   6198    7874         458
```

The final 29-orbit set from this simple rule beats the earlier hand-built
`n=6` selected q=360 subset value `5.1453521482`, while staying below about
`3.1 GB` peak RSS.

For `n=4`, the full/reference value is `3.0`. The same greedy rule gives:

```text
round  value        inputs  pair_orbits  maxblock
1      2.299322012     18       6           8
2      2.794191494     27      14          13
3      2.794191506     39      25          17
4      2.999999983     55      46          23
```

For `n=7`, the first five greedy rounds give:

```text
round  value        inputs  pair_orbits  maxblock
1      4.395719970    114       6           9
2      5.753855378   5574     574         146
3      5.957143645  11244    2531         287
4      6.005144072  11496    2723         298
5      6.230222663  17796    6480         461
```

The exact greedy ranking for `n=8` needs an unlabeled-graph orbit generator.
The current direct implementation ranks candidates by scanning the full
`2^28` labeled graph cube, which is not a good use of memory. A useful
greedy-style seed is still cheap: the star `K_{1,7}` and its one-edge
positive completion give value `5.1008534305` with `178` inputs and max block
`9`. Mixing this star seed with the complete-bipartite `k=2` family gives
value `5.8944691730` with `12` orbits, `1893` inputs, `122` crossing-pair
orbits, and max block `54`.

These data suggest using a more conservative growth heuristic than the earlier
`n^(4/3)` fit. Normalizing at the `n=6` greedy value gives

```text
Adv(triangle,n) ~= 0.551 n^1.25.
```

This predicts:

```text
n=4   3.1169
n=5   4.1196
n=6   5.1741
n=7   6.2736
n=8   7.4132
n=9   8.5891
n=10  9.7981
```

The measured greedy values `n=6 -> 5.1741` and `n=7 -> 6.2302` are very close
to this `n^1.25` scale, while complete-bipartite-only domains sit well below
it for larger `n`.

The file `triangle_adversary_evolution.svg` plots these values against
normalized power curves `5.174(n/6)^t` for `t in {1, 1.1, 1.2, 1.25}`. In this
plot, fixed complete bipartite `|A|=k` means the input family generated by
`K_{k,n-k}` and the graphs obtained from it by adding one edge inside `A` or
inside `B`. This is different from the cumulative balance-window parameter used
above. The plot is generated by:

```bash
python3 plot_adversary_evolution.py
```

The plotted data are:

```text
family                     n=4       n=5       n=6       n=7       n=8
full / best greedy         3.0000    4.0633    5.1741    6.2302      -
greedy almost-triangle     3.0000    4.0464    5.1741    6.2302      -
fixed complete |A|=1       2.2993    2.9907    3.6918    4.3957    5.1009
fixed complete |A|=2       2.0000    2.9586    3.6465    4.2977    4.9621
fixed complete |A|=3          -         -      3.2928    4.1055    4.8026
fixed complete |A|=4          -         -         -         -      4.4570
cumulative complete best   2.7942    3.6636    4.4574    5.1792    5.4661
bipartite plus one            -      4.0059      -         -         -
n=8 mixed seed                -        -         -         -       5.8945
```

With the fixed-size definition, `|A|=1` is the star family `K_{1,n-1}`. It is
the best single complete-bipartite family in these runs, which supports the
intuition that high-degree triangle-free graphs with many completing edges are
more useful than balanced complete bipartite graphs by themselves.
