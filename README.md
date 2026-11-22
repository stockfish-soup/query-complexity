# query-complexity

Python research tools for quantum query complexity: adversary-bound SDPs,
phase-oracle transducers, symmetric-junta learning, and representation-based
SDP hierarchies. The repository also contains Julia triangle-finding solvers
using SymbolicWedderburn.

## Install

```bash
python -m pip install query-complexity
```

Python 3.10 or later is required. NumPy, SciPy and CVXPY are installed as
core dependencies. The examples below use analytic formulas or Clarabel;
they do not require a commercial solver license.

Optional extras:

```bash
python -m pip install 'query-complexity[experiments]'  # Matplotlib
python -m pip install 'query-complexity[symbolic]'     # SymPy audit checks
python -m pip install 'query-complexity[mosek]'        # requires a MOSEK license
python -m pip install 'query-complexity[qiskit]'       # optional circuit export
```

## Use cases

### Construct a small query transducer

```python
import query_complexity as qc

solution = qc.analytic_or_solution(3)
transducer = qc.build_adversary_transducer(solution)
print(solution.objective_value)  # sqrt(3)/2
print(transducer.private_dim)    # 3
print(transducer.map_error)      # approximately 1e-15
```

The Boolean-evaluation module uses the phase-normalized value `Adv±(f)/2`.
It also solves general and symmetric-function SDPs, minimizes witness trace,
and searches for fixed-rank witnesses.

### Learn a majority junta

```python
from junta_learning import majority_sdp, rank_one

z = rank_one.count_witness(7)
print(z @ z)  # explicit rank-one witness cost: 5.155555...
cost, matrix, status = majority_sdp.solve(7)
print(cost, status)  # approximately 5.069052, optimal
```

The explicit period-four witness has proved `O(log k)` cost. The convex
count SDP can improve its cost; it does not impose rank one. Learning costs
use `max_A sum_S X_S[A,A]`, without the phase factor `1/2`.

### Compare an intersection-free SDP with higher irrep levels

```python
from sdp_hierarchy import Hierarchy, symmetric_evaluation

hierarchy = Hierarchy(symmetric_evaluation(3, [0, 0, 1, 1]))
for r in range(hierarchy.max_level + 1):
    result = hierarchy.solve(r)
    print(r, result["cost"], result["numerically_feasible"])
# Level 0: approximately 3; level 1: approximately 2.
```

Level `r` keeps the first `r+1` present irreducible types of each query
stabilizer, starting with the trivial type. Level 0 is the query-orbit
intersection-free model. The final level equals the full covariant SDP.
This package enumerates finite group actions and targets small instances.

### Test triangle finding

```python
from sdp_hierarchy import Hierarchy, triangle_finding

hierarchy = Hierarchy(triangle_finding(4))
result = hierarchy.solve(2)
print(result["cost"], result["status"])  # approximately 3
```

Four vertices means 64 graph inputs. Five vertices already exceed the
reference backend's practical memory budget. Solver status and residuals
must be checked: some boundary cases return `optimal_inaccurate`.
No claimed numerical result here is an exact feasibility certificate.

## Documentation and runnable examples

- [Documentation index](https://github.com/stockfish-soup/query-complexity/blob/main/docs/README.md)
- [Installation and solvers](https://github.com/stockfish-soup/query-complexity/blob/main/docs/installation.md)
- [Example walkthroughs](https://github.com/stockfish-soup/query-complexity/blob/main/docs/examples.md)
- [Python API reference](https://github.com/stockfish-soup/query-complexity/blob/main/docs/api.md)
- [Query complexity and transducers](https://github.com/stockfish-soup/query-complexity/blob/main/docs/guides/query-complexity.md)
- [Intersection-free junta learning](https://github.com/stockfish-soup/query-complexity/blob/main/docs/guides/junta-learning.md)
- [Irrep-prefix hierarchy definition](https://github.com/stockfish-soup/query-complexity/blob/main/docs/guides/irrep-hierarchy.md)
- [Triangle scaling results and limits](https://github.com/stockfish-soup/query-complexity/blob/main/results/hierarchy/TRIANGLE_LEVEL2_SCALING.md)

Run the examples from a source checkout after an editable installation:

```bash
python -m pip install -e '.[dev,experiments]'
python examples/example_or.py
python examples/general_function.py
python examples/majority_junta.py
python examples/symmetric_hierarchy.py
python examples/triangle_hierarchy.py --vertices 4
python -m pytest -q
```

The wheel installs three import packages: `query_complexity`,
`junta_learning`, and `sdp_hierarchy`. Experiments, archives, notebooks and
Julia tools are kept in the repository and source distribution, outside
the runtime packages. See the documentation index for the repository map.

## Julia solvers

Julia is a separate environment and is not installed by pip. From a source
checkout, use Julia 1.12.5 to reproduce the pinned dependencies:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/test/runtests.jl
julia --project=julia julia/triangle/triangle_cb_degree_sdp.jl 3 2 0.3333333333333333 clarabel
```

See [the Julia guide](https://github.com/stockfish-soup/query-complexity/blob/main/julia/triangle/README.md)
and [dependency provenance](https://github.com/stockfish-soup/query-complexity/blob/main/julia/PROVENANCE.md).

## License

MIT. Vendored SDPLRPlus retains its upstream license in
`julia/vendor/SDPLRPlus/LICENSE`.
