# Installation and solver setup

## PyPI

```bash
python -m pip install query-complexity
```

Requires Python >=3.10, NumPy >=1.23, SciPy >=1.9 and CVXPY >=1.3.
Clarabel is used in introductory examples; current CVXPY distributions
include Clarabel and SCS. Inspect the solvers available in your environment:

```bash
python -c 'import cvxpy; print(cvxpy.installed_solvers())'
```

If missing, install `clarabel` and `scs` with pip. Some core routines choose
an installed solver automatically, so pass `solver="CLARABEL"` explicitly
when you want to avoid using a licensed solver.

Extras: `experiments` adds Matplotlib; `symbolic` adds SymPy for exact
symbolic audits; `mosek` adds the Python MOSEK package, which also needs a
valid license; `qiskit` adds optional circuit export support; `dev` adds
pytest, build and twine. Combine extras with commas.

## Source checkout

```bash
git clone https://github.com/stockfish-soup/query-complexity.git
cd query-complexity
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,experiments,symbolic]'
python -m pytest -q
```

Because this project uses `src/`, install it before running scripts.
Setting `PYTHONPATH=.` is not a substitute for installation.
Existing imports remain valid. `query_complexity.core` is also available,
but examples use the public `query_complexity` namespace.

## Julia

Pip does not install Julia or the Julia solvers. Use a source checkout and
Julia 1.12.5 for the pinned environment:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/test/runtests.jl
```

The environment includes a relative local SDPLRPlus dependency. Preserve
`julia/vendor/SDPLRPlus` when copying the project. See
[provenance](../julia/PROVENANCE.md) for its source and license.

## Numerical interpretation

Check solver status, residuals, PSD eigenvalues and scale. An
`optimal_inaccurate` result is not an exact feasibility certificate.
Fixed-rank searches are nonconvex; failure does not prove infeasibility.
The hierarchy's `numerically_feasible` flag uses floating tolerances.
Full graph enumeration grows exponentially and is limited to small sizes.
