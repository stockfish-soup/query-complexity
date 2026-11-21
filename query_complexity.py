"""
query_complexity_single_file.py

A single-file implementation of adversary-bound SDPs and space-efficient
phase-oracle transducers for Boolean query complexity.

Included:
  - one-vector adversary SDP for arbitrary Boolean functions;
  - rescaled primal Schrijver/Terwilliger value SDP for symmetric Boolean functions;
  - Terwilliger/orbit-reduced dual SDP helpers for witness/transducer construction;
  - adversary transducer extraction;
  - repeated transducer simulation and optional Qiskit circuit export;
  - analytic threshold helpers for the rank-one Schrijver blocks.

Removed:
  - two-vector SDP;
  - direct-sum/bit-flip transducer;
  - primal-only and polynomial-method experiments;
  - duplicate main scripts and rescaled variants.

Conventions:
  - Boolean inputs are tuples in {0,1}^n.
  - The phase oracle is Q_x |i> = (-1)^{x_i} |i>.
  - The SDP uses X_i[x,y] = <z_{x,i}, z_{y,i}> and catalyst pieces
        w_{x,i} = z_{x,i}/sqrt(2).
  - The phase-normalized objective is max_x 1/2 sum_i ||z_{x,i}||^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial, sqrt
from numbers import Number
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from scipy.sparse import coo_matrix

try:
    import cvxpy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    _CVXPY_IMPORT_ERROR = exc
else:
    _CVXPY_IMPORT_ERROR = None


Array = np.ndarray
LayerValues = dict[int, int]
OrbitKey = tuple[int, int, int, int, int]  # alpha, beta, a, b, t


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdversarySolution:
    """Explicit one-vector adversary solution in the Boolean input basis."""

    inputs: list[tuple[int, ...]]
    outputs: list[int]
    objective_value: float       # first-pass minimax value
    trace_value: float           # second-pass sum objective
    status: str

    # z[i][x_idx, :] is z_{x,i}; catalyst uses z/sqrt(2).
    z: list[Array]
    grams: list[Array]
    input_costs: Array

    public_dim: int
    witness_dims: list[int]
    private_dim: int

    # Optional provenance when expanded from an orbit solution.
    orbit_solution: OrbitSolution | None = None


@dataclass(frozen=True)
class OrbitSolution:
    """Compact Terwilliger/orbit solution for symmetric Boolean functions."""

    n: int
    layer_values: LayerValues
    allowed_layers: tuple[int, ...]
    objective_value: float       # first-pass minimax value
    trace_value: float           # second-pass weighted trace value
    cap: float
    status: str

    orbit_values: dict[OrbitKey, float]
    block_values: list[Array]
    block_labels: list[list[tuple[int, int]]]
    input_costs_by_layer: dict[int, float]


@dataclass(frozen=True)
class SdpValueResult:
    """Value-only SDP result, with no trace minimization or witness extraction."""

    objective_value: float       # phase-normalized adversary value T
    usual_adversary_value: float # standard adversary value 2T
    status: str


@dataclass(frozen=True)
class AdversaryTransducer:
    """Input-independent adversary transducer unitary."""

    U: Array
    source_states: Array         # columns |0> + Q_x|w_x>
    target_states: Array         # columns |f(x)> + |w_x>
    inputs: list[tuple[int, ...]]
    outputs: list[int]

    public_dim: int
    private_dim: int
    witness_dims: list[int]
    private_slices: dict[int, slice]

    catalyst_norms: Array
    gram_error: float
    map_error: float
    unitarity_error: float


@dataclass(frozen=True)
class FixedRankFeasibilityResult:
    """Candidate from the nonconvex factorization X_i = Z_i Z_i.T."""

    inputs: list[tuple[int, ...]]
    outputs: list[int]
    rank_bound: int
    alpha: float
    adversary_value: float
    cap: float
    feasible: bool
    status: str

    z: list[Array]
    grams: list[Array]
    input_costs: Array
    trace_value: float

    max_equality_residual: float
    max_cap_violation: float
    residual_norm: float
    attempts: int
    total_nfev: int


@dataclass(frozen=True)
class SymmetricFixedRankFeasibilityResult:
    """Candidate from low-rank factors for the symmetric orbit dual blocks."""

    n: int
    layer_values: LayerValues
    allowed_layers: tuple[int, ...]
    block_rank_bound: int
    full_rank_bound: int
    block_multiplicities: list[int]
    alpha: float
    adversary_value: float
    cap: float
    feasible: bool
    status: str

    orbit_values: dict[OrbitKey, float]
    block_values: list[Array]
    block_labels: list[list[tuple[int, int]]]
    input_costs_by_layer: dict[int, float]
    trace_value: float

    max_equality_residual: float
    max_block_residual: float
    max_cap_violation: float
    residual_norm: float
    attempts: int
    total_nfev: int


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def require_cvxpy() -> None:
    if cp is None:  # pragma: no cover
        raise ImportError("cvxpy is required. Install it with `pip install cvxpy`.") from _CVXPY_IMPORT_ERROR


def normalize_bit_tuple(x: Any) -> tuple[int, ...]:
    """Convert a Boolean input to a tuple in {0,1}^n."""
    if isinstance(x, str):
        bits = tuple(int(c) for c in x)
        if not set(bits).issubset({0, 1}):
            raise ValueError(f"string input {x!r} is not binary")
        return bits

    bits = tuple(int(v) for v in x)
    vals = set(bits)
    if vals.issubset({0, 1}):
        return bits
    if vals.issubset({-1, 1}):
        return tuple(1 if v == 1 else 0 for v in bits)
    raise ValueError(f"input {x!r} is not Boolean")


def normalize_output(y: Any) -> int:
    return 0 if y in (-1, 0, False) else 1


def hamming_weight(x: Sequence[int]) -> int:
    return int(sum(int(b) for b in x))


def all_binary_inputs(n: int) -> list[tuple[int, ...]]:
    if n < 0:
        raise ValueError("n must be non-negative")
    return [tuple((k >> (n - 1 - j)) & 1 for j in range(n)) for k in range(2**n)]


def representative_for_weight(n: int, weight: int) -> tuple[int, ...]:
    weight = int(weight)
    if weight < 0 or weight > n:
        raise ValueError("weight must be in 0..n")
    return tuple([1] * weight + [0] * (n - weight))


@lru_cache(maxsize=None)
def binom_safe(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return comb(n, k)


def normalize_layer_values(layer_values: Mapping[int, Any]) -> LayerValues:
    out: LayerValues = {}
    for k, v in layer_values.items():
        out[int(k)] = normalize_output(v)
    return dict(sorted(out.items()))


# Common symmetric Boolean functions.
def OR_layers(n: int) -> LayerValues:
    return {s: int(s > 0) for s in range(n + 1)}


def AND_layers(n: int) -> LayerValues:
    return {s: int(s == n) for s in range(n + 1)}


def THRESHOLD_layers(n: int, k: int) -> LayerValues:
    return {s: int(s >= int(k)) for s in range(n + 1)}


def EXACT_layers(n: int, k: int) -> LayerValues:
    return {s: int(s == int(k)) for s in range(n + 1)}


def PARITY_layers(n: int) -> LayerValues:
    return {s: int(s % 2) for s in range(n + 1)}


# ---------------------------------------------------------------------------
# Numerical linear algebra
# ---------------------------------------------------------------------------

def psd_factor(M: Array, tol: float = 1e-8) -> Array:
    """Return R with M approximately equal to R @ R.T."""
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    vals, vecs = la.eigh(M)
    max_val = float(np.max(np.abs(vals))) if vals.size else 0.0
    cutoff = max(tol, tol * max(1.0, max_val))
    if vals.size and float(np.min(vals)) < -100 * cutoff:
        raise ValueError(f"PSD factorization failed: min eigenvalue {np.min(vals):.3e}")
    keep = vals > cutoff
    if not np.any(keep):
        return np.zeros((M.shape[0], 0), dtype=float)
    return vecs[:, keep] * np.sqrt(vals[keep])[None, :]


def complete_basis(Q: Array, tol: float = 1e-10) -> Array:
    """Complete an orthonormal basis Q to a square unitary matrix."""
    Q = np.asarray(Q, dtype=complex)
    n, r = Q.shape
    if r == n:
        return Q
    N = la.null_space(Q.conj().T, rcond=tol)
    E = np.hstack([Q, N])
    if E.shape == (n, n):
        return E

    # Robust fallback.
    cols: list[Array] = []
    for j in range(r):
        v = Q[:, j].copy()
        for q in cols:
            v -= q * np.vdot(q, v)
        nv = la.norm(v)
        if nv > tol:
            cols.append(v / nv)
    for j in range(n):
        v = np.zeros(n, dtype=complex)
        v[j] = 1.0
        for q in cols:
            v -= q * np.vdot(q, v)
        nv = la.norm(v)
        if nv > tol:
            cols.append(v / nv)
        if len(cols) == n:
            break
    return np.column_stack(cols)


def unitary_from_state_pairs(
    source: Array,
    target: Array,
    *,
    gram_tol: float = 1e-7,
    eig_tol: float = 1e-9,
) -> tuple[Array, float, float, float]:
    """Return a unitary U such that U @ source approximately equals target."""
    source = np.asarray(source, dtype=complex)
    target = np.asarray(target, dtype=complex)
    if source.shape != target.shape:
        raise ValueError("source and target must have the same shape")

    d, _ = source.shape
    Gs = source.conj().T @ source
    Gt = target.conj().T @ target
    gram_error = float(la.norm(Gs - Gt, ord="fro"))

    G = 0.5 * (Gs + Gs.conj().T)
    vals, vecs = la.eigh(G)
    scale = max(1.0, float(np.max(np.abs(vals))) if vals.size else 1.0)
    keep = vals > eig_tol * scale
    if not np.any(keep):
        U = np.eye(d, dtype=complex)
        return U, gram_error, float(la.norm(U @ source - target, ord="fro")), 0.0

    vals_keep = vals[keep]
    V = vecs[:, keep]
    Qs = source @ V @ np.diag(1.0 / np.sqrt(vals_keep))
    Qt = target @ V @ np.diag(1.0 / np.sqrt(vals_keep))

    Qs_qr, R = la.qr(Qs, mode="economic")
    Qt = Qt @ la.inv(R)
    Qs = Qs_qr

    H = Qt.conj().T @ Qt
    if H.size and la.norm(H - np.eye(H.shape[0]), ord="fro") > max(gram_tol, 10 * gram_error):
        vals_h, vecs_h = la.eigh(0.5 * (H + H.conj().T))
        vals_h = np.clip(vals_h, 1e-15, None)
        Qt = Qt @ (vecs_h @ np.diag(1.0 / np.sqrt(vals_h)) @ vecs_h.conj().T)

    Es = complete_basis(Qs)
    Et = complete_basis(Qt)
    U = Et @ Es.conj().T
    map_error = float(la.norm(U @ source - target, ord="fro"))
    unitarity_error = float(la.norm(U.conj().T @ U - np.eye(d), ord="fro"))
    return U, gram_error, map_error, unitarity_error


# ---------------------------------------------------------------------------
# CVXPY helpers
# ---------------------------------------------------------------------------

def default_solver():
    require_cvxpy()
    installed = set(cp.installed_solvers())
    if "MOSEK" in installed:
        return cp.MOSEK
    if "SDPA" in installed:
        return cp.SDPA
    if "CLARABEL" in installed:
        return "CLARABEL"
    return cp.SCS


def solve_problem(problem: Any, solver: str | None = None, solver_kwargs: Mapping[str, Any] | None = None, verbose: bool = False):
    require_cvxpy()
    opts = dict(solver_kwargs or {})
    chosen = solver if solver is not None else default_solver()
    if chosen == cp.SCS or chosen == "SCS":
        opts.setdefault("eps", 1e-6)
        opts.setdefault("max_iters", 200000)
    return problem.solve(solver=chosen, verbose=verbose, **opts)


def scalar_to_1x1(expr: Any):
    if isinstance(expr, Number):
        expr = cp.Constant(float(expr))
    return cp.reshape(expr, (1, 1), order="C")


def scalar_bmat(entries: list[list[Any]]):
    return cp.bmat([[scalar_to_1x1(e) for e in row] for row in entries])


# ---------------------------------------------------------------------------
# General adversary SDP
# ---------------------------------------------------------------------------

def normalize_instance(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any] | None = None,
    f: Callable[[tuple[int, ...]], Any] | None = None,
) -> tuple[list[tuple[int, ...]], list[int]]:
    xs = [normalize_bit_tuple(x) for x in inputs]
    if not xs:
        raise ValueError("inputs must be non-empty")
    n = len(xs[0])
    if any(len(x) != n for x in xs):
        raise ValueError("all inputs must have the same length")
    if outputs is None:
        if f is None:
            raise ValueError("provide outputs or f")
        ys = [normalize_output(f(x)) for x in xs]
    else:
        if len(outputs) != len(xs):
            raise ValueError("len(outputs) must equal len(inputs)")
        ys = [normalize_output(y) for y in outputs]
    return xs, ys


def solve_adversary_value_sdp(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any] | None = None,
    f: Callable[[tuple[int, ...]], Any] | None = None,
    *,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    solver_options: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> SdpValueResult:
    """
    Solve only the minimax adversary SDP value for a general Boolean function.

    This runs only the first pass of :func:`solve_adversary_sdp`. It does not
    perform trace minimization and does not factor Gram matrices, so use it when
    you only need the adversary bound or when the trace-minimization pass is
    numerically fragile.

    Returns:
        SdpValueResult where ``objective_value`` is the phase-normalized value
        T and ``usual_adversary_value`` is ``2*T``.
    """
    require_cvxpy()
    xs, ys = normalize_instance(inputs, outputs, f)
    m = len(xs)
    n = len(xs[0])

    X = [cp.Variable((m, m), symmetric=True, name=f"X_{i}") for i in range(n)]
    T = cp.Variable(name="T")

    constraints: list[Any] = [T >= 0]
    constraints.extend(Xi >> 0 for Xi in X)

    for x_idx in range(m):
        cost = 0.5 * cp.sum([X[i][x_idx, x_idx] for i in range(n)])
        constraints.append(cost <= T)

    for a, x in enumerate(xs):
        for b, y in enumerate(xs):
            terms = [X[i][a, b] for i in range(n) if x[i] != y[i]]
            lhs = cp.sum(terms) if terms else 0
            rhs = 1.0 if ys[a] != ys[b] else 0.0
            constraints.append(lhs == rhs)

    problem = cp.Problem(cp.Minimize(T), constraints)
    value = float(solve_problem(problem, solver, solver_kwargs, verbose))
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"adversary value SDP failed with status {problem.status}")

    return SdpValueResult(
        objective_value=value,
        usual_adversary_value=2.0 * value,
        status=str(problem.status),
    )


def solve_trace_minimized_dual(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any] | None = None,
    f: Callable[[tuple[int, ...]], Any] | None = None,
    *,
    alpha: float = 1.0,
    adversary_value: float | None = None,
    diagonal_weights: Sequence[float] | None = None,
    epsilon: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    psd_tol: float = 1e-8,
    verbose: bool = False,
) -> AdversarySolution:
    """
    Minimize a diagonal dual objective subject to an alpha-relaxed query-cost cap.

    By default, the SDP minimizes

        1/2 sum_i Tr(X_i)

    equivalently ``sum_x cost_x``. Passing ``diagonal_weights`` replaces this
    second-pass objective by

        sum_x diagonal_weights[x] * sum_i X_i[x,x].

    This is useful for exposing nonsymmetric extreme points of the feasible
    face while keeping the adversary equalities and query-cost cap unchanged.

    over the usual one-vector adversary dual constraints, with

        max_x 1/2 sum_i X_i[x,x] <= alpha * adversary_value + epsilon.

    ``adversary_value`` uses this module's phase-normalized convention, i.e.
    it is half the usual negative-weight adversary bound.  If it is omitted,
    a minimax pass computes it.  Thus ``alpha=1`` reproduces the original
    near-optimal trace-minimization pass, while increasing ``alpha`` trades
    query cost for a lower-trace (and often lower-rank) transducer witness.
    """
    require_cvxpy()
    alpha = float(alpha)
    if alpha < 1.0:
        raise ValueError("alpha must be at least 1")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if adversary_value is not None and adversary_value <= 0:
        raise ValueError("adversary_value must be positive")

    xs, ys = normalize_instance(inputs, outputs, f)
    m = len(xs)
    n = len(xs[0])
    if diagonal_weights is not None:
        if len(diagonal_weights) != m:
            raise ValueError("diagonal_weights must have one entry per input")
        weights = np.asarray(diagonal_weights, dtype=float)
        if not np.all(np.isfinite(weights)):
            raise ValueError("diagonal_weights must be finite")
    else:
        weights = None

    def build_problem(mode: str, cap: float | None = None):
        X = [cp.Variable((m, m), symmetric=True, name=f"X_{i}") for i in range(n)]
        constraints: list[Any] = [Xi >> 0 for Xi in X]
        costs = []
        for x_idx in range(m):
            c = 0.5 * cp.sum([X[i][x_idx, x_idx] for i in range(n)])
            costs.append(c)

        if mode == "max":
            T = cp.Variable(name="T")
            constraints.append(T >= 0)
            for c in costs:
                constraints.append(c <= T)
            objective = cp.Minimize(T)
        elif mode == "sum":
            if cap is None:
                raise ValueError("cap is required for second pass")
            for c in costs:
                constraints.append(c <= cap)
            if weights is None:
                objective = cp.Minimize(cp.sum(costs))
            else:
                weighted_diag = cp.sum(
                    [
                        float(weights[x_idx])
                        * cp.sum([X[i][x_idx, x_idx] for i in range(n)])
                        for x_idx in range(m)
                    ]
                )
                objective = cp.Minimize(weighted_diag)
        else:
            raise ValueError("mode must be 'max' or 'sum'")

        for a, x in enumerate(xs):
            for b, y in enumerate(xs):
                terms = [X[i][a, b] for i in range(n) if x[i] != y[i]]
                lhs = cp.sum(terms) if terms else 0
                rhs = 1.0 if ys[a] != ys[b] else 0.0
                constraints.append(lhs == rhs)

        return cp.Problem(objective, constraints), X, costs

    if adversary_value is None:
        first_problem, _, _ = build_problem("max")
        first_value = float(solve_problem(first_problem, solver, solver_kwargs, verbose))
        if first_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"first pass failed with status {first_problem.status}")
    else:
        first_value = float(adversary_value)

    cap = alpha * first_value + epsilon
    second_problem, X_vars, costs = build_problem("sum", cap=cap)
    trace_value = float(solve_problem(second_problem, solver, solver_kwargs, verbose))
    if second_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"second pass failed with status {second_problem.status}")

    grams: list[Array] = []
    z_vecs: list[Array] = []
    witness_dims: list[int] = []
    for Xi in X_vars:
        G = np.asarray(Xi.value, dtype=float)
        G = 0.5 * (G + G.T)
        R = psd_factor(G, tol=psd_tol)
        grams.append(G)
        z_vecs.append(R)
        witness_dims.append(int(R.shape[1]))

    input_costs = np.array([float(c.value) for c in costs], dtype=float)
    public_dim = max(max(ys) + 1, 1)
    private_dim = int(sum(witness_dims))

    return AdversarySolution(
        inputs=xs,
        outputs=ys,
        objective_value=first_value,
        trace_value=trace_value,
        status=str(second_problem.status),
        z=z_vecs,
        grams=grams,
        input_costs=input_costs,
        public_dim=public_dim,
        witness_dims=witness_dims,
        private_dim=private_dim,
        orbit_solution=None,
    )


def solve_adversary_sdp(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any] | None = None,
    f: Callable[[tuple[int, ...]], Any] | None = None,
    *,
    epsilon: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    psd_tol: float = 1e-8,
    verbose: bool = False,
) -> AdversarySolution:
    """Solve the adversary dual and trace-minimize at the optimal query cost."""
    return solve_trace_minimized_dual(
        inputs,
        outputs,
        f,
        alpha=1.0,
        epsilon=epsilon,
        solver=solver,
        solver_kwargs=solver_kwargs,
        psd_tol=psd_tol,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Fixed-rank nonconvex dual feasibility
# ---------------------------------------------------------------------------

def solve_fixed_rank_dual_feasibility(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any] | None = None,
    f: Callable[[tuple[int, ...]], Any] | None = None,
    *,
    rank: int,
    alpha: float = 1.0,
    adversary_value: float | None = None,
    epsilon: float = 1e-6,
    initial_solution: AdversarySolution | None = None,
    initial_factors: Sequence[Array] | None = None,
    restarts: int = 4,
    random_seed: int | None = 0,
    cap_penalties: Sequence[float] = (1.0, 100.0, 10000.0),
    max_nfev: int = 1000,
    feasibility_tol: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> FixedRankFeasibilityResult:
    """
    Heuristically search for a dual with ``rank(X_i) <= rank`` for every i.

    This uses the Burer--Monteiro factorization ``X_i = Z_i Z_i.T`` and
    nonlinear least squares. No input-permutation symmetry is imposed. The
    equality residuals are the adversary constraints, while violations of

        1/2 sum_i ||Z_i[x]||^2 <= alpha * adversary_value + epsilon

    receive successively larger penalties.

    A feasible result is a directly checkable witness. A ``not_found`` result
    is *not* an infeasibility certificate because the optimization is
    nonconvex.
    """
    xs, ys = normalize_instance(inputs, outputs, f)
    m = len(xs)
    n = len(xs[0])
    rank = int(rank)
    alpha = float(alpha)
    if rank <= 0:
        raise ValueError("rank must be positive")
    if alpha < 1.0:
        raise ValueError("alpha must be at least 1")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if restarts < 0:
        raise ValueError("restarts must be non-negative")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    if feasibility_tol <= 0:
        raise ValueError("feasibility_tol must be positive")
    penalties = tuple(float(value) for value in cap_penalties)
    if not penalties or any(value <= 0 for value in penalties):
        raise ValueError("cap_penalties must contain positive values")

    if adversary_value is None:
        adversary_value = solve_adversary_value_sdp(
            xs,
            ys,
            solver=solver,
            solver_kwargs=solver_kwargs,
            verbose=verbose,
        ).objective_value
    adversary_value = float(adversary_value)
    if adversary_value <= 0:
        raise ValueError("adversary_value must be positive")
    cap = alpha * adversary_value + epsilon

    x_array = np.asarray(xs, dtype=int)
    y_array = np.asarray(ys, dtype=int)
    pair_a: list[int] = []
    pair_b: list[int] = []
    pair_differences: list[Array] = []
    pair_rhs: list[float] = []
    for a in range(m):
        for b in range(a + 1, m):
            pair_a.append(a)
            pair_b.append(b)
            pair_differences.append(np.flatnonzero(x_array[a] != x_array[b]))
            pair_rhs.append(float(y_array[a] != y_array[b]))
    pair_a_array = np.asarray(pair_a, dtype=int)
    pair_b_array = np.asarray(pair_b, dtype=int)
    pair_rhs_array = np.asarray(pair_rhs, dtype=float)
    num_pairs = len(pair_a)

    # Static sparse-Jacobian indices for equality rows.
    eq_rows: list[int] = []
    eq_coords: list[int] = []
    eq_dims: list[int] = []
    eq_a: list[int] = []
    eq_b: list[int] = []
    for pair_idx, differences in enumerate(pair_differences):
        for coord in differences:
            for dim in range(rank):
                eq_rows.append(pair_idx)
                eq_coords.append(int(coord))
                eq_dims.append(dim)
                eq_a.append(pair_a[pair_idx])
                eq_b.append(pair_b[pair_idx])
    eq_rows_array = np.asarray(eq_rows, dtype=int)
    eq_coords_array = np.asarray(eq_coords, dtype=int)
    eq_dims_array = np.asarray(eq_dims, dtype=int)
    eq_a_array = np.asarray(eq_a, dtype=int)
    eq_b_array = np.asarray(eq_b, dtype=int)
    eq_cols_a = (eq_coords_array * m + eq_a_array) * rank + eq_dims_array
    eq_cols_b = (eq_coords_array * m + eq_b_array) * rank + eq_dims_array

    # Static sparse-Jacobian indices for cap rows.
    cap_inputs = np.repeat(np.arange(m), n * rank)
    cap_coords = np.tile(np.repeat(np.arange(n), rank), m)
    cap_dims = np.tile(np.arange(rank), m * n)
    cap_rows = num_pairs + cap_inputs
    cap_cols = (cap_coords * m + cap_inputs) * rank + cap_dims

    def evaluate(vector: Array) -> tuple[Array, Array, Array]:
        factors = np.asarray(vector, dtype=float).reshape(n, m, rank)
        equalities = np.empty(num_pairs, dtype=float)
        for pair_idx, differences in enumerate(pair_differences):
            a = pair_a_array[pair_idx]
            b = pair_b_array[pair_idx]
            equalities[pair_idx] = (
                np.sum(factors[differences, a, :] * factors[differences, b, :])
                - pair_rhs_array[pair_idx]
            )
        costs = 0.5 * np.sum(factors * factors, axis=(0, 2))
        return factors, equalities, costs

    def residual(vector: Array, penalty: float) -> Array:
        _, equalities, costs = evaluate(vector)
        cap_residuals = np.sqrt(penalty) * np.maximum(costs - cap, 0.0)
        return np.concatenate([equalities, cap_residuals])

    def jacobian(vector: Array, penalty: float):
        factors, _, costs = evaluate(vector)
        eq_data_a = factors[eq_coords_array, eq_b_array, eq_dims_array]
        eq_data_b = factors[eq_coords_array, eq_a_array, eq_dims_array]
        active = costs[cap_inputs] > cap
        rows = np.concatenate([eq_rows_array, eq_rows_array, cap_rows[active]])
        cols = np.concatenate([eq_cols_a, eq_cols_b, cap_cols[active]])
        data = np.concatenate(
            [
                eq_data_a,
                eq_data_b,
                np.sqrt(penalty)
                * factors[cap_coords[active], cap_inputs[active], cap_dims[active]],
            ]
        )
        return coo_matrix(
            (data, (rows, cols)),
            shape=(num_pairs + m, n * m * rank),
        ).tocsr()

    rng = np.random.default_rng(random_seed)
    starts: list[Array] = []
    supplied = initial_factors
    if supplied is None and initial_solution is not None:
        if initial_solution.inputs != xs or initial_solution.outputs != ys:
            raise ValueError("initial_solution does not match inputs and outputs")
        supplied = initial_solution.z
    if supplied is not None:
        if len(supplied) != n:
            raise ValueError("initial factors must contain one matrix per coordinate")
        projected: list[Array] = []
        for factor in supplied:
            factor = np.asarray(factor, dtype=float)
            if factor.ndim != 2 or factor.shape[0] != m:
                raise ValueError("each initial factor must have shape (num_inputs, dimension)")
            old_rank = factor.shape[1]
            if old_rank == rank:
                projected.append(factor.copy())
            elif old_rank < rank:
                projected.append(np.pad(factor, ((0, 0), (0, rank - old_rank))))
            else:
                projection = rng.normal(size=(old_rank, rank)) / sqrt(float(rank))
                projected.append(factor @ projection)
        starts.append(np.stack(projected).reshape(-1))
    for _ in range(restarts):
        starts.append(rng.normal(scale=1.0 / sqrt(float(n)), size=n * m * rank))
    if not starts:
        raise ValueError("provide initial factors or request at least one restart")

    best: tuple[float, float, float, float, Array, Array, Array] | None = None
    total_nfev = 0
    for attempt, start in enumerate(starts, start=1):
        vector = start
        for penalty in penalties:
            result = opt.least_squares(
                lambda value, weight=penalty: residual(value, weight),
                vector,
                jac=lambda value, weight=penalty: jacobian(value, weight),
                max_nfev=max_nfev,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            vector = result.x
            total_nfev += int(result.nfev)
        factors, equalities, costs = evaluate(vector)
        max_equality = float(np.max(np.abs(equalities))) if equalities.size else 0.0
        max_cap_violation = max(0.0, float(np.max(costs - cap)))
        score = max(max_equality, max_cap_violation)
        residual_norm = float(la.norm(equalities))
        candidate = (
            score,
            max_equality,
            max_cap_violation,
            residual_norm,
            factors.copy(),
            equalities.copy(),
            costs.copy(),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
        if verbose:
            print(
                f"fixed-rank attempt {attempt}/{len(starts)}: "
                f"max equality={max_equality:.3e}, "
                f"cap violation={max_cap_violation:.3e}"
            )
        if score <= feasibility_tol:
            break

    assert best is not None
    _, max_equality, max_cap_violation, residual_norm, factors, _, costs = best
    feasible = max(max_equality, max_cap_violation) <= feasibility_tol
    z = [factors[i].copy() for i in range(n)]
    grams = [factor @ factor.T for factor in z]
    return FixedRankFeasibilityResult(
        inputs=xs,
        outputs=ys,
        rank_bound=rank,
        alpha=alpha,
        adversary_value=adversary_value,
        cap=cap,
        feasible=feasible,
        status="feasible" if feasible else "not_found",
        z=z,
        grams=grams,
        input_costs=costs,
        trace_value=float(np.sum(costs)),
        max_equality_residual=max_equality,
        max_cap_violation=max_cap_violation,
        residual_norm=residual_norm,
        attempts=attempt,
        total_nfev=total_nfev,
    )


def fixed_rank_result_to_adversary_solution(
    result: FixedRankFeasibilityResult,
    *,
    feasibility_tol: float = 1e-6,
) -> AdversarySolution:
    """Convert a verified fixed-rank candidate for transducer construction."""
    if (
        not result.feasible
        or result.max_equality_residual > feasibility_tol
        or result.max_cap_violation > feasibility_tol
    ):
        raise ValueError("fixed-rank result is not feasible at the requested tolerance")
    public_dim = max(max(result.outputs) + 1, 1)
    witness_dims = [int(factor.shape[1]) for factor in result.z]
    return AdversarySolution(
        inputs=result.inputs,
        outputs=result.outputs,
        objective_value=result.adversary_value,
        trace_value=result.trace_value,
        status=(
            f"fixed-rank nonconvex candidate; "
            f"equality_residual={result.max_equality_residual:.3e}; "
            f"cap_violation={result.max_cap_violation:.3e}"
        ),
        z=[factor.copy() for factor in result.z],
        grams=[gram.copy() for gram in result.grams],
        input_costs=result.input_costs.copy(),
        public_dim=public_dim,
        witness_dims=witness_dims,
        private_dim=int(sum(witness_dims)),
        orbit_solution=None,
    )


# ---------------------------------------------------------------------------
# Schrijver/Terwilliger coefficients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def valid_intersections(a: int, b: int, N: int) -> tuple[int, ...]:
    lo = max(0, int(a) + int(b) - int(N))
    hi = min(int(a), int(b))
    return tuple(range(lo, hi + 1))


@lru_cache(maxsize=None)
def beta_schrijver(N: int, i: int, j: int, r: int, t: int) -> int:
    total = 0
    for u in range(r, N - r + 1):
        c1 = binom_safe(u, t)
        c2 = binom_safe(N - 2 * r, u - r)
        c3 = binom_safe(N - r - u, i - u)
        c4 = binom_safe(N - r - u, j - u)
        if c1 and c2 and c3 and c4:
            total += (-1 if ((u - t) % 2) else 1) * c1 * c2 * c3 * c4
    return int(total)


@lru_cache(maxsize=None)
def terwilliger_coeff(N: int, i: int, j: int, r: int, t: int) -> float:
    denom_i = binom_safe(N - 2 * r, i - r)
    denom_j = binom_safe(N - 2 * r, j - r)
    if denom_i == 0 or denom_j == 0:
        return 0.0
    beta = beta_schrijver(N, i, j, r, t)
    if beta == 0:
        return 0.0
    return float(beta) / sqrt(float(denom_i * denom_j))


# ---------------------------------------------------------------------------
# Symmetric Terwilliger/orbit SDP
# ---------------------------------------------------------------------------

def orbit_keys(N: int, allowed_layers: Iterable[int]) -> list[OrbitKey]:
    allowed = set(int(s) for s in allowed_layers)
    keys: list[OrbitKey] = []
    for alpha in (0, 1):
        for beta in (0, 1):
            for a in range(N + 1):
                if alpha + a not in allowed:
                    continue
                for b in range(N + 1):
                    if beta + b not in allowed:
                        continue
                    for t in valid_intersections(a, b, N):
                        keys.append((alpha, beta, a, b, t))
    return keys


def full_layer_size(n: int, k: int) -> int:
    """Number of Boolean inputs of length n with Hamming weight k."""
    return binom_safe(int(n), int(k))


def num_pairs_in_orbit(n: int, a: int, b: int, t: int) -> int:
    """
    Number of ordered pairs (x,y) with |x|=a, |y|=b, and |x intersect y|=t.
    """
    n = int(n)
    a = int(a)
    b = int(b)
    t = int(t)
    if not (0 <= a <= n and 0 <= b <= n):
        return 0
    if not (max(0, a + b - n) <= t <= min(a, b)):
        return 0
    return binom_safe(n, a) * binom_safe(a, t) * binom_safe(n - a, b - t)


def gamma_objective_multiplicity(n: int, a: int, b: int, t: int) -> int:
    """Multiplicity of the stored symmetric gamma variable with a <= b."""
    if a == b:
        return num_pairs_in_orbit(n, a, b, t)
    return num_pairs_in_orbit(n, a, b, t) + num_pairs_in_orbit(n, b, a, t)


def _gamma_key(a: int, b: int, t: int) -> tuple[int, int, int]:
    aa, bb = min(int(a), int(b)), max(int(a), int(b))
    return (aa, bb, int(t))


def _active_primal_key(key: tuple[int, int, int, int, int], layer_values: Mapping[int, int]) -> bool:
    """Whether an H-orbit coefficient can appear in M_1 = diag(beta)-Gamma o Delta_1."""
    x1, y1, a, b, t = key
    if x1 == y1 and a == b == t:
        return True
    if x1 != y1 and layer_values[a] != layer_values[b]:
        return True
    return False


def _split_components_by_active_pattern(block: Mapping[str, Any], layer_values: Mapping[int, int]) -> list[list[int]]:
    size = int(block["size"])
    adj = [set([i]) for i in range(size)]
    for row, col, key, coeff in block["terms"]:
        if coeff != 0.0 and _active_primal_key(key, layer_values):
            adj[row].add(col)
            adj[col].add(row)

    seen = [False] * size
    components: list[list[int]] = []
    for v in range(size):
        if seen[v]:
            continue
        stack = [v]
        seen[v] = True
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        components.append(sorted(comp))
    return components


def _primal_terwilliger_blocks(n: int, allowed_layers: Iterable[int]) -> list[dict[str, Any]]:
    """
    Schrijver/Terwilliger blocks for the primal matrix
    M_1 = diag(beta) - Gamma o Delta_1.
    """
    allowed_set = set(int(s) for s in allowed_layers)
    m = int(n) - 1
    blocks: list[dict[str, Any]] = []
    for r in range(m // 2 + 1):
        labels = [
            (eps, i)
            for eps in (0, 1)
            for i in range(r, m - r + 1)
            if eps + i in allowed_set
        ]
        if not labels:
            continue
        terms: list[tuple[int, int, tuple[int, int, int, int, int], float]] = []
        for row, (x1, i) in enumerate(labels):
            for col, (y1, j) in enumerate(labels):
                for tau in valid_intersections(i, j, m):
                    coeff = terwilliger_coeff(m, i, j, r, tau)
                    if coeff == 0.0:
                        continue
                    a = x1 + i
                    b = y1 + j
                    t = x1 * y1 + tau
                    terms.append((row, col, (x1, y1, a, b, t), float(coeff)))
        blocks.append({"r": r, "indices": labels, "size": len(labels), "terms": terms})
    return blocks


def _coefficient_scale_for_primal_term(n: int, kind: str, data: Any) -> float:
    """Denominator converting rescaled primal variables back to beta/gamma."""
    if kind == "beta":
        return float(full_layer_size(n, int(data)))
    if kind == "gamma":
        mult = gamma_objective_multiplicity(n, *data)
        if mult <= 0:
            raise ValueError(f"zero gamma multiplicity for {data}")
        return float(mult)
    raise ValueError(f"unknown term kind {kind!r}")


def _primal_block_row_scales(
    n: int,
    m_block: int,
    terms: list[tuple[int, int, float, str, Any]],
    enabled: bool,
) -> list[float]:
    """Heuristic diagonal congruence scaling DBD for numerical conditioning."""
    if not enabled:
        return [1.0] * m_block
    row_mass = [0.0] * m_block
    for row, col, coeff, kind, data in terms:
        denom = _coefficient_scale_for_primal_term(n, kind, data)
        c = abs(float(coeff)) / denom
        row_mass[row] += c
        if col != row:
            row_mass[col] += c
    scales = []
    for mass in row_mass:
        scales.append(1.0 if mass <= 0 else 1.0 / sqrt(max(mass, 1e-300)))
    return scales


def solve_symmetric_value(
    n: int,
    layer_values: Mapping[int, Any],
    *,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    solver_options: Mapping[str, Any] | None = None,
    verbose: bool = False,
    split_blocks: bool = True,
    diagonal_blocks_as_inequalities: bool = True,
    balance_psd_blocks: bool = True,
) -> dict[str, Any]:
    """
    Solve the rescaled primal adversary SDP for a symmetric Boolean function.

    This is the Schrijver/Terwilliger block-diagonalized primal SDP. It returns
    only the value and primal variables; it does not run trace minimization,
    does not construct one-vector witnesses, and does not build a transducer.

    Variables are rescaled exactly as in the old rescaled implementation:

        alpha_k       = C(n,k) beta_k
        eta_{a,b,t}   = N_{a,b,t} gamma_{a,b,t}

    where N_{a,b,t} is the ordered-pair multiplicity in the full objective.
    Therefore the objective is simply sum eta, and the returned ``value`` is
    the usual adversary bound Adv^pm(f), not the phase-normalized half-value.
    """
    require_cvxpy()
    if solver_kwargs is not None and solver_options is not None:
        raise ValueError("use only one of solver_kwargs or solver_options")
    if solver_kwargs is None:
        solver_kwargs = solver_options

    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative")

    lv = normalize_layer_values(layer_values)
    if not lv:
        raise ValueError("layer_values must be non-empty")
    allowed = tuple(sorted(lv.keys()))
    if min(allowed) < 0 or max(allowed) > n:
        raise ValueError("layer outside 0..n")
    zero_layers = [s for s in allowed if lv[s] == 0]
    one_layers = [s for s in allowed if lv[s] == 1]
    if not zero_layers or not one_layers:
        raise ValueError("function/promise must contain both outputs")

    alpha = {s: cp.Variable(nonneg=True, name=f"alpha_{s}") for s in allowed}

    gamma_keys: set[tuple[int, int, int]] = set()
    for a in allowed:
        for b in allowed:
            if lv[a] == lv[b]:
                continue
            aa, bb = min(a, b), max(a, b)
            for t in range(max(0, aa + bb - n), min(aa, bb) + 1):
                gamma_keys.add((aa, bb, t))
    gamma_keys_sorted = sorted(gamma_keys)
    eta = {key: cp.Variable(name=f"eta_{key[0]}_{key[1]}_{key[2]}") for key in gamma_keys_sorted}

    constraints: list[Any] = [
        cp.sum([alpha[s] for s in zero_layers]) == 0.5,
        cp.sum([alpha[s] for s in one_layers]) == 0.5,
    ]

    raw_blocks = _primal_terwilliger_blocks(n, allowed)
    final_block_sizes: list[int] = []
    psd_cone_sizes: list[int] = []
    scalar_inequality_count = 0

    for block in raw_blocks:
        components = _split_components_by_active_pattern(block, lv) if split_blocks else [list(range(int(block["size"])))]
        for comp in components:
            local = {old: new for new, old in enumerate(comp)}
            active_terms: list[tuple[int, int, float, str, Any]] = []

            for row, col, key, coeff in block["terms"]:
                if row not in local or col not in local:
                    continue
                x1, y1, a, b, t = key
                if x1 == y1 and a == b == t:
                    active_terms.append((local[row], local[col], float(coeff), "beta", int(a)))
                elif x1 != y1 and lv[a] != lv[b]:
                    active_terms.append((local[row], local[col], -float(coeff), "gamma", _gamma_key(a, b, t)))

            if not active_terms:
                continue

            m_block = len(comp)
            row_scales = _primal_block_row_scales(n, m_block, active_terms, balance_psd_blocks)
            final_block_sizes.append(m_block)

            if diagonal_blocks_as_inequalities and all(row == col for row, col, *_ in active_terms):
                diag_entries = [0 for _ in range(m_block)]
                for row, _col, coeff, kind, data in active_terms:
                    var = alpha[int(data)] if kind == "beta" else eta[data]
                    denom = _coefficient_scale_for_primal_term(n, kind, data)
                    c = float(coeff) / denom
                    c *= row_scales[row] * row_scales[row]
                    diag_entries[row] = diag_entries[row] + c * var
                for expr in diag_entries:
                    if isinstance(expr, Number) and abs(float(expr)) == 0.0:
                        continue
                    constraints.append(expr >= 0)
                    scalar_inequality_count += 1
                continue

            entries = [[0 for _ in range(m_block)] for __ in range(m_block)]
            for row, col, coeff, kind, data in active_terms:
                var = alpha[int(data)] if kind == "beta" else eta[data]
                denom = _coefficient_scale_for_primal_term(n, kind, data)
                c = float(coeff) / denom
                c *= row_scales[row] * row_scales[col]
                entries[row][col] = entries[row][col] + c * var

            B = scalar_bmat(entries)
            B = 0.5 * (B + B.T)
            if m_block == 1:
                constraints.append(B[0, 0] >= 0)
                scalar_inequality_count += 1
            else:
                constraints.append(B >> 0)
                psd_cone_sizes.append(m_block)

    objective = cp.sum([eta[key] for key in gamma_keys_sorted])
    problem = cp.Problem(cp.Maximize(objective), constraints)
    value = float(solve_problem(problem, solver, solver_kwargs, verbose))
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"symmetric primal value SDP failed with status {problem.status}")

    alpha_values = {k: None if v.value is None else float(v.value) for k, v in alpha.items()}
    eta_values = {key: None if v.value is None else float(v.value) for key, v in eta.items()}
    beta_values = {k: None if alpha_values[k] is None else alpha_values[k] / float(full_layer_size(n, k)) for k in alpha}
    gamma_values = {
        key: None if eta_values[key] is None else eta_values[key] / float(gamma_objective_multiplicity(n, *key))
        for key in eta
    }

    return {
        "value": value,
        "usual_adversary_value": value,
        "objective_convention": "usual_adversary_full_symmetric_sum",
        "status": str(problem.status),
        "alpha": alpha_values,
        "eta": eta_values,
        "beta": beta_values,
        "gamma": gamma_values,
        "allowed_layers": allowed,
        "block_sizes": final_block_sizes,
        "psd_cone_sizes": psd_cone_sizes,
        "scalar_inequality_count": scalar_inequality_count,
        "num_blocks": len(final_block_sizes),
        "num_psd_cones": len(psd_cone_sizes),
        "num_alpha_variables": len(alpha),
        "num_eta_variables": len(eta),
        "solver_stats": problem.solver_stats,
        "problem": problem,
    }


def solve_symmetric_trace_minimized_dual(
    n: int,
    layer_values: Mapping[int, Any],
    *,
    alpha: float = 1.0,
    adversary_value: float | None = None,
    epsilon: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> OrbitSolution:
    """
    Trace-minimize the orbit-reduced dual under an alpha-relaxed cost cap.

    Variables are orbit entries X_1[(alpha,A),(beta,B)]. PSD is imposed by
    Schrijver/Terwilliger blocks. ``adversary_value`` is phase-normalized
    (half the usual adversary bound); when omitted it is computed by a minimax
    pass. The per-input cap is

        alpha * adversary_value + epsilon.
    """
    require_cvxpy()
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    alpha = float(alpha)
    if alpha < 1.0:
        raise ValueError("alpha must be at least 1")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if adversary_value is not None and adversary_value <= 0:
        raise ValueError("adversary_value must be positive")
    N = n - 1
    lv = normalize_layer_values(layer_values)
    allowed = tuple(sorted(lv.keys()))
    allowed_set = set(allowed)
    if len({lv[s] for s in allowed}) < 2:
        raise ValueError("function/promise must contain both outputs")
    if min(allowed) < 0 or max(allowed) > n:
        raise ValueError("layer outside 0..n")

    keys = orbit_keys(N, allowed)
    key_set = set(keys)

    def xexpr(var_map: dict[OrbitKey, Any], key: OrbitKey):
        if key not in var_map:
            raise KeyError(f"invalid orbit key {key}")
        return var_map[key]

    def cost_expr(var_map: dict[OrbitKey, Any], alpha: int, a: int):
        expr = xexpr(var_map, (alpha, alpha, a, a, a))
        if a > 0:
            s = alpha + a - 1
            expr += a * xexpr(var_map, (1, 1, s, s, s))
        if N - a > 0:
            s = alpha + a
            expr += (N - a) * xexpr(var_map, (0, 0, s, s, s))
        return 0.5 * expr

    def equality_lhs(var_map: dict[OrbitKey, Any], alpha: int, beta: int, a: int, b: int, t: int):
        expr = 0
        if alpha != beta:
            expr += xexpr(var_map, (alpha, beta, a, b, t))
        if a - t > 0:
            expr += (a - t) * xexpr(var_map, (1, 0, alpha + a - 1, beta + b, alpha * beta + t))
        if b - t > 0:
            expr += (b - t) * xexpr(var_map, (0, 1, alpha + a, beta + b - 1, alpha * beta + t))
        return expr

    def build_psd_constraints(var_map: dict[OrbitKey, Any]):
        constraints: list[Any] = []
        for r in range(N // 2 + 1):
            labels = [
                (alpha, a)
                for alpha in (0, 1)
                for a in range(r, N - r + 1)
                if alpha + a in allowed_set
            ]
            if not labels:
                continue
            entries = [[0 for _ in labels] for __ in labels]
            has_term = False
            for row, (alpha, a) in enumerate(labels):
                for col, (beta, b) in enumerate(labels):
                    e = 0
                    for t in valid_intersections(a, b, N):
                        key = (alpha, beta, a, b, t)
                        if key not in key_set:
                            continue
                        coeff = terwilliger_coeff(N, a, b, r, t)
                        if coeff != 0.0:
                            e = e + coeff * xexpr(var_map, key)
                            has_term = True
                    entries[row][col] = e
            if has_term:
                B = scalar_bmat(entries)
                B = 0.5 * (B + B.T)
                constraints.append(B[0, 0] >= 0 if len(labels) == 1 else B >> 0)
        return constraints

    def solve_pass(mode: str, cap: float | None = None):
        var_map = {key: cp.Variable(name=f"x_{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}") for key in keys}
        constraints: list[Any] = []

        # Symmetry X[x,y] = X[y,x].
        for key in keys:
            alpha, beta, a, b, t = key
            kt = (beta, alpha, b, a, t)
            if kt in key_set and key <= kt:
                constraints.append(xexpr(var_map, key) == xexpr(var_map, kt))

        constraints.extend(build_psd_constraints(var_map))

        costs: list[tuple[int, int, Any]] = []
        for alpha in (0, 1):
            for a in range(N + 1):
                if alpha + a in allowed_set:
                    costs.append((alpha, a, cost_expr(var_map, alpha, a)))

        if mode == "max":
            T = cp.Variable(name="T")
            constraints.append(T >= 0)
            for _, _, c in costs:
                constraints.append(c <= T)
            objective = cp.Minimize(T)
        elif mode == "sum":
            if cap is None:
                raise ValueError("cap is required")
            for _, _, c in costs:
                constraints.append(c <= cap)
            weighted = 0
            for _, a, c in costs:
                weighted += binom_safe(N, a) * c
            objective = cp.Minimize(weighted)
        else:
            raise ValueError("mode must be 'max' or 'sum'")

        for alpha in (0, 1):
            for beta in (0, 1):
                for a in range(N + 1):
                    if alpha + a not in allowed_set:
                        continue
                    for b in range(N + 1):
                        if beta + b not in allowed_set:
                            continue
                        for t in valid_intersections(a, b, N):
                            rhs = 1.0 if lv[alpha + a] != lv[beta + b] else 0.0
                            constraints.append(equality_lhs(var_map, alpha, beta, a, b, t) == rhs)

        problem = cp.Problem(objective, constraints)
        value = float(solve_problem(problem, solver, solver_kwargs, verbose))
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"orbit SDP {mode} pass failed with status {problem.status}")
        values = {key: float(var.value) for key, var in var_map.items()}
        return value, values, str(problem.status), costs

    if adversary_value is None:
        first_value, _, first_status, _ = solve_pass("max")
    else:
        first_value = float(adversary_value)
        first_status = "provided"
    cap = alpha * first_value + epsilon
    trace_value, orbit_values, second_status, _ = solve_pass("sum", cap=cap)

    # Numerical block values and input costs.
    block_values: list[Array] = []
    block_labels: list[list[tuple[int, int]]] = []
    for r in range(N // 2 + 1):
        labels = [(alpha, a) for alpha in (0, 1) for a in range(r, N - r + 1) if alpha + a in allowed_set]
        if not labels:
            continue
        B = np.zeros((len(labels), len(labels)), dtype=float)
        for row, (alpha, a) in enumerate(labels):
            for col, (beta, b) in enumerate(labels):
                val = 0.0
                for t in valid_intersections(a, b, N):
                    key = (alpha, beta, a, b, t)
                    if key in orbit_values:
                        val += terwilliger_coeff(N, a, b, r, t) * orbit_values[key]
                B[row, col] = val
        block_values.append(0.5 * (B + B.T))
        block_labels.append(labels)

    input_costs_by_layer: dict[int, float] = {}
    for s in allowed:
        # Use representative alpha=0 if possible, else alpha=1.
        if s <= N:
            alpha, a = 0, s
        else:
            alpha, a = 1, s - 1
        # Direct numeric version of cost_expr.
        c = orbit_values[(alpha, alpha, a, a, a)]
        if a > 0:
            r0 = alpha + a - 1
            c += a * orbit_values[(1, 1, r0, r0, r0)]
        if N - a > 0:
            r0 = alpha + a
            c += (N - a) * orbit_values[(0, 0, r0, r0, r0)]
        input_costs_by_layer[s] = 0.5 * c

    return OrbitSolution(
        n=n,
        layer_values=lv,
        allowed_layers=allowed,
        objective_value=first_value,
        trace_value=trace_value,
        cap=cap,
        status=f"{second_status}; first_pass={first_status}",
        orbit_values=orbit_values,
        block_values=block_values,
        block_labels=block_labels,
        input_costs_by_layer=input_costs_by_layer,
    )


def solve_symmetric_orbit_sdp(
    n: int,
    layer_values: Mapping[int, Any],
    *,
    epsilon: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> OrbitSolution:
    """Solve and trace-minimize the symmetric dual at optimal query cost."""
    return solve_symmetric_trace_minimized_dual(
        n,
        layer_values,
        alpha=1.0,
        epsilon=epsilon,
        solver=solver,
        solver_kwargs=solver_kwargs,
        verbose=verbose,
    )


def solve_symmetric_fixed_rank_dual_feasibility(
    n: int,
    layer_values: Mapping[int, Any],
    *,
    block_rank: int,
    block_ranks: Sequence[int] | None = None,
    alpha: float = 1.0,
    adversary_value: float | None = None,
    epsilon: float = 1e-6,
    initial_solution: OrbitSolution | SymmetricFixedRankFeasibilityResult | None = None,
    restarts: int = 4,
    random_seed: int | None = 0,
    cap_penalties: Sequence[float] = (1.0, 100.0, 10000.0),
    max_nfev: int = 1000,
    feasibility_tol: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> SymmetricFixedRankFeasibilityResult:
    """
    Heuristically search the symmetric orbit dual with low-rank PSD blocks.

    Each Terwilliger block is factored as ``B_r = L_r L_r.T``. By default all
    blocks use ``rank(B_r) <= block_rank``. Passing ``block_ranks`` gives a
    per-block pattern, and rank 0 means that block is constrained to be zero.
    The orbit entries remain explicit variables; least-squares residuals
    enforce that every factored block equals the Terwilliger linear image of
    those orbit entries, together with the dual equality constraints and the
    alpha-relaxed cost cap.

    A feasible result is directly checkable in the orbit dual. A ``not_found``
    result is not an infeasibility certificate.
    """
    n = int(n)
    block_rank = int(block_rank)
    alpha = float(alpha)
    if n <= 0:
        raise ValueError("n must be positive")
    if block_rank <= 0:
        raise ValueError("block_rank must be positive")
    if alpha < 1.0:
        raise ValueError("alpha must be at least 1")
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if restarts < 0:
        raise ValueError("restarts must be non-negative")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")
    if feasibility_tol <= 0:
        raise ValueError("feasibility_tol must be positive")
    penalties = tuple(float(value) for value in cap_penalties)
    if not penalties or any(value <= 0 for value in penalties):
        raise ValueError("cap_penalties must contain positive values")

    lv = normalize_layer_values(layer_values)
    allowed = tuple(sorted(lv.keys()))
    allowed_set = set(allowed)
    if len({lv[s] for s in allowed}) < 2:
        raise ValueError("function/promise must contain both outputs")
    if min(allowed) < 0 or max(allowed) > n:
        raise ValueError("layer outside 0..n")
    if adversary_value is None:
        adversary_value = solve_symmetric_trace_minimized_dual(
            n,
            lv,
            alpha=1.0,
            epsilon=epsilon,
            solver=solver,
            solver_kwargs=solver_kwargs,
            verbose=verbose,
        ).objective_value
    adversary_value = float(adversary_value)
    if adversary_value <= 0:
        raise ValueError("adversary_value must be positive")
    cap = alpha * adversary_value + epsilon

    N = n - 1
    keys = orbit_keys(N, allowed)
    key_index = {key: idx for idx, key in enumerate(keys)}
    key_set = set(keys)

    block_specs: list[dict[str, Any]] = []
    raw_block_ranks: list[int] = []
    factor_offsets: list[int] = []
    factor_offset = len(keys)
    for r in range(N // 2 + 1):
        labels = [
            (eps, a)
            for eps in (0, 1)
            for a in range(r, N - r + 1)
            if eps + a in allowed_set
        ]
        if not labels:
            continue
        entries: list[tuple[int, int, list[tuple[int, float]]]] = []
        for row, (eps, a) in enumerate(labels):
            for col, (eta, b) in enumerate(labels):
                terms: list[tuple[int, float]] = []
                for t in valid_intersections(a, b, N):
                    key = (eps, eta, a, b, t)
                    if key not in key_set:
                        continue
                    coeff = terwilliger_coeff(N, a, b, r, t)
                    if coeff != 0.0:
                        terms.append((key_index[key], float(coeff)))
                if terms:
                    entries.append((row, col, terms))
        if entries:
            block_specs.append({"r": r, "labels": labels, "entries": entries})
            raw_block_ranks.append(block_rank)
    if block_ranks is not None:
        raw_block_ranks = [int(value) for value in block_ranks]
        if len(raw_block_ranks) != len(block_specs):
            raise ValueError(
                "block_ranks must contain one entry per nonempty Terwilliger block"
            )
        if any(value < 0 for value in raw_block_ranks):
            raise ValueError("block_ranks entries must be non-negative")
    for spec, rank_bound in zip(block_specs, raw_block_ranks):
            factor_offsets.append(factor_offset)
            factor_offset += len(spec["labels"]) * rank_bound
    variable_size = factor_offset

    block_multiplicities = [
        binom_safe(N, int(spec["r"])) - binom_safe(N, int(spec["r"]) - 1)
        for spec in block_specs
    ]
    full_rank_bound = int(
        sum(rank_bound * mult for rank_bound, mult in zip(raw_block_ranks, block_multiplicities))
    )

    equality_terms: list[tuple[list[tuple[int, float]], float]] = []
    for eps in (0, 1):
        for eta in (0, 1):
            for a in range(N + 1):
                if eps + a not in allowed_set:
                    continue
                for b in range(N + 1):
                    if eta + b not in allowed_set:
                        continue
                    for t in valid_intersections(a, b, N):
                        terms: list[tuple[int, float]] = []
                        if eps != eta:
                            terms.append((key_index[(eps, eta, a, b, t)], 1.0))
                        if a - t > 0:
                            key = (1, 0, eps + a - 1, eta + b, eps * eta + t)
                            terms.append((key_index[key], float(a - t)))
                        if b - t > 0:
                            key = (0, 1, eps + a, eta + b - 1, eps * eta + t)
                            terms.append((key_index[key], float(b - t)))
                        rhs = 1.0 if lv[eps + a] != lv[eta + b] else 0.0
                        equality_terms.append((terms, rhs))

    symmetry_pairs = [
        (idx, key_index[(eta, eps, b, a, t)])
        for idx, (eps, eta, a, b, t) in enumerate(keys)
        if (eta, eps, b, a, t) in key_set and idx < key_index[(eta, eps, b, a, t)]
    ]

    cost_terms: list[tuple[int, int, list[tuple[int, float]]]] = []
    for s in allowed:
        if s <= N:
            eps, a = 0, s
        else:
            eps, a = 1, s - 1
        terms = [(key_index[(eps, eps, a, a, a)], 0.5)]
        if a > 0:
            r0 = eps + a - 1
            terms.append((key_index[(1, 1, r0, r0, r0)], 0.5 * a))
        if N - a > 0:
            r0 = eps + a
            terms.append((key_index[(0, 0, r0, r0, r0)], 0.5 * (N - a)))
        cost_terms.append((s, binom_safe(N, a), terms))

    def linear_value(values: Array, terms: Sequence[tuple[int, float]]) -> float:
        return float(sum(coeff * values[idx] for idx, coeff in terms))

    def unpack(vector: Array) -> tuple[Array, list[Array]]:
        orbit = np.asarray(vector[: len(keys)], dtype=float)
        blocks = []
        for spec, rank_bound, offset in zip(block_specs, raw_block_ranks, factor_offsets):
            size = len(spec["labels"])
            if rank_bound == 0:
                blocks.append(np.zeros((size, size), dtype=float))
            else:
                factors = np.asarray(vector[offset: offset + size * rank_bound], dtype=float)
                factors = factors.reshape(size, rank_bound)
                blocks.append(factors @ factors.T)
        return orbit, blocks

    def evaluate(vector: Array) -> tuple[Array, list[Array], Array, Array, Array, dict[int, float]]:
        orbit, blocks = unpack(vector)
        block_residuals: list[float] = []
        for spec, block in zip(block_specs, blocks):
            for row, col, terms in spec["entries"]:
                block_residuals.append(block[row, col] - linear_value(orbit, terms))
        equalities = np.array(
            [linear_value(orbit, terms) - rhs for terms, rhs in equality_terms],
            dtype=float,
        )
        symmetries = np.array([orbit[i] - orbit[j] for i, j in symmetry_pairs], dtype=float)
        costs_by_layer = {
            s: linear_value(orbit, terms)
            for s, _, terms in cost_terms
        }
        costs = np.array(list(costs_by_layer.values()), dtype=float)
        return orbit, blocks, np.array(block_residuals, dtype=float), equalities, symmetries, costs_by_layer

    def residual(vector: Array, penalty: float) -> Array:
        _, _, block_residuals, equalities, symmetries, costs_by_layer = evaluate(vector)
        costs = np.array(list(costs_by_layer.values()), dtype=float)
        cap_residuals = np.sqrt(penalty) * np.maximum(costs - cap, 0.0)
        return np.concatenate([block_residuals, equalities, symmetries, cap_residuals])

    rng = np.random.default_rng(random_seed)
    starts: list[Array] = []
    if initial_solution is not None:
        if initial_solution.n != n or initial_solution.layer_values != lv:
            raise ValueError("initial_solution does not match n and layer_values")
        start = np.zeros(variable_size, dtype=float)
        for key, value in initial_solution.orbit_values.items():
            if key in key_index:
                start[key_index[key]] = float(value)
        blocks = initial_solution.block_values
        if len(blocks) != len(block_specs):
            raise ValueError("initial_solution has incompatible block structure")
        for block, spec, rank_bound, offset in zip(blocks, block_specs, raw_block_ranks, factor_offsets):
            if rank_bound == 0:
                continue
            block = 0.5 * (np.asarray(block, dtype=float) + np.asarray(block, dtype=float).T)
            factor = psd_factor(block, tol=1e-10)
            old_rank = factor.shape[1]
            if old_rank == rank_bound:
                projected = factor
            elif old_rank < rank_bound:
                projected = np.pad(factor, ((0, 0), (0, rank_bound - old_rank)))
            else:
                projection = rng.normal(size=(old_rank, rank_bound)) / sqrt(float(rank_bound))
                projected = factor @ projection
            size = len(spec["labels"])
            start[offset: offset + size * rank_bound] = projected.reshape(-1)
        starts.append(start)
    for _ in range(restarts):
        start = rng.normal(scale=1.0 / sqrt(float(max(n, 1))), size=variable_size)
        starts.append(start)
    if not starts:
        raise ValueError("provide initial_solution or request at least one restart")

    best: tuple[float, float, float, float, float, Array, list[Array], dict[int, float]] | None = None
    total_nfev = 0
    for attempt, start in enumerate(starts, start=1):
        vector = start
        for penalty in penalties:
            result = opt.least_squares(
                lambda value, weight=penalty: residual(value, weight),
                vector,
                max_nfev=max_nfev,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            vector = result.x
            total_nfev += int(result.nfev)
        orbit, blocks, block_residuals, equalities, symmetries, costs_by_layer = evaluate(vector)
        max_block = float(np.max(np.abs(block_residuals))) if block_residuals.size else 0.0
        eq_parts = [float(np.max(np.abs(equalities))) if equalities.size else 0.0]
        if symmetries.size:
            eq_parts.append(float(np.max(np.abs(symmetries))))
        max_equality = max(eq_parts)
        max_cap_violation = max(0.0, max(float(c - cap) for c in costs_by_layer.values()))
        score = max(max_block, max_equality, max_cap_violation)
        residual_norm = float(la.norm(residual(vector, penalties[-1])))
        candidate = (
            score,
            max_equality,
            max_block,
            max_cap_violation,
            residual_norm,
            orbit.copy(),
            [block.copy() for block in blocks],
            dict(costs_by_layer),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
        if verbose:
            print(
                f"symmetric fixed-rank attempt {attempt}/{len(starts)}: "
                f"max equality={max_equality:.3e}, "
                f"block={max_block:.3e}, cap={max_cap_violation:.3e}"
            )
        if score <= feasibility_tol:
            break

    assert best is not None
    _, max_equality, max_block, max_cap_violation, residual_norm, orbit, blocks, costs_by_layer = best
    feasible = max(max_equality, max_block, max_cap_violation) <= feasibility_tol
    orbit_values = {key: float(orbit[idx]) for key, idx in key_index.items()}
    block_labels = [list(spec["labels"]) for spec in block_specs]
    trace_value = float(
        sum(weight * linear_value(orbit, terms) for _, weight, terms in cost_terms)
    )
    return SymmetricFixedRankFeasibilityResult(
        n=n,
        layer_values=lv,
        allowed_layers=allowed,
        block_rank_bound=block_rank,
        full_rank_bound=full_rank_bound,
        block_multiplicities=block_multiplicities,
        alpha=alpha,
        adversary_value=adversary_value,
        cap=cap,
        feasible=feasible,
        status="feasible" if feasible else "not_found",
        orbit_values=orbit_values,
        block_values=blocks,
        block_labels=block_labels,
        input_costs_by_layer=costs_by_layer,
        trace_value=trace_value,
        max_equality_residual=max_equality,
        max_block_residual=max_block,
        max_cap_violation=max_cap_violation,
        residual_norm=residual_norm,
        attempts=attempt,
        total_nfev=total_nfev,
    )


def expand_orbit_solution(orbit: OrbitSolution, *, psd_tol: float = 1e-8) -> AdversarySolution:
    """Expand a compact orbit solution into full Boolean input-basis matrices."""
    n = orbit.n
    allowed = set(orbit.allowed_layers)
    xs = [x for x in all_binary_inputs(n) if hamming_weight(x) in allowed]
    ys = [orbit.layer_values[hamming_weight(x)] for x in xs]
    m = len(xs)

    grams: list[Array] = []
    z_vecs: list[Array] = []
    witness_dims: list[int] = []

    for coord in range(n):
        rest = [j for j in range(n) if j != coord]
        G = np.zeros((m, m), dtype=float)
        for a_idx, x in enumerate(xs):
            alpha = x[coord]
            A = {j for j in rest if x[j] == 1}
            for b_idx, y in enumerate(xs):
                beta = y[coord]
                B = {j for j in rest if y[j] == 1}
                key = (alpha, beta, len(A), len(B), len(A & B))
                G[a_idx, b_idx] = orbit.orbit_values[key]
        G = 0.5 * (G + G.T)
        R = psd_factor(G, tol=psd_tol)
        grams.append(G)
        z_vecs.append(R)
        witness_dims.append(int(R.shape[1]))

    input_costs = []
    for x_idx in range(m):
        input_costs.append(0.5 * sum(float(grams[i][x_idx, x_idx]) for i in range(n)))

    public_dim = max(max(ys) + 1, 1)
    return AdversarySolution(
        inputs=xs,
        outputs=ys,
        objective_value=orbit.objective_value,
        trace_value=orbit.trace_value,
        status=f"expanded from orbit; {orbit.status}",
        z=z_vecs,
        grams=grams,
        input_costs=np.array(input_costs, dtype=float),
        public_dim=public_dim,
        witness_dims=witness_dims,
        private_dim=int(sum(witness_dims)),
        orbit_solution=orbit,
    )


def numerical_psd_rank(M: Array, *, relative_tol: float = 1e-7, absolute_tol: float = 1e-8) -> int:
    """Numerical rank of a symmetric PSD matrix from its positive eigenvalues."""
    if relative_tol < 0 or absolute_tol < 0:
        raise ValueError("rank tolerances must be non-negative")
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    eigenvalues = la.eigvalsh(M)
    largest = max(0.0, float(eigenvalues[-1])) if eigenvalues.size else 0.0
    cutoff = max(float(absolute_tol), float(relative_tol) * largest)
    return int(np.count_nonzero(eigenvalues > cutoff))


def symmetric_witness_rank(
    orbit: OrbitSolution,
    *,
    relative_tol: float = 1e-7,
    absolute_tol: float = 1e-8,
) -> dict[str, Any]:
    """
    Compute the full-basis rank of any coordinate witness matrix X_i.

    Under the stabilizer of coordinate ``i``, Terwilliger block ``r`` occurs
    with multiplicity C(n-1,r)-C(n-1,r-1). Therefore

        rank(X_i) = sum_r multiplicity(r) * rank(block_r).

    This gives the same rank as expanding an exponentially large ``X_i`` but
    only diagonalizes the compact orbit blocks.
    """
    N = orbit.n - 1
    block_ranks: list[int] = []
    block_multiplicities: list[int] = []
    rank = 0
    for r, block in enumerate(orbit.block_values):
        block_rank = numerical_psd_rank(
            block,
            relative_tol=relative_tol,
            absolute_tol=absolute_tol,
        )
        multiplicity = binom_safe(N, r) - binom_safe(N, r - 1)
        block_ranks.append(block_rank)
        block_multiplicities.append(multiplicity)
        rank += multiplicity * block_rank
    return {
        "rank_X_i": int(rank),
        "total_witness_dimension": int(orbit.n * rank),
        "block_ranks": block_ranks,
        "block_multiplicities": block_multiplicities,
    }


def solve_symmetric_adversary_sdp(
    n: int,
    layer_values: Mapping[int, Any],
    *,
    epsilon: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    psd_tol: float = 1e-8,
    verbose: bool = False,
) -> AdversarySolution:
    """Convenience wrapper: solve the symmetric orbit adversary SDP and expand to full basis."""
    orbit = solve_symmetric_orbit_sdp(
        n,
        layer_values,
        epsilon=epsilon,
        solver=solver,
        solver_kwargs=solver_kwargs,
        verbose=verbose,
    )
    return expand_orbit_solution(orbit, psd_tol=psd_tol)


# ---------------------------------------------------------------------------
# Adversary transducer extraction and simulation
# ---------------------------------------------------------------------------

def build_adversary_state_matrices(solution: AdversarySolution) -> tuple[Array, Array, int, dict[int, slice]]:
    xs, ys = solution.inputs, solution.outputs
    m = len(xs)
    n = len(xs[0])
    public_dim = solution.public_dim

    private_slices: dict[int, slice] = {}
    offset = 0
    for i, dim in enumerate(solution.witness_dims):
        private_slices[i] = slice(offset, offset + dim)
        offset += dim
    private_dim = offset
    total_dim = public_dim + private_dim

    source = np.zeros((total_dim, m), dtype=float)
    target = np.zeros((total_dim, m), dtype=float)
    for x_idx, x in enumerate(xs):
        source[0, x_idx] = 1.0
        target[ys[x_idx], x_idx] = 1.0
        for i in range(n):
            piece = solution.z[i][x_idx, :] / np.sqrt(2.0)
            sl = private_slices[i]
            full = slice(public_dim + sl.start, public_dim + sl.stop)
            sign = -1.0 if x[i] else 1.0
            source[full, x_idx] += sign * piece
            target[full, x_idx] += piece
    return source, target, public_dim, private_slices


def build_adversary_transducer(
    solution: AdversarySolution,
    *,
    gram_tol: float = 1e-7,
    eig_tol: float = 1e-9,
) -> AdversaryTransducer:
    source, target, public_dim, private_slices = build_adversary_state_matrices(solution)
    U, gram_error, map_error, unitarity_error = unitary_from_state_pairs(
        source,
        target,
        gram_tol=gram_tol,
        eig_tol=eig_tol,
    )
    catalyst_norms = np.sum(np.abs(target[public_dim:, :]) ** 2, axis=0)
    return AdversaryTransducer(
        U=U,
        source_states=source,
        target_states=target,
        inputs=solution.inputs,
        outputs=solution.outputs,
        public_dim=public_dim,
        private_dim=source.shape[0] - public_dim,
        witness_dims=solution.witness_dims,
        private_slices=private_slices,
        catalyst_norms=catalyst_norms,
        gram_error=gram_error,
        map_error=map_error,
        unitarity_error=unitarity_error,
    )


def adversary_query(transducer: AdversaryTransducer, x: Sequence[int]) -> Array:
    """Return Q_x = I_H ⊕ ⊕_i (-1)^{x_i} I_{W_i}."""
    x = tuple(int(b) for b in x)
    h, l = transducer.public_dim, transducer.private_dim
    Q = np.eye(h + l, dtype=complex)
    for i, bit in enumerate(x):
        if i not in transducer.private_slices:
            continue
        sl = transducer.private_slices[i]
        full = slice(h + sl.start, h + sl.stop)
        if bit == 1:
            Q[full, full] *= -1.0
    return Q


def adversary_step(transducer: AdversaryTransducer, x: Sequence[int]) -> Array:
    """Return S_x = U Q_x."""
    return transducer.U @ adversary_query(transducer, x)


def catalyst_norm_bound(transducer: AdversaryTransducer) -> float:
    return float(np.max(transducer.catalyst_norms)) if transducer.catalyst_norms.size else 0.0


def choose_repetitions(W: float, epsilon: float, safety_factor: float = 4.0) -> int:
    if W < -1e-12:
        raise ValueError("W must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return max(1, int(np.ceil(safety_factor * max(0.0, W) / (epsilon**2))))


def public_spread(public_dim: int, private_dim: int, K: int) -> Array:
    if K <= 0:
        raise ValueError("K must be positive")
    omega = np.exp(2j * np.pi / K)
    F = np.array([[omega ** (a * b) for b in range(K)] for a in range(K)], dtype=complex) / np.sqrt(K)
    return la.block_diag(np.kron(F, np.eye(public_dim, dtype=complex)), np.eye(private_dim, dtype=complex))


def lift_step_to_slot(S: Array, public_dim: int, private_dim: int, K: int, slot: int) -> Array:
    if not 0 <= slot < K:
        raise ValueError("slot out of range")
    h, l = public_dim, private_dim
    if S.shape != (h + l, h + l):
        raise ValueError("S has incompatible shape")
    N = K * h + l
    out = np.eye(N, dtype=complex)
    ps = slice(slot * h, (slot + 1) * h)
    ls = slice(K * h, K * h + l)
    out[ps, ps] = S[:h, :h]
    out[ps, ls] = S[:h, h:]
    out[ls, ps] = S[h:, :h]
    out[ls, ls] = S[h:, h:]
    return out


def repeated_algorithm_unitary(S: Array, public_dim: int, private_dim: int, K: int) -> Array:
    spread = public_spread(public_dim, private_dim, K)
    A = spread.copy()
    for slot in range(K):
        A = lift_step_to_slot(S, public_dim, private_dim, K, slot) @ A
    return spread.conj().T @ A


def run_adversary_algorithm(
    transducer: AdversaryTransducer,
    x: Sequence[int],
    K: int,
    initial_public_state: Array | None = None,
) -> dict[str, Array | float]:
    h, l = transducer.public_dim, transducer.private_dim
    if initial_public_state is None:
        psi = np.zeros(h, dtype=complex)
        psi[0] = 1.0
    else:
        psi = np.asarray(initial_public_state, dtype=complex)
        if psi.shape != (h,):
            raise ValueError("initial_public_state has incompatible shape")
    Sx = adversary_step(transducer, x)
    A = repeated_algorithm_unitary(Sx, h, l, K)
    init = np.zeros(K * h + l, dtype=complex)
    init[:h] = psi
    final = A @ init
    output = final[:h]
    return {
        "algorithm_unitary": A,
        "final_state": final,
        "output_slot": output,
        "output_probabilities": np.abs(output) ** 2,
        "garbage_norm": float(la.norm(final[h:])),
    }


# ---------------------------------------------------------------------------
# Optional Qiskit export
# ---------------------------------------------------------------------------

def next_power_of_two(d: int) -> int:
    if d <= 0:
        raise ValueError("dimension must be positive")
    return 1 << int(np.ceil(np.log2(d)))


def qiskit_adversary_circuit(transducer: AdversaryTransducer, x: Sequence[int], K: int, *, measure_all: bool = False):
    """Build a dense-unitary Qiskit circuit for the repeated adversary-transducer algorithm."""
    from qiskit import QuantumCircuit

    h, l = transducer.public_dim, transducer.private_dim
    Sx = adversary_step(transducer, x)
    logical_dim = K * h + l
    padded_dim = next_power_of_two(logical_dim)
    num_qubits = int(np.log2(padded_dim))
    qc = QuantumCircuit(num_qubits)
    qubits = list(range(num_qubits))

    def append_logical(M: Array, label: str):
        padded = np.eye(padded_dim, dtype=complex)
        padded[:logical_dim, :logical_dim] = M
        qc.unitary(padded, qubits, label=label)

    spread = public_spread(h, l, K)
    append_logical(spread, "Spread")
    for slot in range(K):
        append_logical(lift_step_to_slot(Sx, h, l, K, slot), f"S[{slot}]")
    append_logical(spread.conj().T, "Unspread")
    if measure_all:
        qc.measure_all()
    metadata = {
        "logical_dim": logical_dim,
        "padded_dim": padded_dim,
        "num_qubits": num_qubits,
        "public_dim": h,
        "private_dim": l,
        "K": K,
    }
    return qc, metadata


def decode_statevector(statevector: Any, *, public_dim: int, logical_dim: int) -> dict[str, Array | float]:
    data = np.asarray(statevector.data, dtype=complex)
    probs = np.abs(data) ** 2
    return {
        "output_probabilities": probs[:public_dim],
        "garbage_probability": float(np.sum(probs[public_dim:logical_dim])),
        "padding_probability": float(np.sum(probs[logical_dim:])),
        "total_probability": float(np.sum(probs)),
    }


# ---------------------------------------------------------------------------
# Diagnostics and analytic threshold helpers
# ---------------------------------------------------------------------------

def show_catalysts(solution: AdversarySolution, *, decimals: int = 8) -> None:
    """Print w_{x,i}=z_{x,i}/sqrt(2) and ||w_x||^2 for each input."""
    for x_idx, x in enumerate(solution.inputs):
        parts = []
        norm_sq = 0.0
        print(f"x={x}, f(x)={solution.outputs[x_idx]}")
        for i, Z in enumerate(solution.z):
            w = Z[x_idx, :] / np.sqrt(2.0)
            parts.append(w)
            norm_sq += float(np.vdot(w, w).real)
            print(f"  i={i}: {np.array2string(w, precision=decimals, suppress_small=True)}")
        full = np.concatenate(parts) if parts else np.array([])
        print(f"  ||w_x||^2 = {norm_sq:.{decimals}g}")
        print(f"  full = {np.array2string(full, precision=decimals, suppress_small=True)}\n")


def show_block_factors(orbit: OrbitSolution, *, tol: float = 1e-8, decimals: int = 8) -> dict[int, dict[str, Any]]:
    """Print numerical rank-one factors of Schrijver blocks."""
    report: dict[int, dict[str, Any]] = {}
    for r, B in enumerate(orbit.block_values):
        B = 0.5 * (np.asarray(B, dtype=float) + np.asarray(B, dtype=float).T)
        vals, vecs = la.eigh(B)
        scale = max(1.0, float(np.max(np.abs(vals))) if vals.size else 1.0)
        keep = vals > tol * scale
        factors = []
        for lam, q in zip(vals[keep], vecs[:, keep].T):
            eta = np.sqrt(max(float(lam), 0.0)) * q
            if eta.size:
                pivot = int(np.argmax(np.abs(eta)))
                if eta[pivot] < 0:
                    eta = -eta
            factors.append(eta)
        labels = orbit.block_labels[r]
        print(f"r={r}, shape={B.shape}, rank={len(factors)}, labels={labels}")
        for j, eta in enumerate(factors):
            print(f"  eta[{j}]:")
            for lab, val in zip(labels, eta):
                print(f"    {lab}: {val:.{decimals}g}")
        report[r] = {"rank": len(factors), "eigenvalues": vals, "labels": labels, "factors": factors}
    return report


def threshold_adversary_value(n: int, k: int) -> float:
    """Usual adversary value for THRESHOLD^k_n."""
    return sqrt(k * (n - k + 1))


def threshold_gamma(n: int, k: int) -> float:
    """Fourth-root scale used in the analytic threshold eta formulas."""
    return (k * (n - k + 1)) ** 0.25


def threshold_P_entry(n: int, k: int, r: int, a: int, b: int) -> float:
    """Closed form for P_r(a,b) in the threshold cross-kernel expansion."""
    N = n - 1
    if not (0 <= r <= N // 2 and r <= a <= N - r and r <= b <= N - r):
        return 0.0
    numerator = factorial(r) * factorial(N - 2 * r) * binom_safe(N - r + 1, a - r)
    denominator = (
        factorial(b + 1)
        * factorial(N - r - b)
        * sqrt(binom_safe(N - 2 * r, a - r) * binom_safe(N - 2 * r, b - r))
    )
    return numerator / denominator


def threshold_eta_blocks(n: int, k: int) -> dict[int, dict[str, Any]]:
    """Analytic rank-one Schrijver block factors for THRESHOLD^k_n."""
    N = n - 1
    d = min(k - 1, n - k)
    Gamma = threshold_gamma(n, k)
    out: dict[int, dict[str, Any]] = {}
    for r in range(d + 1):
        labels = [(alpha, a) for alpha in (0, 1) for a in range(r, N - r + 1)]
        C = threshold_P_entry(n, k, r, k - 1, k - 1)
        eta = []
        for alpha, a in labels:
            if alpha == 0 and a <= k - 1:
                val = sqrt(k / C) / Gamma * threshold_P_entry(n, k, r, a, k - 1)
            elif alpha == 1 and a >= k - 1:
                val = Gamma / sqrt(k * C) * threshold_P_entry(n, k, r, k - 1, a)
            else:
                val = 0.0
            eta.append(val)
        eta_arr = np.array(eta, dtype=float)
        out[r] = {"labels": labels, "eta": eta_arr, "B": np.outer(eta_arr, eta_arr)}
    return out


def analytic_or_solution(n: int) -> AdversarySolution:
    """Closed-form rank-one adversary solution for OR_n, useful for tests/examples."""
    n = int(n)
    xs = all_binary_inputs(n)
    ys = [int(any(x)) for x in xs]
    gamma = n ** 0.25
    m = len(xs)
    z_vecs: list[Array] = []
    grams: list[Array] = []
    for i in range(n):
        z = np.zeros((m, 1), dtype=float)
        for idx, x in enumerate(xs):
            s = hamming_weight(x)
            if s == 0:
                z[idx, 0] = 1.0 / gamma
            elif x[i] == 1:
                z[idx, 0] = gamma / s
        z_vecs.append(z)
        grams.append(z @ z.T)
    costs = np.array([0.5 * sum(G[idx, idx] for G in grams) for idx in range(m)], dtype=float)
    return AdversarySolution(
        inputs=xs,
        outputs=ys,
        objective_value=float(np.max(costs)),
        trace_value=float(np.sum(costs)),
        status="analytic_or_solution",
        z=z_vecs,
        grams=grams,
        input_costs=costs,
        public_dim=2,
        witness_dims=[1] * n,
        private_dim=n,
        orbit_solution=None,
    )
