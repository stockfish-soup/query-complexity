"""
query_complexity.py

A compact phase-oracle implementation for adversary-based state-conversion
transducers.

Kept from the old codebase:
  - one-vector phase adversary SDP for arbitrary Boolean functions;
  - Terwilliger/orbit-reduced phase SDP for symmetric Boolean functions;
  - phase transducer extraction;
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
class PhaseSolution:
    """Explicit one-vector phase solution in the Boolean input basis."""

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
class PhaseTransducer:
    """Input-independent phase transducer unitary."""

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
    if "CLARABEL" in installed:
        return "CLARABEL"
    if "SDPA" in installed:
        return cp.SDPA
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
# General phase SDP
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


def solve_phase_sdp(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any] | None = None,
    f: Callable[[tuple[int, ...]], Any] | None = None,
    *,
    epsilon: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    psd_tol: float = 1e-8,
    verbose: bool = False,
) -> PhaseSolution:
    """
    Solve the one-vector phase adversary SDP and factor the Gram matrices.

    First pass computes the minimax value. Second pass minimizes the sum of
    input costs under the cap value + epsilon to encourage lower-rank factors.
    """
    require_cvxpy()
    xs, ys = normalize_instance(inputs, outputs, f)
    m = len(xs)
    n = len(xs[0])

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
            objective = cp.Minimize(cp.sum(costs))
        else:
            raise ValueError("mode must be 'max' or 'sum'")

        for a, x in enumerate(xs):
            for b, y in enumerate(xs):
                terms = [X[i][a, b] for i in range(n) if x[i] != y[i]]
                lhs = cp.sum(terms) if terms else 0
                rhs = 1.0 if ys[a] != ys[b] else 0.0
                constraints.append(lhs == rhs)

        return cp.Problem(objective, constraints), X, costs

    first_problem, _, _ = build_problem("max")
    first_value = float(solve_problem(first_problem, solver, solver_kwargs, verbose))
    if first_problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"first pass failed with status {first_problem.status}")

    cap = first_value + epsilon
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

    return PhaseSolution(
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


def solve_symmetric_orbit_sdp(
    n: int,
    layer_values: Mapping[int, Any],
    *,
    epsilon: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    verbose: bool = False,
) -> OrbitSolution:
    """
    Solve the orbit-reduced phase SDP for a symmetric Boolean function.

    Variables are orbit entries X_1[(alpha,A),(beta,B)]. PSD is imposed by
    Schrijver/Terwilliger blocks.
    """
    require_cvxpy()
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
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

    first_value, _, first_status, _ = solve_pass("max")
    cap = first_value + epsilon
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


def expand_orbit_solution(orbit: OrbitSolution, *, psd_tol: float = 1e-8) -> PhaseSolution:
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
    return PhaseSolution(
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


def solve_symmetric_phase_sdp(
    n: int,
    layer_values: Mapping[int, Any],
    *,
    epsilon: float = 1e-6,
    solver: str | None = None,
    solver_kwargs: Mapping[str, Any] | None = None,
    psd_tol: float = 1e-8,
    verbose: bool = False,
) -> PhaseSolution:
    """Convenience wrapper: solve the symmetric orbit SDP and expand to full basis."""
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
# Phase transducer extraction and simulation
# ---------------------------------------------------------------------------

def build_phase_state_matrices(solution: PhaseSolution) -> tuple[Array, Array, int, dict[int, slice]]:
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
            phase = -1.0 if x[i] else 1.0
            source[full, x_idx] += phase * piece
            target[full, x_idx] += piece
    return source, target, public_dim, private_slices


def build_phase_transducer(
    solution: PhaseSolution,
    *,
    gram_tol: float = 1e-7,
    eig_tol: float = 1e-9,
) -> PhaseTransducer:
    source, target, public_dim, private_slices = build_phase_state_matrices(solution)
    U, gram_error, map_error, unitarity_error = unitary_from_state_pairs(
        source,
        target,
        gram_tol=gram_tol,
        eig_tol=eig_tol,
    )
    catalyst_norms = np.sum(np.abs(target[public_dim:, :]) ** 2, axis=0)
    return PhaseTransducer(
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


def phase_query(transducer: PhaseTransducer, x: Sequence[int]) -> Array:
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


def phase_step(transducer: PhaseTransducer, x: Sequence[int]) -> Array:
    """Return S_x = U Q_x."""
    return transducer.U @ phase_query(transducer, x)


def catalyst_norm_bound(transducer: PhaseTransducer) -> float:
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


def run_phase_algorithm(
    transducer: PhaseTransducer,
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
    Sx = phase_step(transducer, x)
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


def qiskit_phase_circuit(transducer: PhaseTransducer, x: Sequence[int], K: int, *, measure_all: bool = False):
    """Build a dense-unitary Qiskit circuit for the repeated phase algorithm."""
    from qiskit import QuantumCircuit

    h, l = transducer.public_dim, transducer.private_dim
    Sx = phase_step(transducer, x)
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

def show_catalysts(solution: PhaseSolution, *, decimals: int = 8) -> None:
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


def analytic_or_solution(n: int) -> PhaseSolution:
    """Closed-form rank-one phase solution for OR_n, useful for tests/examples."""
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
    return PhaseSolution(
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
