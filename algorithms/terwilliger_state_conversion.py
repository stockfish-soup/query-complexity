"""
terwilliger_state_conversion.py

Minimal Terwilliger/orbit-reduced state-conversion adversary SDP for symmetric
Boolean functions, with transducer extraction and repeated-algorithm simulation.

This file implements the simplified phase-oracle, one-vector state-conversion SDP
for function evaluation:

    minimise    max_x  1/2 * sum_i ||z_{x,i}||^2

    subject to  1[f(x) != f(y)] = sum_{i: x_i != y_i} <z_{x,i}, z_{y,i}>
                X_i[x,y] = <z_{x,i}, z_{y,i}>,  X_i >= 0.

For symmetric functions, it stores only orbit variables for X_0.  An input is
written as x=(alpha,A), where alpha is the distinguished bit and A is a subset
of the remaining N=n-1 coordinates.  X_0 depends only on

    (alpha, beta, |A|, |B|, |A cap B|).

PSD of X_0 is imposed with Schrijver's explicit Terwilliger block
diagonalisation of the Hamming cube.

The solver performs two passes:
  1. minimise the worst-case catalyst norm;
  2. minimise the weighted trace/sum of catalyst norms under a per-input cap.

For speed, the explicit witnesses and transducer are constructed only on one
canonical representative per promised Hamming layer.  Thus sol.inputs are
Hamming-weight labels like (0,), (1,), ..., and sol.query_inputs are the actual
canonical bit strings used for phase queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, sqrt
from numbers import Number
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import scipy.linalg as la

from adversary_transducer import *
from qiskit_algorithm import *

try:
    import cvxpy as cp
except Exception as exc:  # pragma: no cover
    cp = None
    _CVXPY_IMPORT_ERROR = exc
else:
    _CVXPY_IMPORT_ERROR = None


Array = np.ndarray
LayerValue = dict[int, int]
OrbitKey = tuple[int, int, int, int, int]  # (alpha, beta, a, b, t)


# ============================================================
# Dataclasses
# ============================================================

@dataclass(frozen=True)
class TerwilligerPhaseAdversarySolution:
    # Effective inputs are Hamming weights, stored as one-tuples.
    inputs: list[tuple[int, ...]]
    # Canonical bit-string representatives of those Hamming weights.
    query_inputs: list[tuple[int, ...]]
    outputs: list[int]

    n: int
    allowed_layers: tuple[int, ...]
    layer_value: dict[int, int]

    objective_value: float      # weighted second-pass objective, sum_x W_x
    adversary_value: float      # first-pass minimax objective
    adversary_cap: float        # cap used in second pass
    status: str

    # z[i][layer_idx, :] is z_{x,i} in W_i for the canonical representative x.
    # w[i] is the actual catalyst block z[i]/sqrt(2).
    z: list[Array]
    w: list[Array]

    # Layer-representative Gram matrices X_i.
    grams: list[Array]

    # Raw orbit values for X_0.
    orbit_values: dict[OrbitKey, float]

    # Numerical Terwilliger block values for diagnostics.
    block_values: list[Array]

    public_dim: int
    witness_dims: list[int]
    private_dim: int
    solver_stats: Any | None = None


@dataclass(frozen=True)
class TerwilligerPhaseTransducerUnitary:
    U: Array
    source_states: Array        # columns: |0> plus after-query catalyst
    target_states: Array        # columns: |f(x)> plus before-query catalyst

    inputs: list[tuple[int, ...]]
    query_inputs: list[tuple[int, ...]]
    outputs: list[int]

    z: list[Array]
    w: list[Array]

    public_dim: int
    witness_dims: list[int]
    private_dim: int

    catalyst_norms: Array
    adversary_value: float

    gram_error: float
    map_error: float
    unitarity_error: float


# ============================================================
# Basic helpers
# ============================================================

def _require_cvxpy() -> None:
    if cp is None:  # pragma: no cover
        raise ImportError(
            "cvxpy is required for the SDP solver. Install it with `pip install cvxpy`."
        ) from _CVXPY_IMPORT_ERROR


def normalize_input_bits(x: Any) -> tuple[int, ...]:
    """Convert an input to a 0/1 tuple."""
    if isinstance(x, str):
        bits = tuple(int(c) for c in x)
        if not set(bits).issubset({0, 1}):
            raise ValueError(f"String input {x!r} is not binary.")
        return bits

    x_tuple = tuple(int(v) for v in x)
    vals = set(x_tuple)

    if vals.issubset({0, 1}):
        return x_tuple
    if vals.issubset({-1, 1}):
        return tuple(1 if v == 1 else 0 for v in x_tuple)

    raise ValueError(f"Input {x!r} is not binary.")


def normalize_output(v: Any) -> int:
    """Normalize Boolean output to 0/1."""
    return 0 if v in (-1, 0, False) else 1


def hamming_weight(x: Sequence[int]) -> int:
    return int(sum(int(bit) for bit in x))


def all_binary_inputs(n: int) -> list[tuple[int, ...]]:
    """Return all bit strings of length n in lexicographic order."""
    if n < 0:
        raise ValueError("n must be non-negative")
    return [tuple((k >> (n - 1 - j)) & 1 for j in range(n)) for k in range(2**n)]


def representative_for_weight(n: int, k: int) -> tuple[int, ...]:
    """Canonical representative of Hamming weight k: 11...100...0."""
    k = int(k)
    if k < 0 or k > n:
        raise ValueError("weight is outside 0..n")
    return tuple([1] * k + [0] * (n - k))


@lru_cache(maxsize=None)
def binom_safe(n: int, k: int) -> int:
    """Safe binomial coefficient, returning 0 outside the valid range."""
    n = int(n)
    k = int(k)
    if k < 0 or k > n:
        return 0
    return comb(n, k)


def normalize_layer_value(layer_value: Mapping[int, Any]) -> LayerValue:
    """Normalize a layer -> output map into sorted integer keys and 0/1 values."""
    out: LayerValue = {}
    for k_raw, v_raw in layer_value.items():
        k = int(k_raw)
        out[k] = normalize_output(v_raw)
    return dict(sorted(out.items()))


def layer_values_from_truth_table(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any] | None = None,
    f: Callable[[tuple[int, ...]], Any] | None = None,
) -> tuple[int, LayerValue]:
    """Convert an explicit symmetric truth/promise table into layer values."""
    raw_inputs = [normalize_input_bits(x) for x in inputs]
    if not raw_inputs:
        raise ValueError("inputs must be non-empty")

    n = len(raw_inputs[0])
    if any(len(x) != n for x in raw_inputs):
        raise ValueError("all inputs must have the same length")

    if outputs is None:
        if f is None:
            raise ValueError("provide outputs or f")
        raw_outputs = [f(x) for x in raw_inputs]
    else:
        raw_outputs = list(outputs)
        if len(raw_outputs) != len(raw_inputs):
            raise ValueError("len(outputs) must equal len(inputs)")

    layer_value: dict[int, int] = {}
    layer_count: dict[int, int] = {}

    for x, y_raw in zip(raw_inputs, raw_outputs):
        k = hamming_weight(x)
        y = normalize_output(y_raw)
        layer_count[k] = layer_count.get(k, 0) + 1

        if k in layer_value and layer_value[k] != y:
            raise ValueError(f"function is not symmetric: layer {k} has multiple outputs")
        layer_value[k] = y

    for k, count in layer_count.items():
        expected = binom_safe(n, k)
        if count != expected:
            raise ValueError(
                f"domain is not a union of full Hamming layers: layer {k} has "
                f"{count} points but should have {expected}"
            )

    return n, dict(sorted(layer_value.items()))


# Common symmetric Boolean functions by layer.
def OR_layers(n: int) -> LayerValue:
    return {k: int(k > 0) for k in range(n + 1)}


def AND_layers(n: int) -> LayerValue:
    return {k: int(k == n) for k in range(n + 1)}


def EXACT_layers(n: int, r: int) -> LayerValue:
    return {k: int(k == int(r)) for k in range(n + 1)}


def THRESHOLD_layers(n: int, r: int) -> LayerValue:
    return {k: int(k >= int(r)) for k in range(n + 1)}


def PARITY_layers(n: int) -> LayerValue:
    return {k: k % 2 for k in range(n + 1)}


def DJ_layers(N: int) -> LayerValue:
    """Deutsch-Jozsa promise layers on truth-table length N."""
    N = int(N)
    if N % 2 != 0:
        raise ValueError("Deutsch-Jozsa length must be even")
    return {0: 0, N // 2: 1, N: 0}


# ============================================================
# Terwilliger block coefficients
# ============================================================

@lru_cache(maxsize=None)
def beta_schrijver(m: int, i: int, j: int, k: int, t: int) -> int:
    """
    Schrijver's beta coefficient beta^t_{i,j,k} for H(m,2):

        sum_u (-1)^(u-t) C(u,t) C(m-2k,u-k)
              C(m-k-u,i-u) C(m-k-u,j-u).
    """
    total = 0
    for u in range(k, m - k + 1):
        c1 = binom_safe(u, t)
        c2 = binom_safe(m - 2 * k, u - k)
        c3 = binom_safe(m - k - u, i - u)
        c4 = binom_safe(m - k - u, j - u)
        if c1 and c2 and c3 and c4:
            total += (-1 if ((u - t) % 2) else 1) * c1 * c2 * c3 * c4
    return int(total)


@lru_cache(maxsize=None)
def terwilliger_block_coeff(m: int, i: int, j: int, k: int, t: int) -> float:
    """
    Coefficient of M^t_{i,j} in the kth Terwilliger block:

        C(m-2k,i-k)^(-1/2) C(m-2k,j-k)^(-1/2) beta^t_{i,j,k}.
    """
    denom_i = binom_safe(m - 2 * k, i - k)
    denom_j = binom_safe(m - 2 * k, j - k)
    if denom_i == 0 or denom_j == 0:
        return 0.0

    beta = beta_schrijver(m, i, j, k, t)
    if beta == 0:
        return 0.0

    return float(beta) / sqrt(float(denom_i * denom_j))


def valid_intersections(a: int, b: int, N: int) -> range:
    """Valid t values for subsets A,B of an N-set."""
    lo = max(0, int(a) + int(b) - int(N))
    hi = min(int(a), int(b))
    return range(lo, hi + 1)


# ============================================================
# CVXPY and numerical linear algebra helpers
# ============================================================

def scalar_to_1x1(expr: Any):
    if isinstance(expr, Number):
        expr = cp.Constant(float(expr))
    return cp.reshape(expr, (1, 1), order="C")


def scalar_bmat(entries: list[list[Any]]):
    return cp.bmat([[scalar_to_1x1(e) for e in row] for row in entries])


def available_default_solver():
    _require_cvxpy()
    installed = set(cp.installed_solvers())
    if "MOSEK" in installed:
        return cp.MOSEK
    if "CLARABEL" in installed:
        return "CLARABEL"
    if "SDPA" in installed:
        return cp.SDPA
    return cp.SCS


def solve_cvxpy_problem(
    problem: Any,
    solver: str | None = None,
    verbose: bool = False,
    solver_kwargs: Mapping[str, Any] | None = None,
):
    _require_cvxpy()
    opts = dict(solver_kwargs or {})
    chosen = solver if solver is not None else available_default_solver()
    if chosen == cp.SCS or chosen == "SCS":
        opts.setdefault("eps", 1e-6)
        opts.setdefault("max_iters", 200000)
    return problem.solve(solver=chosen, verbose=verbose, **opts)


def _psd_factor(M: Array, tol: float = 1e-8) -> Array:
    """Return R with M approximately equal to R @ R.T."""
    M = np.asarray(M, dtype=float)
    M = 0.5 * (M + M.T)
    vals, vecs = la.eigh(M)
    max_val = float(np.max(np.abs(vals))) if vals.size else 0.0
    cutoff = max(tol, tol * max(1.0, max_val))

    if np.min(vals) < -100 * cutoff:
        raise ValueError(
            f"matrix has a negative eigenvalue {np.min(vals):.3e}; "
            "the SDP solution may be inaccurate"
        )

    keep = vals > cutoff
    if not np.any(keep):
        return np.zeros((M.shape[0], 0), dtype=float)

    return vecs[:, keep] * np.sqrt(vals[keep])[None, :]


def _complete_basis(Q: Array, tol: float = 1e-10) -> Array:
    """Complete an orthonormal basis Q to a square unitary matrix."""
    Q = np.asarray(Q, dtype=complex)
    n, r = Q.shape
    if r == n:
        return Q

    N = la.null_space(Q.conj().T, rcond=tol)
    E = np.hstack([Q, N])
    if E.shape == (n, n):
        return E

    # Fallback: Gram-Schmidt with computational basis.
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
    """Return a unitary U with U @ source approximately target."""
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

    Es = _complete_basis(Qs)
    Et = _complete_basis(Qt)
    U = Et @ Es.conj().T

    map_error = float(la.norm(U @ source - target, ord="fro"))
    unitarity_error = float(la.norm(U.conj().T @ U - np.eye(d), ord="fro"))
    return U, gram_error, map_error, unitarity_error


# ============================================================
# Terwilliger reduced SDP
# ============================================================

def orbit_variable_keys(N: int, allowed_layers: Iterable[int]) -> list[OrbitKey]:
    """All orbit keys needed for X_0 on the promised Hamming layers."""
    allowed = set(int(k) for k in allowed_layers)
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


def solve_symmetric_adversary_min_rank_terwilliger_from_layers(
    n: int,
    layer_value: Mapping[int, Any],
    epsilon: float = 0.1**3,
    *,
    solver: str | None = None,
    solver_kwargs: dict[str, Any] | None = None,
    psd_factor_tol: float = 1e-8,
    cap_factor: float = 1.0,
    adversary_cap_override: float | None = None,
    verbose: bool = True,
) -> TerwilligerPhaseAdversarySolution:
    """
    Solve the Terwilliger-orbit-reduced phase-oracle state-conversion SDP.

    The solver always performs:
      1. first pass: minimise max_x W_x;
      2. second pass: minimise sum_x W_x under W_x <= cap.

    The explicit witnesses are constructed only for one canonical representative
    of each promised Hamming layer.
    """
    _require_cvxpy()
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")

    lv = normalize_layer_value(layer_value)
    allowed_layers = tuple(sorted(lv.keys()))
    if not allowed_layers:
        raise ValueError("layer_value must be non-empty")
    for k in allowed_layers:
        if k < 0 or k > n:
            raise ValueError(f"layer {k} is outside 0..n")
    if len({lv[k] for k in allowed_layers}) < 2:
        raise ValueError("the function/promise must include both output values")

    N = n - 1
    allowed_set = set(allowed_layers)
    keys = orbit_variable_keys(N, allowed_layers)
    key_set = set(keys)
    solver_kwargs = dict(solver_kwargs or {})

    def xexpr(var_map: dict[OrbitKey, Any], key: OrbitKey):
        if key not in var_map:
            raise KeyError(f"invalid orbit key {key}")
        return var_map[key]

    def cost_expr(var_map: dict[OrbitKey, Any], alpha: int, a: int):
        """Cost C_x = 1/2 sum_i X_i[x,x]."""
        expr = xexpr(var_map, (alpha, alpha, a, a, a))

        if a > 0:
            r = alpha + a - 1
            expr += a * xexpr(var_map, (1, 1, r, r, r))

        if N - a > 0:
            r = alpha + a
            expr += (N - a) * xexpr(var_map, (0, 0, r, r, r))

        return 0.5 * expr

    def equality_lhs(var_map: dict[OrbitKey, Any], alpha: int, beta: int, a: int, b: int, t: int):
        """Orbit expression for sum_{i:x_i != y_i} X_i[x,y]."""
        expr = 0

        if alpha != beta:
            expr += xexpr(var_map, (alpha, beta, a, b, t))

        count_A_minus_B = a - t
        if count_A_minus_B > 0:
            aa = alpha + a - 1
            bb = beta + b
            tt = alpha * beta + t
            expr += count_A_minus_B * xexpr(var_map, (1, 0, aa, bb, tt))

        count_B_minus_A = b - t
        if count_B_minus_A > 0:
            aa = alpha + a
            bb = beta + b - 1
            tt = alpha * beta + t
            expr += count_B_minus_A * xexpr(var_map, (0, 1, aa, bb, tt))

        return expr

    def build_psd_constraints(var_map: dict[OrbitKey, Any]):
        constraints = []
        for k in range(N // 2 + 1):
            weights = list(range(k, N - k + 1))
            labels = [
                (alpha, i)
                for alpha in (0, 1)
                for i in weights
                if alpha + i in allowed_set
            ]
            if not labels:
                continue

            entries = [[0 for _ in labels] for __ in labels]
            has_term = False

            for row, (alpha, i) in enumerate(labels):
                for col, (beta, j) in enumerate(labels):
                    entry = 0
                    for tau in valid_intersections(i, j, N):
                        key = (alpha, beta, i, j, tau)
                        if key not in key_set:
                            continue
                        coeff = terwilliger_block_coeff(N, i, j, k, tau)
                        if coeff != 0.0:
                            entry = entry + coeff * xexpr(var_map, key)
                            has_term = True
                    entries[row][col] = entry

            if not has_term:
                continue

            B = scalar_bmat(entries)
            B = 0.5 * (B + B.T)
            if len(labels) == 1:
                constraints.append(B[0, 0] >= 0)
            else:
                constraints.append(B >> 0)
        return constraints

    def solve_pass(mode: str, cap: float | None = None):
        var_map: dict[OrbitKey, Any] = {
            key: cp.Variable(name=f"x_{key[0]}_{key[1]}_{key[2]}_{key[3]}_{key[4]}")
            for key in keys
        }

        constraints: list[Any] = []

        # Gram symmetry: X_0[x,y] = X_0[y,x].
        for key in keys:
            alpha, beta, a, b, t = key
            key_T = (beta, alpha, b, a, t)
            if key_T in key_set and key <= key_T:
                constraints.append(xexpr(var_map, key) == xexpr(var_map, key_T))

        constraints.extend(build_psd_constraints(var_map))

        input_costs: list[tuple[int, int, Any]] = []
        for alpha in (0, 1):
            for a in range(N + 1):
                if alpha + a in allowed_set:
                    input_costs.append((alpha, a, cost_expr(var_map, alpha, a)))

        if mode == "max":
            tau = cp.Variable(name="phase_adv")
            constraints.append(tau >= 0)
            for _, _, c in input_costs:
                constraints.append(c <= tau)
            objective = cp.Minimize(tau)
        elif mode == "sum_under_cap":
            if cap is None:
                raise ValueError("cap is required for sum_under_cap")
            for _, _, c in input_costs:
                constraints.append(c <= cap)

            # Weighted by number of inputs with distinguished bit alpha and rest weight a.
            weighted_sum = 0
            for _alpha, a, c in input_costs:
                weighted_sum += binom_safe(N, a) * c
            objective = cp.Minimize(weighted_sum)
        else:
            raise ValueError("mode must be 'max' or 'sum_under_cap'")

        # Equality constraints for all pair orbits on promised layers.
        for alpha in (0, 1):
            for beta in (0, 1):
                for a in range(N + 1):
                    if alpha + a not in allowed_set:
                        continue
                    for b in range(N + 1):
                        if beta + b not in allowed_set:
                            continue
                        for t in valid_intersections(a, b, N):
                            lhs = equality_lhs(var_map, alpha, beta, a, b, t)
                            rhs = 1.0 if lv[alpha + a] != lv[beta + b] else 0.0
                            constraints.append(lhs == rhs)

        problem = cp.Problem(objective, constraints)
        value = solve_cvxpy_problem(problem, solver=solver, verbose=verbose, solver_kwargs=solver_kwargs)

        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise RuntimeError(f"Terwilliger phase SDP failed with status {problem.status}")

        values: dict[OrbitKey, float] = {}
        for key, var in var_map.items():
            if var.value is None:
                raise RuntimeError(f"variable {key} has no value")
            values[key] = float(var.value)

        return float(value), values, str(problem.status), problem.solver_stats

    first_value, _first_orbits, status_1, _stats_1 = solve_pass("max")

    if adversary_cap_override is None:
        cap = cap_factor * first_value + epsilon
    else:
        cap = float(adversary_cap_override) + epsilon

    final_value, orbit_values, final_status, final_stats = solve_pass("sum_under_cap", cap=cap)

    # Numerical Terwilliger block values for diagnostics.
    block_values: list[Array] = []
    for k in range(N // 2 + 1):
        weights = list(range(k, N - k + 1))
        labels = [(alpha, i) for alpha in (0, 1) for i in weights if alpha + i in allowed_set]
        if not labels:
            continue
        B = np.zeros((len(labels), len(labels)), dtype=float)
        for row, (alpha, i) in enumerate(labels):
            for col, (beta, j) in enumerate(labels):
                val = 0.0
                for tau in valid_intersections(i, j, N):
                    key = (alpha, beta, i, j, tau)
                    if key in orbit_values:
                        val += terwilliger_block_coeff(N, i, j, k, tau) * orbit_values[key]
                B[row, col] = val
        block_values.append(0.5 * (B + B.T))

    # Build one canonical representative per promised Hamming layer.
    weights = list(allowed_layers)
    inputs = [(k,) for k in weights]
    query_inputs = [representative_for_weight(n, k) for k in weights]
    outputs = [lv[k] for k in weights]
    m_eff = len(weights)

    # Expand orbit solution only to the layer representatives.
    grams: list[Array] = []
    for coord in range(n):
        X_i = np.zeros((m_eff, m_eff), dtype=float)
        rest_coords = [j for j in range(n) if j != coord]

        for x_idx, x in enumerate(query_inputs):
            alpha = x[coord]
            A_set = {j for j in rest_coords if x[j] == 1}
            a = len(A_set)

            for y_idx, y in enumerate(query_inputs):
                beta = y[coord]
                B_set = {j for j in rest_coords if y[j] == 1}
                b = len(B_set)
                t = len(A_set & B_set)
                X_i[x_idx, y_idx] = orbit_values[(alpha, beta, a, b, t)]

        grams.append(0.5 * (X_i + X_i.T))

    z_vecs: list[Array] = []
    w_vecs: list[Array] = []
    witness_dims: list[int] = []

    for X_i in grams:
        R_i = _psd_factor(X_i, tol=psd_factor_tol)
        z_vecs.append(R_i)
        w_vecs.append(R_i / np.sqrt(2.0))
        witness_dims.append(int(R_i.shape[1]))

    private_dim = int(sum(witness_dims))
    public_dim = max(max(outputs) + 1, 1) if outputs else 1

    return TerwilligerPhaseAdversarySolution(
        inputs=inputs,
        query_inputs=query_inputs,
        outputs=outputs,
        n=n,
        allowed_layers=allowed_layers,
        layer_value=lv,
        objective_value=float(final_value),
        adversary_value=float(first_value),
        adversary_cap=float(cap),
        status=(
            f"{final_status}; first_pass_status={status_1}; "
            f"first_pass_adv={first_value}; cap={cap}"
        ),
        z=z_vecs,
        w=w_vecs,
        grams=grams,
        orbit_values=orbit_values,
        block_values=block_values,
        public_dim=public_dim,
        witness_dims=witness_dims,
        private_dim=private_dim,
        solver_stats=final_stats,
    )


def solve_symmetric_adversary_min_rank_terwilliger(
    inputs: Sequence[Sequence[Any]],
    outputs: Sequence[Any] | None = None,
    f: Callable[[tuple[int, ...]], Any] | None = None,
    epsilon: float = 0.1**3,
    *,
    solver: str | None = None,
    solver_kwargs: dict[str, Any] | None = None,
    psd_factor_tol: float = 1e-8,
    cap_factor: float = 2.0,
    adversary_cap_override: float | None = None,
    verbose: bool = True,
) -> TerwilligerPhaseAdversarySolution:
    """Wrapper accepting an explicit symmetric truth/promise table."""
    n, layer_value = layer_values_from_truth_table(inputs, outputs, f)
    return solve_symmetric_adversary_min_rank_terwilliger_from_layers(
        n=n,
        layer_value=layer_value,
        epsilon=epsilon,
        solver=solver,
        solver_kwargs=solver_kwargs,
        psd_factor_tol=psd_factor_tol,
        cap_factor=cap_factor,
        adversary_cap_override=adversary_cap_override,
        verbose=verbose,
    )

def expand_terwilliger_orbit_solution_to_one_vector_direct_sum_solution(
    sol,
    *,
    psd_factor_tol: float = 1e-8,
):
    """
    Expand a Terwilliger/orbit-basis phase-oracle solution back to the full
    Boolean input basis, but return it in the same format as
    OneVectorDirectSumAdversarySolution.

    If sol is already a OneVectorDirectSumAdversarySolution, this function
    returns it unchanged.
    """

    # Already expanded / already in the expected format.
    if isinstance(sol, OneVectorDirectSumAdversarySolution):
        return sol

    # Otherwise this must be a Terwilliger/orbit solution.
    if not hasattr(sol, "n"):
        raise TypeError(
            "Expected a Terwilliger/orbit solution with attribute `n`, "
            "or an already-expanded OneVectorDirectSumAdversarySolution. "
            f"Got object of type {type(sol).__name__}."
        )

    if not hasattr(sol, "orbit_values"):
        raise TypeError(
            "Expected a Terwilliger/orbit solution with attribute `orbit_values`. "
            f"Got object of type {type(sol).__name__}."
        )

    n = int(sol.n)

    if not hasattr(sol, "allowed_layers"):
        allowed_layers = sorted({sum(x) for x in sol.query_inputs})
    else:
        allowed_layers = tuple(sorted(int(k) for k in sol.allowed_layers))

    allowed_set = set(allowed_layers)

    if not hasattr(sol, "layer_value"):
        layer_value = {}
        for x, y in zip(sol.query_inputs, sol.outputs):
            layer_value[sum(x)] = int(y)
    else:
        layer_value = dict(sol.layer_value)

    # Full promised Boolean domain.
    full_inputs = [
        x for x in all_binary_inputs(n)
        if hamming_weight(x) in allowed_set
    ]

    full_outputs = [
        int(layer_value[hamming_weight(x)])
        for x in full_inputs
    ]

    m = len(full_inputs)

    # Reconstruct the full simplified phase Gram matrices X_i.
    #
    # X_i[x,y] = <z_{x,i}, z_{y,i}>.
    phase_grams = []

    for coord in range(n):
        X_i = np.zeros((m, m), dtype=float)

        rest_coords = [j for j in range(n) if j != coord]

        for x_idx, x in enumerate(full_inputs):
            alpha = int(x[coord])
            A_set = {j for j in rest_coords if int(x[j]) == 1}
            a = len(A_set)

            for y_idx, y in enumerate(full_inputs):
                beta = int(y[coord])
                B_set = {j for j in rest_coords if int(y[j]) == 1}
                b = len(B_set)
                t = len(A_set & B_set)

                key = (alpha, beta, a, b, t)

                if key not in sol.orbit_values:
                    raise KeyError(
                        f"missing orbit key {key}; cannot expand this orbit solution"
                    )

                X_i[x_idx, y_idx] = float(sol.orbit_values[key])

        X_i = 0.5 * (X_i + X_i.T)
        phase_grams.append(X_i)

    # Factor X_i = R_i R_i^T.
    #
    # Rows of R_i are z_{x,i}.
    z_vecs = []
    witness_dims = []

    for X_i in phase_grams:
        R_i = _psd_factor(X_i, tol=psd_factor_tol)
        z_vecs.append(R_i)
        witness_dims.append(int(R_i.shape[1]))

    # Convert phase witnesses z_{x,i} into old direct-sum witnesses:
    #
    #   w_{x,i} = [ z_{x,i}/2 , -z_{x,i}/2 ].
    #
    # Then the bit-flip oracle X acts as a phase flip on this antisymmetric
    # subspace.
    direct_sum_w = []
    block_grams = []

    for i in range(n):
        Z_i = z_vecs[i]
        r_i = witness_dims[i]

        W_i = np.zeros((m, 2 * r_i), dtype=float)
        W_i[:, :r_i] = 0.5 * Z_i
        W_i[:, r_i:] = -0.5 * Z_i

        direct_sum_w.append(W_i)

        # Old one-vector direct-sum local Gram matrix indexed by (x,a),
        # where a in {0,1}.
        G_i = np.zeros((2 * m, 2 * m), dtype=float)

        for x_idx in range(m):
            for y_idx in range(m):
                base = phase_grams[i][x_idx, y_idx]

                G_i[2 * x_idx + 0, 2 * y_idx + 0] = 0.25 * base
                G_i[2 * x_idx + 0, 2 * y_idx + 1] = -0.25 * base
                G_i[2 * x_idx + 1, 2 * y_idx + 0] = -0.25 * base
                G_i[2 * x_idx + 1, 2 * y_idx + 1] = 0.25 * base

        G_i = 0.5 * (G_i + G_i.T)
        block_grams.append(G_i)

    # Build Boolean local bit-flip oracles for the old direct-sum implementation.
    I2 = np.eye(2, dtype=float)
    X2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)

    local_oracles = []

    for coord in range(n):
        local_oracles_i = []

        for x in full_inputs:
            local_oracles_i.append(I2 if int(x[coord]) == 0 else X2)

        local_oracles.append(local_oracles_i)

    alphabet_values = [[0, 1] for _ in range(n)]
    local_oracle_dims = [2 for _ in range(n)]

    public_dim = int(getattr(sol, "public_dim", max(full_outputs) + 1 if full_outputs else 1))
    private_dim = int(sum(2 * r for r in witness_dims))

    return OneVectorDirectSumAdversarySolution(
        inputs=full_inputs,
        outputs=full_outputs,
        objective_value=float(sol.objective_value),
        adversary_cap=float(sol.adversary_cap),
        status=str(sol.status) + "; expanded_to_one_vector_direct_sum=True",
        w=direct_sum_w,
        block_grams=block_grams,
        alphabet_values=alphabet_values,
        local_oracles=local_oracles,
        public_dim=public_dim,
        local_oracle_dims=local_oracle_dims,
        witness_dims=witness_dims,
        private_dim=private_dim,
    )

def solve_symmetric_layers(
    n: int,
    layer_value,
    *,
    solver: str | None = None,
    psd_factor_tol: float = 1e-8,
    epsilon: float = 1e-8,
    solver_kwargs: dict[str, Any] | None = None,
    cap_factor: float = 2.0,
    adversary_cap_override: float | None = None,
):
    """
    Convenience wrapper for symmetric Boolean functions given by Hamming layers.

    Example:

        final_sol = solve_symmetric_layers(
            n,
            THRESHOLD_layers(n, k),
            solver="MOSEK",
            psd_factor_tol=0.1**5,
            epsilon=0.1**5,
        )

    This performs:

        sol = solve_symmetric_adversary_min_rank_terwilliger_from_layers(...)
        final_sol = expand_terwilliger_orbit_solution_to_one_vector_direct_sum_solution(...)

    and returns final_sol, which is compatible with:

        compute_one_vector_direct_sum_transducer_unitary(final_sol)
    """
    sol = solve_symmetric_adversary_min_rank_terwilliger_from_layers(
        n,
        layer_value,
        solver=solver,
        solver_kwargs=solver_kwargs,
        psd_factor_tol=psd_factor_tol,
        epsilon=epsilon,
        cap_factor=cap_factor,
        adversary_cap_override=adversary_cap_override,
    )

    final_sol = expand_terwilliger_orbit_solution_to_one_vector_direct_sum_solution(
        sol,
        psd_factor_tol=psd_factor_tol,
    )

    return final_sol


def qiskit_circuit_for_one_vector_direct_sum_input(
    transducer,
    x_idx: int,
    K: int,
    *,
    measure_all: bool = False,
):
    """
    Build a Qiskit circuit for a fixed input x_idx using the old one-vector
    direct-sum transducer.

    Requires the functions:
        one_vector_direct_sum_query_for_input(...)
    and a transducer returned by:
        compute_one_vector_direct_sum_transducer_unitary(...)
    """
    oracle_Q = one_vector_direct_sum_query_for_input(transducer, x_idx)

    return qiskit_repeated_transducer_circuit_from_oracle_matrix(
        transducer_U=transducer.U,
        oracle_Q=oracle_Q,
        public_dim=transducer.public_dim,
        private_dim=transducer.private_dim,
        K=K,
        measure_all=measure_all,
        name=f"RepeatedTransducer_x{x_idx}",
    )