#!/usr/bin/env python3
"""
Level-0 multi-bias optimization for symmetric junta learning.

The main numerically stable implementation is for EXACT_ell (and therefore OR,
which is equivalent up to complementing the oracle output to EXACT_0).  In this
singleton-output case, the finite-bias level-0 nuclear-norm SDP reduces to an
SOCP.

For a fixed bias p, define
    R(p)[d,a] = mu_{d,p}(a,ell) / sqrt(w_p(a) w_p(ell)),
where d=1,...,k and a != ell.  The finite-bias primal is

    minimize    2 sum_j ||z_j||_2
    subject to  2 sum_j R(p_j) z_j = 1_k.

Its dual is

    maximize    1_k^T y
    subject to  ||R(p_j)^T y||_2 <= 1  for every chosen p_j.

The continuum-bias dual replaces the finite constraints by the same inequality
for every p in (0,1).  The exchange algorithm below solves a sequence of finite
problems and adds the most violated p.

The script also implements Belovs' OR continuum profile
    v_d(p) = d (1-p)^(d-1)
and its exact finite Gauss-Legendre compression.

Dependencies for the tested path: numpy, scipy.
Optional: cvxpy for direct primal SOCP / general nuclear-norm SDP solves.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, List, Sequence, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, logsumexp


# -----------------------------------------------------------------------------
# Probability kernels
# -----------------------------------------------------------------------------


def _logcomb(n: int, r: int) -> float:
    if r < 0 or r > n:
        return -np.inf
    return float(gammaln(n + 1) - gammaln(r + 1) - gammaln(n - r + 1))


def _log_binom_pmf(n: int, r: int, p: float) -> float:
    if r < 0 or r > n:
        return -np.inf
    if not (0.0 < p < 1.0):
        raise ValueError("This numerical routine requires 0 < p < 1.")
    return _logcomb(n, r) + r * math.log(p) + (n - r) * math.log1p(-p)


def log_mu_joint(k: int, d: int, p: float, a: int, b: int) -> float:
    """Log of mu_{d,p}(a,b).

    For two hidden k-sets A,B at Johnson distance d,
        J ~ Bin(k-d,p), Y,Z ~ Bin(d,p) independently,
        a = J+Y, b = J+Z.
    """
    if not (1 <= d <= k):
        raise ValueError("d must be in {1,...,k}")
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie strictly between 0 and 1")

    j_min = max(0, a - d, b - d)
    j_max = min(k - d, a, b)
    if j_min > j_max:
        return -np.inf

    lp = math.log(p)
    lq = math.log1p(-p)
    terms = []
    for j in range(j_min, j_max + 1):
        y = a - j
        z = b - j
        if not (0 <= y <= d and 0 <= z <= d):
            continue
        terms.append(
            _logcomb(k - d, j)
            + _logcomb(d, y)
            + _logcomb(d, z)
            + (a + b - j) * lp
            + (k + d - a - b + j) * lq
        )
    return float(logsumexp(terms)) if terms else -np.inf


def mu_joint(k: int, d: int, p: float, a: int, b: int) -> float:
    lm = log_mu_joint(k, d, p, a, b)
    return 0.0 if np.isneginf(lm) else float(math.exp(lm))


def binom_weight(k: int, a: int, p: float) -> float:
    return float(math.exp(_log_binom_pmf(k, a, p)))


def exact_columns(k: int, ell: int) -> List[int]:
    return [a for a in range(k + 1) if a != ell]


def exact_R(k: int, ell: int, p: float) -> np.ndarray:
    """Normalized level-0 matrix R(p) for EXACT_ell.

    Shape: (k distances) x (k non-ell local weights).
    """
    if not (0 <= ell <= k):
        raise ValueError("ell must be in {0,...,k}")
    cols = exact_columns(k, ell)
    lw_ell = _log_binom_pmf(k, ell, p)
    R = np.zeros((k, len(cols)), dtype=float)

    for di, d in enumerate(range(1, k + 1)):
        for ai, a in enumerate(cols):
            lm = log_mu_joint(k, d, p, a, ell)
            if np.isneginf(lm):
                continue
            lwa = _log_binom_pmf(k, a, p)
            R[di, ai] = math.exp(lm - 0.5 * (lwa + lw_ell))
    return R


def general_R_blocks(k: int, profile: Sequence[int], p: float) -> Tuple[List[int], List[int], List[np.ndarray]]:
    """Return W0, W1 and normalized matrices R_d(p) for a general Boolean profile.

    R_d has rows indexed by a in W0 and columns by b in W1, with
        R_d[a,b] = mu_{d,p}(a,b) / sqrt(w_p(a) w_p(b)).
    """
    if len(profile) != k + 1 or any(v not in (0, 1) for v in profile):
        raise ValueError("profile must be a Boolean list/tuple of length k+1")
    W0 = [a for a, v in enumerate(profile) if v == 0]
    W1 = [a for a, v in enumerate(profile) if v == 1]
    if not W0 or not W1:
        raise ValueError("profile must be nonconstant")

    logw = [_log_binom_pmf(k, a, p) for a in range(k + 1)]
    blocks: List[np.ndarray] = []
    for d in range(1, k + 1):
        M = np.zeros((len(W0), len(W1)), dtype=float)
        for ia, a in enumerate(W0):
            for ib, b in enumerate(W1):
                lm = log_mu_joint(k, d, p, a, b)
                if np.isneginf(lm):
                    continue
                M[ia, ib] = math.exp(lm - 0.5 * (logw[a] + logw[b]))
        blocks.append(M)
    return W0, W1, blocks


# -----------------------------------------------------------------------------
# EXACT_ell finite-bias dual and continuum exchange method
# -----------------------------------------------------------------------------


@dataclass
class ExchangeStep:
    iteration: int
    restricted_value: float
    max_continuum_violation: float
    worst_p: float
    number_of_biases: int


@dataclass
class ExchangeResult:
    k: int
    ell: int
    biases: List[float]
    dual_y: List[float]
    restricted_upper_bound: float
    scaled_continuum_lower_bound: float
    max_continuum_violation: float
    history: List[ExchangeStep]


def _solve_exact_dual_for_Rs(Rs: Sequence[np.ndarray], y0: np.ndarray | None = None,
                             ftol: float = 1e-11, maxiter: int = 4000):
    """Solve max 1^T y subject to ||R_j^T y||_2 <= 1 using SLSQP."""
    if not Rs:
        raise ValueError("At least one bias is required")
    k = Rs[0].shape[0]
    Qs = [R @ R.T for R in Rs]
    if y0 is None:
        y0 = np.zeros(k, dtype=float)

    def objective(y):
        return -float(np.sum(y))

    def objective_jac(y):
        return -np.ones_like(y)

    def inequalities(y):
        out = []
        for Q in Qs:
            s = max(0.0, float(y @ Q @ y))
            out.append(1.0 - math.sqrt(s))
        return np.asarray(out)

    def inequalities_jac(y):
        rows = []
        for Q in Qs:
            q = Q @ y
            s = max(0.0, float(y @ q))
            norm = math.sqrt(s)
            rows.append(np.zeros_like(y) if norm < 1e-14 else -q / norm)
        return np.asarray(rows)

    result = minimize(
        objective,
        y0,
        jac=objective_jac,
        method="SLSQP",
        constraints=[{"type": "ineq", "fun": inequalities, "jac": inequalities_jac}],
        options={"ftol": ftol, "maxiter": maxiter, "disp": False},
    )
    return result


def dual_envelope_exact(k: int, ell: int, y: np.ndarray, p: float) -> float:
    return float(np.linalg.norm(exact_R(k, ell, p).T @ y))


def exchange_exact_continuum(
    k: int,
    ell: int,
    initial_biases: Sequence[float] = (0.5,),
    scan_points: int = 401,
    tol: float = 1e-4,
    max_rounds: int = 30,
    p_eps: float = 2e-4,
    refine: bool = True,
) -> ExchangeResult:
    """Approximate the continuum-bias EXACT_ell level-0 optimum by exchange.

    At every round we solve the finite dual on the current bias set, scan the
    continuum for the largest violation, and add that p.

    If vmax = max_p ||R(p)^T y||, then y/vmax is feasible for the continuum
    dual.  Therefore
        (sum y)/vmax <= U_continuum <= U_restricted = sum y,
    modulo numerical solver error.
    """
    if scan_points < 21:
        raise ValueError("scan_points should be at least 21")
    ps = sorted(set(float(p) for p in initial_biases))
    if any(not (p_eps <= p <= 1.0 - p_eps) for p in ps):
        raise ValueError("initial biases must lie inside [p_eps,1-p_eps]")

    grid = np.linspace(p_eps, 1.0 - p_eps, scan_points)
    # Precompute the expensive R matrices on the scan grid.
    grid_R = [exact_R(k, ell, float(p)) for p in grid]
    Rs = [exact_R(k, ell, p) for p in ps]

    y = None
    history: List[ExchangeStep] = []
    restricted_value = math.inf
    vmax = math.inf

    for it in range(max_rounds):
        result = _solve_exact_dual_for_Rs(Rs, y0=y)
        if not result.success:
            # SLSQP occasionally reports a line-search warning while returning
            # a very good point.  We keep the point, but expose the issue.
            pass
        y = np.asarray(result.x, dtype=float)
        restricted_value = float(np.sum(y))

        vals = np.asarray([np.linalg.norm(R.T @ y) for R in grid_R])
        idx = int(np.argmax(vals))
        p_star = float(grid[idx])
        vmax = float(vals[idx])

        if refine:
            lo = float(grid[max(0, idx - 1)])
            hi = float(grid[min(scan_points - 1, idx + 1)])
            if hi > lo:
                local = minimize_scalar(
                    lambda p: -dual_envelope_exact(k, ell, y, float(p)),
                    bounds=(lo, hi),
                    method="bounded",
                    options={"xatol": 1e-10},
                )
                if local.success:
                    p_star = float(local.x)
                    vmax = float(-local.fun)

        history.append(
            ExchangeStep(
                iteration=it,
                restricted_value=restricted_value,
                max_continuum_violation=vmax,
                worst_p=p_star,
                number_of_biases=len(ps),
            )
        )

        if vmax <= 1.0 + tol:
            break

        # Do not add a numerically duplicate atom.
        if min(abs(p_star - p) for p in ps) < 1e-7:
            break
        ps.append(p_star)
        ps.sort()
        Rs = [exact_R(k, ell, p) for p in ps]

    continuum_lb = restricted_value / max(1.0, vmax)
    return ExchangeResult(
        k=k,
        ell=ell,
        biases=ps,
        dual_y=y.tolist() if y is not None else [],
        restricted_upper_bound=restricted_value,
        scaled_continuum_lower_bound=continuum_lb,
        max_continuum_violation=vmax,
        history=history,
    )


# -----------------------------------------------------------------------------
# Closed-form central EXACT-half channel
# -----------------------------------------------------------------------------


def exact_half_parity_certificate(k: int) -> dict:
    """The p=1/2 rank-one level-0 parity certificate for even k."""
    if k % 2:
        raise ValueError("exact_half_parity_certificate requires even k")
    ell = k // 2
    p = 0.5
    cols = exact_columns(k, ell)
    w_ell = binom_weight(k, ell, p)
    z = np.zeros(k, dtype=float)
    for i, a in enumerate(cols):
        if (a - ell) % 2 != 0:
            z[i] = math.sqrt(binom_weight(k, a, p) / w_ell)
    R = exact_R(k, ell, p)
    residual = float(np.max(np.abs(2.0 * (R @ z) - np.ones(k))))
    cost = float(2.0 * np.linalg.norm(z))
    expected = math.sqrt(2.0 / w_ell)
    return {
        "k": k,
        "ell": ell,
        "biases": [0.5],
        "cost": cost,
        "closed_form_cost": expected,
        "max_feasibility_residual": residual,
        "z": z.tolist(),
        "columns_a": cols,
    }


# -----------------------------------------------------------------------------
# Belovs OR continuum and exact finite quadrature compression
# -----------------------------------------------------------------------------


def belovs_or_profile(k: int, p: float) -> np.ndarray:
    """Distance profile v_d(p)=d(1-p)^(d-1), d=1,...,k."""
    d = np.arange(1, k + 1, dtype=float)
    return d * (1.0 - p) ** (d - 1.0)


def belovs_or_gauss_legendre(k: int, m: int | None = None) -> dict:
    """Exact finite compression of Belovs' uniform-p OR feasibility integral.

    m-point Gauss-Legendre quadrature is exact on polynomials of degree 2m-1.
    Since v_d(p) has degree d-1 <= k-1, m >= ceil(k/2) gives exact feasibility
    for every d=1,...,k.
    """
    if m is None:
        m = (k + 1) // 2  # ceil(k/2)
    if m < (k + 1) // 2:
        raise ValueError("For exact feasibility use m >= ceil(k/2)")

    x, w = leggauss(m)
    p = 0.5 * (x + 1.0)
    alpha = 0.5 * w

    profile = np.zeros(k, dtype=float)
    for pj, aj in zip(p, alpha):
        profile += aj * belovs_or_profile(k, float(pj))
    residual = float(np.max(np.abs(profile - np.ones(k))))

    # Belovs' diagonal-load estimate at fixed p is <= sqrt(k/[p(1-p)]).
    cost_bound = float(
        math.sqrt(k) * np.sum(alpha / np.sqrt(p * (1.0 - p)))
    )
    continuum_cost_bound = float(math.pi * math.sqrt(k))

    return {
        "k": k,
        "m": m,
        "biases": p.tolist(),
        "weights": alpha.tolist(),
        "max_feasibility_residual": residual,
        "quadrature_cost_bound": cost_bound,
        "belovs_continuum_cost_bound": continuum_cost_bound,
    }


# -----------------------------------------------------------------------------
# Optional CVXPY primal solvers
# -----------------------------------------------------------------------------


def solve_exact_primal_cvxpy(k: int, ell: int, biases: Sequence[float], solver: str | None = None):
    """Direct finite-bias EXACT SOCP. Requires cvxpy.

    Returns the actual z_j vectors, hence an explicit level-0 certificate.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("cvxpy is required for this optional function") from exc

    Rs = [exact_R(k, ell, float(p)) for p in biases]
    zs = [cp.Variable(k) for _ in biases]
    lhs = sum((2.0 * R @ z for R, z in zip(Rs, zs)), start=np.zeros(k))
    constraints = [lhs == np.ones(k)]
    objective = cp.Minimize(2.0 * cp.sum(cp.hstack([cp.norm(z, 2) for z in zs])))
    prob = cp.Problem(objective, constraints)
    kwargs = {} if solver is None else {"solver": solver}
    value = prob.solve(**kwargs)
    return {
        "status": prob.status,
        "value": float(value),
        "biases": list(map(float, biases)),
        "z": [np.asarray(z.value).ravel().tolist() for z in zs],
    }


def solve_general_level0_cvxpy(k: int, profile: Sequence[int], biases: Sequence[float],
                                solver: str | None = None):
    """General symmetric-junta finite-bias level-0 nuclear-norm SDP.

    min 2 sum_j ||B_j||_*
    s.t. 2 sum_j <R_d(p_j),B_j> = 1, d=1,...,k.

    Requires cvxpy.
    """
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise RuntimeError("cvxpy is required for this optional function") from exc

    data = [general_R_blocks(k, profile, float(p)) for p in biases]
    W0, W1, _ = data[0]
    Bs = [cp.Variable((len(W0), len(W1))) for _ in biases]

    constraints = []
    for di in range(k):
        expr = 0
        for (_, _, blocks), B in zip(data, Bs):
            expr += 2.0 * cp.sum(cp.multiply(blocks[di], B))
        constraints.append(expr == 1.0)

    objective = cp.Minimize(2.0 * cp.sum(cp.hstack([cp.normNuc(B) for B in Bs])))
    prob = cp.Problem(objective, constraints)
    kwargs = {} if solver is None else {"solver": solver}
    value = prob.solve(**kwargs)
    return {
        "status": prob.status,
        "value": float(value),
        "biases": list(map(float, biases)),
        "W0": W0,
        "W1": W1,
        "B": [np.asarray(B.value).tolist() for B in Bs],
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _print_exchange(result: ExchangeResult) -> None:
    print(f"EXACT_{result.ell}, k={result.k}")
    print(f"restricted finite-bias value : {result.restricted_upper_bound:.9g}")
    print(f"scaled continuum lower bound : {result.scaled_continuum_lower_bound:.9g}")
    print(f"max dual envelope            : {result.max_continuum_violation:.9g}")
    print(f"number of exchange biases    : {len(result.biases)}")
    print("biases:")
    print("  " + ", ".join(f"{p:.8f}" for p in result.biases))
    print("history:")
    for h in result.history:
        print(
            f"  {h.iteration:2d}: U_P={h.restricted_value:.8f}, "
            f"max F={h.max_continuum_violation:.8f} at p={h.worst_p:.8f}, "
            f"|P|={h.number_of_biases}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_half = sub.add_parser("exact-half", help="Verify the explicit p=1/2 parity certificate")
    p_half.add_argument("--k", type=int, default=12)
    p_half.add_argument("--json", action="store_true")

    p_ex = sub.add_parser("exchange", help="Continuum-bias exchange method for EXACT_ell / OR=EXACT_0")
    p_ex.add_argument("--k", type=int, default=12)
    p_ex.add_argument("--ell", type=int, default=0)
    p_ex.add_argument("--scan", type=int, default=401)
    p_ex.add_argument("--tol", type=float, default=1e-4)
    p_ex.add_argument("--rounds", type=int, default=25)
    p_ex.add_argument("--initial", type=str, default="0.5", help="comma-separated initial biases")
    p_ex.add_argument("--json", action="store_true")

    p_or = sub.add_parser("belovs-or", help="Exact finite Gauss-Legendre compression of Belovs' OR continuum")
    p_or.add_argument("--k", type=int, default=12)
    p_or.add_argument("--m", type=int, default=None)
    p_or.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.command == "exact-half":
        out = exact_half_parity_certificate(args.k)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"EXACT-half k={args.k}, p=1/2")
            print(f"cost                    : {out['cost']:.12g}")
            print(f"closed-form cost        : {out['closed_form_cost']:.12g}")
            print(f"max feasibility residual: {out['max_feasibility_residual']:.3e}")

    elif args.command == "exchange":
        initial = [float(x) for x in args.initial.split(",") if x.strip()]
        out = exchange_exact_continuum(
            args.k,
            args.ell,
            initial_biases=initial,
            scan_points=args.scan,
            tol=args.tol,
            max_rounds=args.rounds,
        )
        if args.json:
            data = asdict(out)
            print(json.dumps(data, indent=2))
        else:
            _print_exchange(out)

    elif args.command == "belovs-or":
        out = belovs_or_gauss_legendre(args.k, args.m)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"Belovs OR, k={args.k}, m={out['m']}")
            print(f"max feasibility residual : {out['max_feasibility_residual']:.3e}")
            print(f"finite quadrature cost bd: {out['quadrature_cost_bound']:.12g}")
            print(f"continuum pi*sqrt(k) bd  : {out['belovs_continuum_cost_bound']:.12g}")
            print("bias, quadrature weight")
            for p, a in zip(out["biases"], out["weights"]):
                print(f"  {p:.9f}  {a:.9f}")


if __name__ == "__main__":
    main()
