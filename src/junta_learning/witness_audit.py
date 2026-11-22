#!/usr/bin/env python3
"""Numerical audit for the odd-Majority dyadic-current witness.

The Krawtchouk transform is obtained from its symmetric Jacobi matrix, whereas
the optional common-spectrum check builds the joint-count kernel directly from
binomial coefficients.  Those two paths are intentionally independent.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass

import cvxpy as cp
import numpy as np
from scipy.linalg import eigh_tridiagonal


@dataclass
class EdgeAudit:
    left: list[int]
    right: list[int]
    rank: int
    frobenius: float
    nuclear: float
    exact_formula: float
    exact_formula_residual: float
    analytic_bound: float


def krawtchouk_from_jacobi(k: int) -> np.ndarray:
    """Return U[x,t] using the Jacobi recurrence, with U[0,t] > 0."""
    x = np.arange(k + 1, dtype=float)
    diagonal = np.full(k + 1, k / 2.0)
    off_diagonal = -0.5 * np.sqrt((x[:-1] + 1.0) * (k - x[:-1]))
    eigenvalues, U = eigh_tridiagonal(diagonal, off_diagonal)
    if np.max(np.abs(eigenvalues - np.arange(k + 1))) > 2e-10:
        raise ArithmeticError("Jacobi eigenvalues are not 0,...,k")
    U *= np.where(U[0, :] < 0.0, -1.0, 1.0)
    return U


def dyadic_chain(k: int) -> list[np.ndarray]:
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be positive and odd")
    if k == 1:
        return [np.array([0]), np.array([1])]
    L = (k + 1) // 2
    q_star = int(math.floor(math.log2(L)))
    left = [np.array([0], dtype=int)]
    for q in range(1, q_star + 1):
        left.append(
            np.array(
                [t for t in range(2 ** (q - 1), 2**q) if t % 2 == q % 2],
                dtype=int,
            )
        )
    reflected = [np.array([k - t for t in block], dtype=int) for block in left]
    return left + [reflected[q] for q in range(q_star, -1, -1)]


def construct_witness(k: int) -> dict:
    U = krawtchouk_from_jacobi(k)
    m = (k - 1) // 2
    signs = np.r_[np.ones(m + 1), -np.ones(m + 1)]
    H = (U.T * signs) @ U
    rho = math.sqrt(k + 1.0) * np.abs(U[m, :])
    current = np.zeros_like(H)
    omega = np.zeros_like(H)
    edges: list[EdgeAudit] = []

    chain = dyadic_chain(k)
    for left, right in zip(chain, chain[1:]):
        z_left = float(np.sum(rho[left] ** 2))
        z_right = float(np.sum(rho[right] ** 2))
        mu_left = rho[left] ** 2 / z_left
        mu_right = rho[right] ** 2 / z_right
        edge_current = np.outer(mu_left, mu_right)
        block = edge_current / H[np.ix_(left, right)]
        current[np.ix_(left, right)] += edge_current
        current[np.ix_(right, left)] -= edge_current.T
        omega[np.ix_(left, right)] += block
        omega[np.ix_(right, left)] -= block.T
        singular = np.linalg.svd(block, compute_uv=False)
        # Division by a numerically formed H perturbs the exact rank-two block;
        # use a declared relative threshold when reporting its numerical rank.
        tolerance = 1e-9 * singular[0]
        diameter = int(np.max(right[:, None] - left[None, :]))
        mean_left = float(np.sum(left * mu_left))
        mean_right = float(np.sum(right * mu_right))
        std_left = float(np.sqrt(np.sum((left - mean_left) ** 2 * mu_left)))
        std_right = float(np.sqrt(np.sum((right - mean_right) ** 2 * mu_right)))
        exact_formula = float(
            2.0
            * math.sqrt((mean_right - mean_left) ** 2 + (std_left + std_right) ** 2)
            / math.sqrt(z_left * z_right)
        )
        actual_nuclear = float(2.0 * np.sum(singular))
        edges.append(
            EdgeAudit(
                left=left.tolist(),
                right=right.tolist(),
                rank=int(np.sum(singular > tolerance)),
                frobenius=float(np.linalg.norm(block, "fro")),
                nuclear=actual_nuclear,
                exact_formula=exact_formula,
                exact_formula_residual=abs(actual_nuclear - exact_formula),
                analytic_bound=float(
                    2.0
                    * math.sqrt(2.0)
                    * diameter
                    / math.sqrt(z_left * z_right)
                ),
            )
        )

    C = -0.5 * (H @ omega - omega @ H)
    target = np.zeros(k + 1)
    target[0], target[-1] = 1.0, -1.0
    opposite = np.fromfunction(lambda i, j: (i + j) % 2 == 1, H.shape)
    predicted = np.outer(rho, rho) / np.maximum(
        1.0, np.abs(np.subtract.outer(np.arange(k + 1), np.arange(k + 1)))
    )
    cauchy_error = np.max(np.abs(np.abs(H[opposite]) - predicted[opposite]))
    same_off_diagonal = np.fromfunction(
        lambda i, j: ((i + j) % 2 == 0) & (i != j), H.shape
    )
    zero_error = float(np.max(np.abs(H[same_off_diagonal]), initial=0.0))
    residuals = {
        "U_orthogonal": float(np.linalg.norm(U.T @ U - np.eye(k + 1), np.inf)),
        "H_involution": float(np.linalg.norm(H @ H - np.eye(k + 1), np.inf)),
        "cauchy_opposite_parity": float(cauchy_error),
        "cauchy_same_parity_zero": zero_error,
        "current_skew": float(np.linalg.norm(current + current.T, np.inf)),
        "omega_skew": float(np.linalg.norm(omega + omega.T, np.inf)),
        "C_symmetric": float(np.linalg.norm(C - C.T, np.inf)),
        "HCH_plus_C": float(np.linalg.norm(H @ C @ H + C, np.inf)),
        "diagonal": float(np.linalg.norm(np.diag(C) - target, np.inf)),
    }
    return {
        "k": k,
        "U": U,
        "H": H,
        "rho": rho,
        "current": current,
        "omega": omega,
        "C": C,
        "chain": chain,
        "edges": edges,
        "residuals": residuals,
        "nuclear_C": float(np.linalg.norm(C, "nuc")),
        "nuclear_omega": float(np.linalg.norm(omega, "nuc")),
        "ratio_C_log": float(np.linalg.norm(C, "nuc") / math.log(k + 1.0)),
    }


def construct_nearest_neighbor(k: int) -> dict:
    """Construct the conjecturally O(log k) nearest-neighbor witness."""
    U = krawtchouk_from_jacobi(k)
    m = (k - 1) // 2
    signs = np.r_[np.ones(m + 1), -np.ones(m + 1)]
    H = (U.T * signs) @ U
    omega = np.zeros_like(H)
    for t in range(k):
        omega[t, t + 1] = 1.0 / H[t, t + 1]
        omega[t + 1, t] = -omega[t, t + 1]
    current = H * omega
    C = -0.5 * (H @ omega - omega @ H)
    target = np.zeros(k + 1)
    target[0], target[-1] = 1.0, -1.0
    return {
        "nuclear_C": float(np.linalg.norm(C, "nuc")),
        "nuclear_omega": float(np.linalg.norm(omega, "nuc")),
        "frobenius_C": float(np.linalg.norm(C, "fro")),
        "rank_C": int(np.linalg.matrix_rank(C, tol=1e-9)),
        "residuals": {
            "current_skew": float(np.linalg.norm(current + current.T, np.inf)),
            "omega_skew": float(np.linalg.norm(omega + omega.T, np.inf)),
            "C_symmetric": float(np.linalg.norm(C - C.T, np.inf)),
            "HCH_plus_C": float(np.linalg.norm(H @ C @ H + C, np.inf)),
            "diagonal": float(np.linalg.norm(np.diag(C) - target, np.inf)),
        },
    }


def joint_count_kernel(k: int, d: int) -> np.ndarray:
    """Build T_d by directly counting subsets of A union B."""
    common = k - d
    probability = np.zeros((k + 1, k + 1))
    denominator = 2.0 ** (k + d)
    for a in range(common + 1):
        ca = math.comb(common, a)
        for b in range(d + 1):
            for c in range(d + 1):
                probability[a + b, a + c] += (
                    ca * math.comb(d, b) * math.comb(d, c) / denominator
                )
    pi = np.array([math.comb(k, r) / 2.0**k for r in range(k + 1)])
    return probability / np.sqrt(np.outer(pi, pi))


def common_spectrum_residual(k: int) -> float:
    U = krawtchouk_from_jacobi(k)
    worst = 0.0
    for d in range(1, k + 1):
        T = joint_count_kernel(k, d)
        lambdas = np.array(
            [
                math.comb(k - d, t) / math.comb(k, t) if t <= k - d else 0.0
                for t in range(k + 1)
            ]
        )
        worst = max(worst, float(np.linalg.norm(T - (U * lambdas) @ U.T, np.inf)))
    return worst


def explicit_dual_residual(k: int, n: int) -> dict:
    """Enumerate concepts and queries and check the original dual SDP."""
    if n < k:
        raise ValueError("n must be at least k")
    audit = construct_witness(k)
    U, C = audit["U"], audit["C"]
    B = U @ C @ U.T
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    operator_absolute = (eigenvectors * np.abs(eigenvalues)) @ eigenvectors.T
    Z = operator_absolute + B
    pi = np.array([math.comb(k, r) / 2.0**k for r in range(k + 1)])
    concepts = [sum(1 << i for i in indices) for indices in itertools.combinations(range(n), k)]
    size = len(concepts)
    constraints = np.zeros((size, size))
    objectives = np.zeros(size)
    minimum_psd_eigenvalue = math.inf
    for query in range(1 << n):
        counts = np.array([(concept & query).bit_count() for concept in concepts])
        outputs = counts >= (k + 1) // 2
        X = (2.0 ** -n) * Z[np.ix_(counts, counts)] / np.sqrt(
            np.outer(pi[counts], pi[counts])
        )
        objectives += np.diag(X)
        constraints += X * (outputs[:, None] != outputs[None, :])
        minimum_psd_eigenvalue = min(
            minimum_psd_eigenvalue, float(np.linalg.eigvalsh(X)[0])
        )
    off_diagonal = ~np.eye(size, dtype=bool)
    return {
        "n": n,
        "concepts": size,
        "constraint_residual": float(np.max(np.abs(constraints[off_diagonal] - 1.0))),
        "objective_residual": float(np.max(np.abs(objectives - np.trace(Z)))),
        "minimum_X_eigenvalue": minimum_psd_eigenvalue,
        "trace_Z": float(np.trace(Z)),
    }


def exact_symbolic_checks(k: int) -> dict:
    """Check the degree-space identities in exact SymPy arithmetic."""
    import sympy as sp
    if k > 9:
        raise ValueError("symbolic audit is intended only for small k")
    U = sp.zeros(k + 1)
    for x in range(k + 1):
        for t in range(k + 1):
            coefficient = sum(
                (-1) ** j * math.comb(x, j) * math.comb(k - x, t - j)
                for j in range(max(0, t - (k - x)), min(x, t) + 1)
            )
            U[x, t] = sp.sqrt(sp.Rational(math.comb(k, x), 2**k * math.comb(k, t))) * coefficient
    m = (k - 1) // 2
    D = sp.diag(*([1] * (m + 1) + [-1] * (m + 1)))
    H = sp.simplify(U.T * D * U)
    rho_squared = [sp.simplify((k + 1) * U[m, t] ** 2) for t in range(k + 1)]
    current, omega = sp.zeros(k + 1), sp.zeros(k + 1)
    chain = dyadic_chain(k)
    for left_np, right_np in zip(chain, chain[1:]):
        left, right = left_np.tolist(), right_np.tolist()
        z_left = sum(rho_squared[i] for i in left)
        z_right = sum(rho_squared[j] for j in right)
        for i in left:
            for j in right:
                flow = sp.simplify(rho_squared[i] * rho_squared[j] / (z_left * z_right))
                current[i, j], current[j, i] = flow, -flow
                voltage = sp.simplify(flow / H[i, j])
                omega[i, j], omega[j, i] = voltage, -voltage
    C = sp.simplify(-sp.Rational(1, 2) * (H * omega - omega * H))
    target = sp.diag(*([1] + [0] * (k - 1) + [-1]))
    cauchy = True
    for s in range(k + 1):
        for t in range(k + 1):
            if s == t:
                continue
            if (s + t) % 2 == 0:
                cauchy &= sp.simplify(H[s, t]) == 0
            else:
                cauchy &= sp.simplify(
                    H[s, t] ** 2 - rho_squared[s] * rho_squared[t] / (t - s) ** 2
                ) == 0
    return {
        "U_orthogonal": U.T * U == sp.eye(k + 1),
        "H_involution": H * H == sp.eye(k + 1),
        "cauchy_squared": bool(cauchy),
        "current_skew": current.T == -current,
        "omega_skew": omega.T == -omega,
        "C_symmetric": C.T == C,
        "HCH_plus_C_zero": sp.simplify(H * C * H + C) == sp.zeros(k + 1),
        "diagonal": sp.diag(*[C[i, i] for i in range(k + 1)]) == target,
    }


def optimize_witness(k: int) -> dict:
    """Solve the exact reduced convex problem in the two D-eigenspaces."""
    U = krawtchouk_from_jacobi(k)
    m = (k - 1) // 2
    top, bottom = U[: m + 1, :], U[m + 1 :, :]
    K = cp.Variable((m + 1, m + 1))
    target = np.zeros(k + 1)
    target[0], target[-1] = 1.0, -1.0
    constraints = []
    for t in range(k + 1):
        coefficient = np.outer(top[:, t], bottom[:, t])
        constraints.append(2.0 * cp.sum(cp.multiply(coefficient, K)) == target[t])
    problem = cp.Problem(cp.Minimize(2.0 * cp.normNuc(K)), constraints)
    try:
        value = problem.solve(
            solver="CLARABEL", tol_gap_abs=1e-9, tol_feas=1e-9, max_iter=2000
        )
    except cp.error.SolverError:
        value = problem.solve(solver="SCS", eps=2e-7, max_iters=200000)
    singular = np.linalg.svd(K.value, compute_uv=False)
    residual = max(
        abs(2.0 * np.sum(np.outer(top[:, t], bottom[:, t]) * K.value) - target[t])
        for t in range(k + 1)
    )
    return {
        "status": problem.status,
        "optimal_cost": float(value),
        "constraint_residual": float(residual),
        "singular_values_cross_block": singular.tolist(),
    }


def compact_record(
    audit: dict,
    spectrum: float | None,
    optimum: dict | None,
    explicit: dict | None,
    exact: dict | None,
    nearest: dict,
) -> dict:
    result = {
        "k": audit["k"],
        "nuclear_C": audit["nuclear_C"],
        "nuclear_omega": audit["nuclear_omega"],
        "ratio_C_log": audit["ratio_C_log"],
        "residuals": audit["residuals"],
        "edges": [asdict(edge) for edge in audit["edges"]],
        "nearest_neighbor": nearest,
    }
    if spectrum is not None:
        result["common_spectrum_residual"] = spectrum
    if optimum is not None:
        result["optimization"] = optimum
        result["construction_over_optimum"] = (
            audit["nuclear_C"] / optimum["optimal_cost"]
        )
    if explicit is not None:
        result["explicit_dual"] = explicit
    if exact is not None:
        result["exact_symbolic"] = exact
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 7, 9, 15, 31, 63, 127])
    parser.add_argument(
        "--optimize-max-k", type=int, default=15, help="optimize only for k at most this"
    )
    parser.add_argument(
        "--exact-max-k",
        type=int,
        default=5,
        help="run exact symbolic degree-space checks up to this k",
    )
    parser.add_argument(
        "--spectrum-max-k", type=int, default=9, help="directly count T_d only up to this"
    )
    parser.add_argument(
        "--explicit-max-k",
        type=int,
        default=5,
        help="enumerate the original SDP up to this k, using n=k+2",
    )
    args = parser.parse_args()
    records = []
    for k in args.k:
        audit = construct_witness(k)
        spectrum = common_spectrum_residual(k) if k <= args.spectrum_max_k else None
        optimum = optimize_witness(k) if k <= args.optimize_max_k else None
        explicit = explicit_dual_residual(k, k + 2) if k <= args.explicit_max_k else None
        exact = exact_symbolic_checks(k) if k <= args.exact_max_k else None
        nearest = construct_nearest_neighbor(k)
        records.append(compact_record(audit, spectrum, optimum, explicit, exact, nearest))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
