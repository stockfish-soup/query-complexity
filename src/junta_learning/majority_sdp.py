#!/usr/bin/env python3
"""Direct count-kernel SDP for learning odd Majority (no H parametrization)."""

from __future__ import annotations

import argparse
import math

import cvxpy as cp
import numpy as np


def joint_count_kernel(k: int, d: int) -> np.ndarray:
    """T_d[x,y]=Pr(X=x,Y=y)/sqrt(pi_x pi_y), computed combinatorially."""
    pi = np.array([math.comb(k, x) / 2.0**k for x in range(k + 1)])
    probability = np.zeros((k + 1, k + 1))
    # X=C+P and Y=C+Q, with C,P,Q independent fair binomials of
    # sizes k-d,d,d respectively.
    denominator = 2.0 ** (k + d)
    for c in range(k - d + 1):
        common = math.comb(k - d, c)
        for p in range(d + 1):
            x = c + p
            first = common * math.comb(d, p)
            for q in range(d + 1):
                y = c + q
                probability[x, y] += first * math.comb(d, q) / denominator
    return probability / np.sqrt(np.outer(pi, pi))


def solve(k: int) -> tuple[float, np.ndarray, str]:
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be positive and odd")
    m = (k - 1) // 2
    cross = np.zeros((k + 1, k + 1))
    cross[: m + 1, m + 1 :] = 1.0
    cross[m + 1 :, : m + 1] = 1.0
    kernels = [joint_count_kernel(k, d) * cross for d in range(1, k + 1)]
    z = cp.Variable((k + 1, k + 1), symmetric=True)
    constraints = [z >> 0]
    constraints += [cp.sum(cp.multiply(kernel, z)) == 1 for kernel in kernels]
    problem = cp.Problem(cp.Minimize(cp.trace(z)), constraints)
    try:
        problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1e-9,
            tol_feas=1e-9,
            max_iter=1000,
        )
    except cp.error.SolverError:
        problem.solve(solver="SCS", eps=1e-7, max_iters=200000)
    return float(problem.value), np.asarray(z.value), str(problem.status)


def solve_cross_block(k: int) -> tuple[float, np.ndarray, str]:
    """Equivalent trace-minimization written as a nuclear-norm problem."""
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be positive and odd")
    m = (k - 1) // 2
    kernels = [
        joint_count_kernel(k, d)[: m + 1, m + 1 :]
        for d in range(1, k + 1)
    ]
    q = cp.Variable((m + 1, m + 1))
    constraints = [2 * cp.sum(cp.multiply(kernel, q)) == 1 for kernel in kernels]
    problem = cp.Problem(cp.Minimize(2 * cp.normNuc(q)), constraints)
    try:
        problem.solve(
            solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, max_iter=2000
        )
    except cp.error.SolverError:
        problem.solve(solver="SCS", eps=1e-7, max_iters=300000)
    return float(problem.value), np.asarray(q.value), str(problem.status)


def audit(k: int) -> None:
    value, cross_block, status = solve_cross_block(k)
    singular_values = np.linalg.svd(cross_block, compute_uv=False)
    rank = int(np.sum(singular_values > 1e-7 * max(1.0, singular_values[0])))
    print(
        f"k={k} status={status} cost={value:.10g} rank={rank} "
        f"cross-sv={singular_values[:min(6,len(singular_values))]}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5, 7, 9, 11, 15])
    args = parser.parse_args()
    for k in args.ks:
        audit(k)


if __name__ == "__main__":
    main()
