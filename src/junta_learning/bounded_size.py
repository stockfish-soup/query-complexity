#!/usr/bin/env python3
"""Intersection-free p=1/2 SDP for learning Majority on all |A| <= k.

For a hidden set A of size r, the oracle returns

    Maj_r(A cap S) = 1{|A cap S| >= ceil(r/2)}.

The ansatz is

    X_S[A,B] = 2^{-n} Y[(r,|A cap S|),(s,|B cap S|)],

where one global matrix Y is positive semidefinite.  This allows the entry to
depend on |A| and |B| while retaining PSD across all cardinality layers.
The optimum is independent of n once n >= 2k.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class Result:
    k: int
    status: str
    cost: float
    rank: int
    residual: float
    minimum_eigenvalue: float
    layer_costs: np.ndarray
    eigenvalues: np.ndarray
    layers: list[int]


def majority(size: int, count: int) -> int:
    """Tie-positive majority, agreeing with the even-to-odd reduction."""
    return int(count >= (size + 1) // 2)


def layer_indices(layers: list[int]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    states = [(r, a) for r in layers for a in range(r + 1)]
    return states, {state: i for i, state in enumerate(states)}


def joint_counts(r: int, s: int, d: int) -> np.ndarray:
    """Law of (|A cap S|,|B cap S|) for |A|=r, |B|=s, |A cap B|=d."""
    probability = np.zeros((r + 1, s + 1))
    denominator = 2.0 ** (r + s - d)
    for common in range(d + 1):
        common_weight = math.comb(d, common)
        for left in range(r - d + 1):
            a = common + left
            left_weight = common_weight * math.comb(r - d, left)
            for right in range(s - d + 1):
                b = common + right
                probability[a, b] += (
                    left_weight * math.comb(s - d, right) / denominator
                )
    return probability


def constraint_matrix(
    k: int,
    r: int,
    s: int,
    d: int,
    index: dict[tuple[int, int], int],
) -> np.ndarray:
    """Coefficient C with <C,Y> = separating expectation for an (r,s,d) pair."""
    dimension = len(index)
    coefficient = np.zeros((dimension, dimension))
    probability = joint_counts(r, s, d)
    for a in range(r + 1):
        for b in range(s + 1):
            if majority(r, a) == majority(s, b):
                continue
            i, j = index[(r, a)], index[(s, b)]
            if i == j:
                coefficient[i, i] += probability[a, b]
            else:
                # CVXPY's Frobenius inner product sees both symmetric entries.
                coefficient[i, j] += 0.5 * probability[a, b]
                coefficient[j, i] += 0.5 * probability[a, b]
    return coefficient


def solve(k: int, solver: str = "CLARABEL", odd_only: bool = False) -> Result:
    if k < 1:
        raise ValueError("k must be positive")
    layers = list(range(1, k + 1, 2)) if odd_only else list(range(1, k + 1))
    states, index = layer_indices(layers)
    dimension = len(states)
    y = cp.Variable((dimension, dimension), symmetric=True)
    tau = cp.Variable()
    constraints: list[cp.Constraint] = [y >> 0]
    coefficient_matrices: list[np.ndarray] = []

    for r in layers:
        pi = np.array([math.comb(r, a) / 2.0**r for a in range(r + 1)])
        diagonal = cp.hstack([y[index[(r, a)], index[(r, a)]] for a in range(r + 1)])
        constraints.append(pi @ diagonal <= tau)

    for position, r in enumerate(layers):
        for s in layers[position:]:
            maximum_d = r - 1 if r == s else r
            for d in range(maximum_d + 1):
                coefficient = constraint_matrix(k, r, s, d, index)
                coefficient_matrices.append(coefficient)
                constraints.append(cp.sum(cp.multiply(coefficient, y)) == 1.0)

    problem = cp.Problem(cp.Minimize(tau), constraints)
    if solver == "CLARABEL":
        try:
            problem.solve(
                solver="CLARABEL",
                tol_gap_abs=2e-9,
                tol_feas=2e-9,
                max_iter=1000,
            )
        except cp.error.SolverError:
            problem.solve(solver="SCS", eps=2e-6, max_iters=300000)
    else:
        problem.solve(solver="SCS", eps=2e-6, max_iters=300000)

    value = np.asarray(y.value)
    value = (value + value.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(value)
    cutoff = 2e-7 * max(1.0, float(eigenvalues[-1]))
    rank = int(np.count_nonzero(eigenvalues > cutoff))
    layer_costs = np.empty(len(layers))
    for position, r in enumerate(layers):
        layer_costs[position] = sum(
            math.comb(r, a) / 2.0**r * value[index[(r, a)], index[(r, a)]]
            for a in range(r + 1)
        )
    residual = max(
        abs(float(np.sum(coefficient * value)) - 1.0)
        for coefficient in coefficient_matrices
    )
    return Result(
        k=k,
        status=str(problem.status),
        cost=float(problem.value),
        rank=rank,
        residual=residual,
        minimum_eigenvalue=float(eigenvalues[0]),
        layer_costs=layer_costs,
        eigenvalues=eigenvalues,
        layers=layers,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--solver", choices=["CLARABEL", "SCS"], default="CLARABEL")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--odd-only", action="store_true")
    args = parser.parse_args()
    print("k dimension status cost rank residual min-eigen max-layer")
    for k in args.ks:
        result = solve(k, args.solver, args.odd_only)
        dimension = sum(r + 1 for r in result.layers)
        print(
            f"{k:2d} {dimension:3d} {result.status:18s} "
            f"{result.cost:12.8f} {result.rank:3d} "
            f"{result.residual:.2e} {result.minimum_eigenvalue:.2e} "
            f"{result.layers[int(np.argmax(result.layer_costs))]}"
        )
        if args.details:
            print("  layer-costs", " ".join(f"{x:.8f}" for x in result.layer_costs))
            print("  positive-eigenvalues", " ".join(
                f"{x:.5g}" for x in result.eigenvalues if x > 1e-8
            ))


if __name__ == "__main__":
    main()
