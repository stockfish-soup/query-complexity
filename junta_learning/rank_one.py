#!/usr/bin/env python3
"""Explicit mod-four rank-one witnesses for odd Majority.

The construction is performed in Fourier degree space.  It uses the
weighted-Cauchy formula for the majority sign matrix and rational
interpolation weights, so no numerical nullspace is needed.
"""

from __future__ import annotations

import argparse
import itertools
import math
from fractions import Fraction

import numpy as np
from scipy.linalg import eigh_tridiagonal


def rho_squared(k: int, t: int) -> Fraction:
    """Square of rho_t = sqrt(k+1)|U[(k-1)/2,t]|."""
    m = (k - 1) // 2
    if t % 2 == 0:
        i = t // 2
        return Fraction(
            math.comb(2 * i, i)
            * (2 * m - 2 * i + 1)
            * math.comb(2 * m - 2 * i, m - i),
            4**m,
        )
    # rho_{2i+1}^2/rho_{2i}^2=(2i+1)/(k-2i)
    i = (t - 1) // 2
    return rho_squared(k, 2 * i) * Fraction(2 * i + 1, k - 2 * i)


def cauchy_null_weights(nodes: list[int], zeros: list[int]) -> dict[int, Fraction]:
    """Weights y_t with sum_t y_t/(t-s)=0 for every prescribed zero s."""
    if len(nodes) != len(zeros) + 1:
        raise ValueError("need one more support node than zero")
    ans = {}
    for t in nodes:
        numerator = math.prod(t - s for s in zeros)
        denominator = math.prod(t - u for u in nodes if u != t)
        ans[t] = Fraction(numerator, denominator)
    return ans


def plus_eigen_y(k: int, zeros: list[int]) -> list[Fraction]:
    """Return y_t=delta_t rho_t alpha_t for a prescribed zero pattern."""
    if k < 1 or k % 2 == 0:
        raise ValueError("k must be positive and odd")
    zero_set = set(zeros)
    support = [t for t in range(k + 1) if t not in zero_set]
    even_support = [t for t in support if t % 2 == 0]
    odd_support = [t for t in support if t % 2 == 1]
    even_zeros = [t for t in zeros if t % 2 == 0]
    odd_zeros = [t for t in zeros if t % 2 == 1]

    y_even = cauchy_null_weights(even_support, odd_zeros)
    y_odd = cauchy_null_weights(odd_support, even_zeros)
    even_scale = 1 / y_even[0]
    y_even = {t: even_scale * value for t, value in y_even.items()}

    # The s=0 eigenvector equation fixes the relative parity scale.
    odd_sum = sum((value / t for t, value in y_odd.items()), Fraction())
    odd_scale = Fraction(1, 1) / (rho_squared(k, 0) * odd_sum)
    y_odd = {t: odd_scale * value for t, value in y_odd.items()}

    y = [Fraction() for _ in range(k + 1)]
    for t, value in y_even.items():
        y[t] = value
    for t, value in y_odd.items():
        y[t] = value
    return y


def explicit_y(k: int) -> list[Fraction]:
    """The first mod-four eigenvector, retained as a convenient alias."""
    zeros = [t for t in range(1, k) if t % 4 in (0, 1)]
    return plus_eigen_y(k, zeros)


def alpha_from_y(k: int, y: list[Fraction]) -> np.ndarray:
    rho2 = np.array([float(rho_squared(k, t)) for t in range(k + 1)])
    delta = np.array([1.0 if t % 4 in (0, 1) else -1.0 for t in range(k + 1)])
    return np.array([float(value) for value in y]) / (delta * np.sqrt(rho2))


def alpha_beta(k: int) -> tuple[np.ndarray, np.ndarray]:
    zeros_a = [t for t in range(1, k) if t % 4 in (0, 1)]
    zeros_ap = (
        zeros_a
        if k % 4 == 3
        else [t for t in range(1, k) if t % 4 in (2, 3)]
    )
    alpha = alpha_from_y(k, plus_eigen_y(k, zeros_a))
    alpha_prime = alpha_from_y(k, plus_eigen_y(k, zeros_ap))
    beta = ((-1.0) ** np.arange(k + 1)) * alpha_prime[::-1]
    # The two endpoint ratios agree; a harmless scaling makes this exact
    # despite floating conversion of large rational products.
    beta *= (alpha[0] * beta[-1]) / (-alpha[-1] * beta[0])
    return alpha, beta


def squared_shape_ratio(k: int, y: list[Fraction]) -> Fraction:
    """Return ||alpha||^2/|alpha_0 alpha_k| exactly."""
    norm_squared = sum((y[t] ** 2 / rho_squared(k, t) for t in range(k + 1)), Fraction())
    return abs(norm_squared * rho_squared(k, 0) / (y[0] * y[k]))


def exact_cost_squared(k: int) -> Fraction:
    zeros_a = [t for t in range(1, k) if t % 4 in (0, 1)]
    zeros_ap = zeros_a if k % 4 == 3 else [t for t in range(1, k) if t % 4 in (2, 3)]
    return squared_shape_ratio(k, plus_eigen_y(k, zeros_a)) * squared_shape_ratio(
        k, plus_eigen_y(k, zeros_ap)
    )


def exact_cost(k: int) -> float:
    return math.sqrt(float(exact_cost_squared(k)))


def cost_for_zeros(k: int, zeros: tuple[int, ...]) -> float:
    """Cost of the rank-one family member indexed by an m-subset of [1,k-1]."""
    interior = set(range(1, k))
    reflected = {k - t for t in zeros}
    partner = sorted(interior - reflected)
    first = squared_shape_ratio(k, plus_eigen_y(k, list(zeros)))
    second = squared_shape_ratio(k, plus_eigen_y(k, partner))
    return math.sqrt(float(first * second))


def exhaustive_best(k: int) -> tuple[float, tuple[int, ...]]:
    m = (k - 1) // 2
    return min(
        (cost_for_zeros(k, zeros), zeros)
        for zeros in itertools.combinations(range(1, k), m)
    )


def count_witness(k: int) -> np.ndarray:
    """Return z with Z=zz.T feasible for the fixed-size majority count SDP.

    The squared norm is the explicit period-four cost. For a query S,
    X_S[A,B] = 2**(-n) z[a] z[b] / sqrt(pi[a] pi[b]), where
    a=|A intersect S|, b=|B intersect S|, pi[a]=binom(k,a)/2**k.
    """
    alpha, beta = alpha_beta(k)
    x = np.arange(k + 1, dtype=float)
    _, U = eigh_tridiagonal(
        np.full(k + 1, k / 2.0),
        -0.5 * np.sqrt((x[:-1] + 1.0) * (k - x[:-1])),
    )
    U *= np.where(U[0, :] < 0.0, -1.0, 1.0)
    cut = (k + 1) // 2
    low = (U @ alpha)[:cut]
    high = (U @ beta)[cut:]
    scale = 1.0 / (2.0 * alpha[0] * beta[0])
    low_norm, high_norm = np.linalg.norm(low), np.linalg.norm(high)
    return np.r_[
        math.sqrt(abs(scale) * high_norm / low_norm) * low,
        np.sign(scale) * math.sqrt(abs(scale) * low_norm / high_norm) * high,
    ]


def numerical_audit(k: int) -> dict[str, float]:
    alpha, beta = alpha_beta(k)

    x = np.arange(k + 1, dtype=float)
    diagonal = np.full(k + 1, k / 2.0)
    jacobi_off = -0.5 * np.sqrt((x[:-1] + 1.0) * (k - x[:-1]))
    eigenvalues, U = eigh_tridiagonal(diagonal, jacobi_off)
    if np.max(np.abs(eigenvalues - np.arange(k + 1))) > 2e-10:
        raise ArithmeticError("Jacobi eigenvalues are not 0,...,k")
    U *= np.where(U[0, :] < 0.0, -1.0, 1.0)
    signs = np.r_[np.ones((k + 1) // 2), -np.ones((k + 1) // 2)]
    H = (U.T * signs) @ U

    scale = 1.0 / (2.0 * alpha[0] * beta[0])
    C = scale * (np.outer(alpha, beta) + np.outer(beta, alpha))
    degree_residual = np.max(np.abs(np.diag(C) - np.r_[1.0, np.zeros(k - 1), -1.0]))
    return {
        "cost": exact_cost(k),
        "eigen_residual": float(np.max(np.abs(H @ alpha - alpha))),
        "beta_residual": float(np.max(np.abs(H @ beta + beta))),
        "degree_residual": float(degree_residual),
        "anti_residual": float(np.max(np.abs(H @ C @ H + C))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 5, 7, 15, 31, 63])
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--exhaustive", action="store_true")
    args = parser.parse_args()
    print("k cost cost/log(k+1) eig-res beta-res degree-res anti-res")
    for k in args.ks:
        row = numerical_audit(k)
        print(
            f"{k} {row['cost']:.12g} {row['cost']/math.log(k+1):.9g} "
            f"{row['eigen_residual']:.2e} {row['beta_residual']:.2e} "
            f"{row['degree_residual']:.2e} {row['anti_residual']:.2e}"
        )
        if args.exact:
            print(exact_cost_squared(k))
        if args.exhaustive:
            if k > 17:
                print("exhaustive search skipped (k>17)")
            else:
                best_cost, zeros = exhaustive_best(k)
                print(f"best={best_cost:.12g} zeros={zeros}")


if __name__ == "__main__":
    main()
