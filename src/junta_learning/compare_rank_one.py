#!/usr/bin/env python3
"""Compare structured exact rank-one Majority adversaries."""

from __future__ import annotations

import argparse
import itertools
import math

import numpy as np

from .majority_sdp import joint_count_kernel
from .rank_one import cost_for_zeros, exhaustive_best


def parity_cost(k: int) -> float:
    m = (k - 1) // 2
    return 2.0**k / math.comb(k - 1, m)


def parity_residual(k: int) -> float:
    m = (k - 1) // 2
    pi = np.asarray([math.comb(k, x) / 2.0**k for x in range(k + 1)])
    low = np.sqrt(pi[: m + 1])
    high = np.sqrt(pi[m + 1 :]) * (-1.0) ** np.arange(m + 1)
    c = parity_cost(k)
    return max(
        abs(2.0 * c * np.sum(joint_count_kernel(k, d)[: m + 1, m + 1 :] *
                               np.outer(low, high)) - 1.0)
        for d in range(1, k + 1)
    )


def period_four_zeros(k: int) -> tuple[int, ...]:
    return tuple(t for t in range(1, k) if t % 4 in (0, 1))


def best_periodic(k: int, period: int) -> float:
    m = (k - 1) // 2
    interior = range(1, k)
    values = []
    for residues in itertools.combinations(range(period), period // 2):
        zero_set = tuple(t for t in interior if t % period in residues)
        if len(zero_set) == m:
            values.append(cost_for_zeros(k, zero_set))
    return min(values, default=float("nan"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", nargs="+", type=int,
                        default=[3, 5, 7, 9, 11, 13, 15, 31, 63])
    args = parser.parse_args()
    print("k parity period4 period6 period10 exhaustive parity-residual")
    for k in args.ks:
        p4 = cost_for_zeros(k, period_four_zeros(k))
        best = exhaustive_best(k)[0] if k <= 15 else float("nan")
        print(f"{k:3d} {parity_cost(k):.10g} {p4:.10g} "
              f"{best_periodic(k,6):.10g} {best_periodic(k,10):.10g} "
              f"{best:.10g} {parity_residual(k):.2e}")


if __name__ == "__main__":
    main()
