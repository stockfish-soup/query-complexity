"""Bisection search for alpha thresholds in fixed-rank BM feasibility."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import query_complexity as qc


def feasible_at(
    kind: str,
    n: int,
    k: int,
    rank: int,
    alpha: float,
    *,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    epsilon: float,
    seed: int,
    adversary_value: float,
    warm_factors,
):
    inputs = qc.all_binary_inputs(n)
    layers = qc.EXACT_layers(n, k) if kind == "exact" else qc.THRESHOLD_layers(n, k)
    outputs = [layers[sum(x)] for x in inputs]
    orbit = qc.solve_symmetric_trace_minimized_dual(
        n,
        layers,
        alpha=alpha,
        adversary_value=adversary_value,
        epsilon=epsilon,
    )
    symmetric_start = qc.expand_orbit_solution(orbit, psd_tol=1e-7)
    result = qc.solve_fixed_rank_dual_feasibility(
        inputs,
        outputs,
        rank=rank,
        alpha=alpha,
        adversary_value=adversary_value,
        epsilon=epsilon,
        initial_solution=symmetric_start if warm_factors is None else None,
        initial_factors=warm_factors,
        restarts=restarts,
        random_seed=seed,
        max_nfev=max_nfev,
        feasibility_tol=feasibility_tol,
    )
    return result, symmetric_start


def bisect_case(
    kind: str,
    n: int,
    k: int,
    rank: int,
    lo: float,
    hi: float,
    *,
    iters: int,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    epsilon: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    warm_factors = None
    best_factors = None
    layers = qc.EXACT_layers(n, k) if kind == "exact" else qc.THRESHOLD_layers(n, k)
    adversary_value = qc.solve_symmetric_trace_minimized_dual(
        n,
        layers,
        alpha=1.0,
        epsilon=epsilon,
    ).objective_value
    confirmed_lo = lo
    confirmed_hi = hi

    for step in range(iters):
        mid = 0.5 * (lo + hi)
        print(
            f"{kind.upper()}({n},{k}) rank<={rank}: testing alpha={mid:.10g} "
            f"in [{lo:.8g},{hi:.8g}]",
            flush=True,
        )
        result, _ = feasible_at(
            kind,
            n,
            k,
            rank,
            mid,
            restarts=restarts,
            max_nfev=max_nfev,
            feasibility_tol=feasibility_tol,
            epsilon=epsilon,
            seed=90_000 + 1_000 * n + 100 * k + 10 * rank + step,
            adversary_value=adversary_value,
            warm_factors=warm_factors,
        )
        rows.append(
            {
                "kind": kind,
                "n": n,
                "k": k,
                "rank": rank,
                "alpha": mid,
                "feasible": result.feasible,
                "max_equality_residual": result.max_equality_residual,
                "max_cap_violation": result.max_cap_violation,
                "attempts": result.attempts,
                "total_nfev": result.total_nfev,
            }
        )
        print(
            f"  -> {result.status}; eq={result.max_equality_residual:.3e}, "
            f"cap={result.max_cap_violation:.3e}",
            flush=True,
        )
        if result.feasible:
            hi = mid
            confirmed_hi = mid
            best_factors = result.z
            warm_factors = result.z
        else:
            lo = mid
            confirmed_lo = mid
            warm_factors = best_factors
    print(
        f"{kind.upper()}({n},{k}) rank<={rank}: "
        f"confirmed bracket ({confirmed_lo:.10g}, {confirmed_hi:.10g}]",
        flush=True,
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("exact", "threshold"), default="exact")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--cases", default="1:2:1:1.5,1:1:2:3,2:2:1:1.5,2:1:2:3")
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--restarts", type=int, default=0)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--csv", type=Path, default=Path("exact_alpha_bisect_n5.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for spec in args.cases.split(","):
        k_s, rank_s, lo_s, hi_s = spec.split(":")
        rows.extend(
            bisect_case(
                args.kind,
                args.n,
                int(k_s),
                int(rank_s),
                float(lo_s),
                float(hi_s),
                iters=args.iters,
                restarts=args.restarts,
                max_nfev=args.max_nfev,
                feasibility_tol=args.feasibility_tol,
                epsilon=args.epsilon,
            )
        )

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
