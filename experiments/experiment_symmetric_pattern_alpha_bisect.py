"""Alpha bisection for selected symmetric orbit block-rank patterns."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from math import isclose
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "query-complexity-matplotlib"),
)

import query_complexity as qc


FIELDS = [
    "kind",
    "n",
    "k",
    "target_rank",
    "alpha",
    "lo",
    "hi",
    "pattern",
    "multiplicities",
    "rank_bound",
    "actual_block_ranks",
    "actual_rank",
    "feasible",
    "max_equality_residual",
    "max_block_residual",
    "max_cap_violation",
    "attempts",
    "total_nfev",
]


def layers_for(kind: str, n: int, k: int) -> dict[int, int]:
    if kind == "exact":
        return qc.EXACT_layers(n, k)
    if kind == "threshold":
        return qc.THRESHOLD_layers(n, k)
    raise ValueError(kind)


def multiplicities_for(n: int, num_blocks: int) -> list[int]:
    N = n - 1
    return [
        qc.binom_safe(N, r) - qc.binom_safe(N, r - 1)
        for r in range(N // 2 + 1)
    ][:num_blocks]


def weighted_rank(pattern: tuple[int, ...], multiplicities: list[int]) -> int:
    return int(sum(rank * mult for rank, mult in zip(pattern, multiplicities)))


def patterns_with_total_rank(
    num_blocks: int,
    multiplicities: list[int],
    target_rank: int,
    max_block_rank: int,
) -> list[tuple[int, ...]]:
    patterns: list[tuple[int, ...]] = []

    def rec(idx: int, remaining: int, current: list[int]) -> None:
        if idx == num_blocks:
            if remaining == 0 and any(current):
                patterns.append(tuple(current))
            return
        mult = multiplicities[idx]
        for rank in range(max_block_rank + 1):
            cost = mult * rank
            if cost <= remaining:
                current.append(rank)
                rec(idx + 1, remaining - cost, current)
                current.pop()

    rec(0, target_rank, [])
    return sorted(patterns)


def parse_patterns(spec: str, num_blocks: int) -> list[tuple[int, ...]]:
    patterns = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        pattern = tuple(int(part.strip()) for part in item.split(";"))
        if len(pattern) != num_blocks:
            raise ValueError(
                f"pattern {item!r} has length {len(pattern)}, expected {num_blocks}"
            )
        patterns.append(pattern)
    return list(dict.fromkeys(patterns))


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def feasible_any_pattern(
    *,
    kind: str,
    n: int,
    k: int,
    target_rank: int,
    alpha: float,
    patterns: list[tuple[int, ...]],
    multiplicities: list[int],
    layers: dict[int, int],
    adversary_value: float,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    solver: str,
    csv_path: Path,
) -> bool:
    convex = qc.solve_symmetric_trace_minimized_dual(
        n,
        layers,
        alpha=alpha,
        adversary_value=adversary_value,
        epsilon=epsilon,
        solver=solver,
    )
    found = False
    for pattern in patterns:
        print(
            f"{kind.upper()}({n},{k}) alpha={alpha:.10g} pattern={';'.join(map(str, pattern))}: starting",
            flush=True,
        )
        result = qc.solve_symmetric_fixed_rank_dual_feasibility(
            n,
            layers,
            block_rank=max(pattern) if max(pattern) > 0 else 1,
            block_ranks=pattern,
            alpha=alpha,
            adversary_value=adversary_value,
            epsilon=epsilon,
            initial_solution=convex,
            restarts=restarts,
            random_seed=170_000 + 1_000 * k + 100 * target_rank + round(1000 * alpha),
            max_nfev=max_nfev,
            feasibility_tol=feasibility_tol,
            solver=solver,
        )
        actual_block_ranks = [
            qc.numerical_psd_rank(block, relative_tol=1e-7, absolute_tol=1e-7)
            for block in result.block_values
        ]
        actual_rank = weighted_rank(tuple(actual_block_ranks), multiplicities)
        append_row(
            csv_path,
            {
                "kind": kind,
                "n": n,
                "k": k,
                "target_rank": target_rank,
                "alpha": alpha,
                "lo": "",
                "hi": "",
                "pattern": ";".join(map(str, pattern)),
                "multiplicities": ";".join(map(str, multiplicities)),
                "rank_bound": weighted_rank(pattern, multiplicities),
                "actual_block_ranks": ";".join(map(str, actual_block_ranks)),
                "actual_rank": actual_rank if result.feasible else "",
                "feasible": result.feasible,
                "max_equality_residual": result.max_equality_residual,
                "max_block_residual": result.max_block_residual,
                "max_cap_violation": result.max_cap_violation,
                "attempts": result.attempts,
                "total_nfev": result.total_nfev,
            },
        )
        print(
            f"  -> {result.status}; eq={result.max_equality_residual:.3e}, "
            f"block={result.max_block_residual:.3e}, cap={result.max_cap_violation:.3e}",
            flush=True,
        )
        if result.feasible:
            found = True
            break
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("exact", "threshold"), default="exact")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--ks", required=True)
    parser.add_argument("--target-rank", type=int, default=2)
    parser.add_argument("--alpha-lo", type=float, default=1.0)
    parser.add_argument("--alpha-hi", type=float, default=5.0)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-block-rank", type=int, default=2)
    parser.add_argument(
        "--patterns",
        default=None,
        help="comma-separated explicit block-rank patterns, e.g. '2;0;0'",
    )
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--restarts", type=int, default=0)
    parser.add_argument("--max-nfev", type=int, default=1200)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--csv", type=Path, default=Path("symmetric_pattern_alpha_bisect.csv"))
    args = parser.parse_args()

    if args.csv.exists():
        args.csv.unlink()
    for k in sorted({int(value.strip()) for value in args.ks.split(",") if value.strip()}):
        layers = layers_for(args.kind, args.n, k)
        base = qc.solve_symmetric_trace_minimized_dual(
            args.n,
            layers,
            alpha=1.0,
            epsilon=args.epsilon,
            solver=args.solver,
        )
        adversary_value = base.objective_value
        multiplicities = multiplicities_for(args.n, len(base.block_values))
        if args.patterns:
            patterns = parse_patterns(args.patterns, len(base.block_values))
        else:
            patterns = patterns_with_total_rank(
                len(base.block_values),
                multiplicities,
                args.target_rank,
                args.max_block_rank,
            )
        if not patterns:
            print(
                f"{args.kind.upper()}({args.n},{k}): no block-rank pattern has total rank {args.target_rank} "
                f"with multiplicities {multiplicities}",
                flush=True,
            )
            continue
        print(
            f"{args.kind.upper()}({args.n},{k}): total-rank-{args.target_rank} patterns "
            f"{[';'.join(map(str, pattern)) for pattern in patterns]} with multiplicities {multiplicities}",
            flush=True,
        )
        lo = args.alpha_lo
        hi = args.alpha_hi
        hi_feasible = feasible_any_pattern(
            kind=args.kind,
            n=args.n,
            k=k,
            target_rank=args.target_rank,
            alpha=hi,
            patterns=patterns,
            multiplicities=multiplicities,
            layers=layers,
            adversary_value=adversary_value,
            epsilon=args.epsilon,
            restarts=args.restarts,
            max_nfev=args.max_nfev,
            feasibility_tol=args.feasibility_tol,
            solver=args.solver,
            csv_path=args.csv,
        )
        if not hi_feasible:
            print(
                f"{args.kind.upper()}({args.n},{k}): rank {args.target_rank} not found at alpha_hi={hi}",
                flush=True,
            )
            continue
        for _ in range(args.iters):
            mid = 0.5 * (lo + hi)
            feasible = feasible_any_pattern(
                kind=args.kind,
                n=args.n,
                k=k,
                target_rank=args.target_rank,
                alpha=mid,
                patterns=patterns,
                multiplicities=multiplicities,
                layers=layers,
                adversary_value=adversary_value,
                epsilon=args.epsilon,
                restarts=args.restarts,
                max_nfev=args.max_nfev,
                feasibility_tol=args.feasibility_tol,
                solver=args.solver,
                csv_path=args.csv,
            )
            if feasible:
                hi = mid
            else:
                lo = mid
            print(
                f"{args.kind.upper()}({args.n},{k}) rank<={args.target_rank}: bracket ({lo:.10g}, {hi:.10g}]",
                flush=True,
            )


if __name__ == "__main__":
    main()
