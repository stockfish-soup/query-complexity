"""Adaptive alpha/rank BM search.

For each instance, the script evaluates the minimum feasible rank at interval
endpoints and midpoint.  It then recursively searches the left and right
subintervals, stopping when both sides of an interval have the same minimum
rank, or when the alpha interval is below tolerance/max depth.

Every rank test is checkpointed immediately.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_bm_n7_pipeline import (
    SWEEP_FIELDS,
    CsvCheckpoint,
    Instance,
    append_sweep_row,
    make_instance,
    symmetric_start,
    test_rank,
)


SUMMARY_FIELDS = [
    "kind",
    "n",
    "k",
    "alpha",
    "min_rank",
    "benchmark_rank",
]


@dataclass(frozen=True)
class Point:
    alpha: float
    rank: int


def rank_search(
    instance: Instance,
    alpha: float,
    *,
    upper_rank: int,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    checkpoint: CsvCheckpoint,
    warm_starts: dict[int, list[np.ndarray]],
) -> int:
    start = symmetric_start(instance, alpha, epsilon)
    lo, hi = 1, max(1, upper_rank)
    best: int | None = None
    previous_feasible = None
    print(
        f"{instance.kind.upper()}({instance.n},{instance.k}) alpha={alpha:.10g}: "
        f"rank search in [1,{hi}]",
        flush=True,
    )
    while lo <= hi:
        rank = (lo + hi) // 2
        initial_factors = warm_starts.get(rank) or previous_feasible
        print(f"  rank<={rank}: starting", flush=True)
        result = test_rank(
            instance,
            rank,
            alpha,
            epsilon=epsilon,
            restarts=restarts,
            max_nfev=max_nfev,
            feasibility_tol=feasibility_tol,
            seed=110_000 + 1_000 * instance.k + 100 * rank + round(10000 * alpha),
            initial_solution=start,
            initial_factors=initial_factors,
        )
        append_sweep_row(checkpoint, instance, alpha, rank, result)
        print(
            f"  rank<={rank}: {result.status}; "
            f"eq={result.max_equality_residual:.3e}, "
            f"cap={result.max_cap_violation:.3e}",
            flush=True,
        )
        if result.feasible:
            warm_starts[rank] = result.z
            previous_feasible = result.z
            best = rank
            hi = rank - 1
        else:
            lo = rank + 1
    if best is None:
        # We failed to find a below-benchmark witness.  The benchmark rank is
        # always feasible by the symmetric construction, so use it as fallback.
        best = instance.benchmark_rank
        print(f"  no below-benchmark feasible rank found; using benchmark {best}", flush=True)
    else:
        print(f"  minimum rank found: {best}", flush=True)
    return best


def adaptive_search(
    instance: Instance,
    left: Point,
    right: Point,
    *,
    depth: int,
    max_depth: int,
    alpha_tol: float,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    checkpoint: CsvCheckpoint,
    warm_starts: dict[int, list[np.ndarray]],
    points: dict[float, int],
) -> None:
    if left.rank == right.rank:
        return
    if depth >= max_depth or right.alpha - left.alpha <= alpha_tol:
        return
    mid_alpha = 0.5 * (left.alpha + right.alpha)
    if mid_alpha in points:
        mid = Point(mid_alpha, points[mid_alpha])
    else:
        # Ranks are nonincreasing in alpha, so the left endpoint rank is a
        # safe upper bound for the midpoint.
        mid_rank = rank_search(
            instance,
            mid_alpha,
            upper_rank=left.rank,
            epsilon=epsilon,
            restarts=restarts,
            max_nfev=max_nfev,
            feasibility_tol=feasibility_tol,
            checkpoint=checkpoint,
            warm_starts=warm_starts,
        )
        points[mid_alpha] = mid_rank
        mid = Point(mid_alpha, mid_rank)
    adaptive_search(
        instance,
        left,
        mid,
        depth=depth + 1,
        max_depth=max_depth,
        alpha_tol=alpha_tol,
        epsilon=epsilon,
        restarts=restarts,
        max_nfev=max_nfev,
        feasibility_tol=feasibility_tol,
        checkpoint=checkpoint,
        warm_starts=warm_starts,
        points=points,
    )
    adaptive_search(
        instance,
        mid,
        right,
        depth=depth + 1,
        max_depth=max_depth,
        alpha_tol=alpha_tol,
        epsilon=epsilon,
        restarts=restarts,
        max_nfev=max_nfev,
        feasibility_tol=feasibility_tol,
        checkpoint=checkpoint,
        warm_starts=warm_starts,
        points=points,
    )


def write_summary(path: Path, all_points: dict[tuple[str, int, int], dict[float, int]], benchmarks: dict[tuple[str, int, int], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for key in sorted(all_points):
            kind, n, k = key
            running = benchmarks[key]
            for alpha, rank in sorted(all_points[key].items()):
                running = min(running, rank)
                writer.writerow(
                    {
                        "kind": kind,
                        "n": n,
                        "k": k,
                        "alpha": alpha,
                        "min_rank": running,
                        "benchmark_rank": benchmarks[key],
                    }
                )


def plot_summary(summary_csv: Path, prefix: Path) -> None:
    rows = []
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], int(row["n"]), int(row["k"]))].append(row)
    for suffix, xlim in [
        ("full", None),
        ("near1", (1.0, 1.15)),
        ("rank2", (1.0, 2.5)),
        ("rank1", (2.0, None)),
    ]:
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, key in enumerate(sorted(grouped)):
            case = sorted(grouped[key], key=lambda row: float(row["alpha"]))
            xs = [float(row["alpha"]) for row in case]
            ys = [int(row["min_rank"]) for row in case]
            ax.step(xs, ys, where="post", marker="o", color=colors[idx % len(colors)], label=f"{key[0]} n={key[1]} k={key[2]}")
        if xlim is not None:
            lo, hi = xlim
            if hi is None:
                max_alpha = max(float(row["alpha"]) for row in rows)
                hi = max_alpha * 1.02
            ax.set_xlim(lo, hi)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("lowest feasible rank found")
        ax.set_title(f"Adaptive BM alpha/rank search ({suffix})")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = prefix.with_name(f"{prefix.name}_{suffix}.png")
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("exact", "threshold"), required=True)
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--ks", required=True)
    parser.add_argument("--alpha-lo", type=float, default=1.0)
    parser.add_argument("--alpha-hi", type=float, default=10.0)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--alpha-tol", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--restarts", type=int, default=0)
    parser.add_argument("--max-nfev", type=int, default=400)
    parser.add_argument("--out-prefix", type=Path, default=Path("bm_adaptive"))
    args = parser.parse_args()

    rank_csv = args.out_prefix.with_name(f"{args.out_prefix.name}_rank_tests.csv")
    summary_csv = args.out_prefix.with_name(f"{args.out_prefix.name}_summary.csv")
    for path in [rank_csv, summary_csv]:
        if path.exists():
            path.unlink()
    checkpoint = CsvCheckpoint(rank_csv, SWEEP_FIELDS)
    all_points: dict[tuple[str, int, int], dict[float, int]] = {}
    benchmarks: dict[tuple[str, int, int], int] = {}

    for k in sorted({int(part.strip()) for part in args.ks.split(",") if part.strip()}):
        instance = make_instance(args.kind, args.n, k, args.epsilon)
        key = (instance.kind, instance.n, instance.k)
        benchmarks[key] = instance.benchmark_rank
        warm_starts: dict[int, list[np.ndarray]] = {}
        points: dict[float, int] = {args.alpha_lo: instance.benchmark_rank}
        right_rank = rank_search(
            instance,
            args.alpha_hi,
            upper_rank=instance.benchmark_rank - 1,
            epsilon=args.epsilon,
            restarts=args.restarts,
            max_nfev=args.max_nfev,
            feasibility_tol=args.feasibility_tol,
            checkpoint=checkpoint,
            warm_starts=warm_starts,
        )
        points[args.alpha_hi] = right_rank
        adaptive_search(
            instance,
            Point(args.alpha_lo, points[args.alpha_lo]),
            Point(args.alpha_hi, points[args.alpha_hi]),
            depth=0,
            max_depth=args.max_depth,
            alpha_tol=args.alpha_tol,
            epsilon=args.epsilon,
            restarts=args.restarts,
            max_nfev=args.max_nfev,
            feasibility_tol=args.feasibility_tol,
            checkpoint=checkpoint,
            warm_starts=warm_starts,
            points=points,
        )
        all_points[key] = points
        write_summary(summary_csv, all_points, benchmarks)
    plot_summary(summary_csv, args.out_prefix)
    print(f"Wrote {rank_csv}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
