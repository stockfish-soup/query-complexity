"""Nonsymmetric fixed-rank search for THRESHOLD(5,k)."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from math import comb
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "query-complexity-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import query_complexity as qc


def alpha_grid(n: int, k: int, points: int) -> list[float]:
    adversary = qc.threshold_adversary_value(n, k)
    alpha_max = n / adversary
    values = set(np.linspace(1.0, alpha_max, points))
    for value in (1.01, 1.02, 1.03, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20):
        if value <= alpha_max:
            values.add(value)
    return sorted(float(value) for value in values)


def run_case(
    k: int,
    *,
    points: int,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
) -> list[dict[str, object]]:
    n = 5
    benchmark_rank = comb(n - 1, k - 1)
    ranks = list(range(1, benchmark_rank))
    if not ranks:
        return []

    inputs = qc.all_binary_inputs(n)
    outputs = [int(sum(x) >= k) for x in inputs]
    phase_adversary = qc.threshold_adversary_value(n, k) / 2.0
    rows: list[dict[str, object]] = []
    warm_starts: dict[int, list[np.ndarray]] = {}

    for alpha in alpha_grid(n, k, points):
        convex = qc.solve_trace_minimized_dual(
            inputs,
            outputs,
            alpha=alpha,
            adversary_value=phase_adversary,
            epsilon=epsilon,
            psd_tol=1e-7,
        )
        for rank in ranks:
            initial_factors = warm_starts.get(rank)
            result = qc.solve_fixed_rank_dual_feasibility(
                inputs,
                outputs,
                rank=rank,
                alpha=alpha,
                adversary_value=phase_adversary,
                epsilon=epsilon,
                initial_solution=None if initial_factors is not None else convex,
                initial_factors=initial_factors,
                restarts=restarts,
                random_seed=50_000 + 1_000 * k + 100 * rank + round(1000 * alpha),
                max_nfev=max_nfev,
                feasibility_tol=feasibility_tol,
            )
            if result.feasible:
                warm_starts[rank] = result.z
            rows.append(
                {
                    "n": n,
                    "k": k,
                    "alpha": alpha,
                    "threshold_rank": benchmark_rank,
                    "rank_bound": rank,
                    "feasible": result.feasible,
                    "coordinate_ranks": ";".join(
                        str(qc.numerical_psd_rank(gram)) for gram in result.grams
                    ),
                    "trace": result.trace_value,
                    "max_equality_residual": result.max_equality_residual,
                    "max_cap_violation": result.max_cap_violation,
                    "status": result.status,
                    "attempts": result.attempts,
                    "total_nfev": result.total_nfev,
                }
            )
            print(
                f"THRESHOLD(5,{k}) alpha={alpha:.6g}, rank<={rank} "
                f"(beat {benchmark_rank}): {result.status}; "
                f"eq={result.max_equality_residual:.2e}, cap={result.max_cap_violation:.2e}"
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_min_rank(rows: list[dict[str, object]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, k in zip(colors, sorted({int(row["k"]) for row in rows})):
        case = [row for row in rows if int(row["k"]) == k]
        alphas = sorted({float(row["alpha"]) for row in case})
        benchmark = int(case[0]["threshold_rank"])
        minimums = []
        for alpha in alphas:
            feasible = [
                int(row["rank_bound"])
                for row in case
                if float(row["alpha"]) == alpha and str(row["feasible"]) == "True"
            ]
            minimums.append(min(feasible) if feasible else np.nan)

        ax.plot(
            alphas,
            minimums,
            "o-",
            color=color,
            label=rf"$k={k}$: min feasible rank below ${benchmark}$",
        )
        ax.axhline(
            benchmark,
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.55,
            label=rf"$k={k}$ threshold rank ${benchmark}$",
        )

        not_found = [
            (float(row["alpha"]), int(row["rank_bound"]))
            for row in case
            if str(row["feasible"]) != "True"
        ]
        if not_found:
            ax.scatter(
                [a for a, _ in not_found],
                [r for _, r in not_found],
                marker="x",
                s=28,
                color=color,
                alpha=0.38,
            )

    ax.set_title(r"Nonsymmetric fixed-rank search for $\mathrm{THRESHOLD}(5,k)$")
    ax.set_xlabel(r"query relaxation $\alpha$")
    ax.set_ylabel(r"rank bound per $X_i$")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks", default="2,3,4")
    parser.add_argument("--points", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=800)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=Path("nonsymmetric_n5_min_rank.png"))
    parser.add_argument("--csv", type=Path, default=Path("nonsymmetric_n5_min_rank.csv"))
    args = parser.parse_args()

    ks = sorted({int(value.strip()) for value in args.ks.split(",") if value.strip()})
    if any(k < 1 or k > 5 for k in ks):
        parser.error("every k must lie in 1..5")
    if args.points < 2:
        parser.error("--points must be at least 2")

    rows: list[dict[str, object]] = []
    for k in ks:
        rows.extend(
            run_case(
                k,
                points=args.points,
                epsilon=args.epsilon,
                restarts=args.restarts,
                max_nfev=args.max_nfev,
                feasibility_tol=args.feasibility_tol,
            )
        )
    if not rows:
        parser.error("no ranks below the threshold benchmark for the selected k values")
    write_csv(rows, args.csv)
    plot_min_rank(rows, args.output)
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
