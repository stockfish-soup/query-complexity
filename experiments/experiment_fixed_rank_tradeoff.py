"""Compare symmetric/full convex duals with nonsymmetric fixed-rank searches."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
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
    for value in (1.02, 1.04, 1.06, 1.10):
        if value <= alpha_max:
            values.add(value)
    return sorted(float(value) for value in values)


def fixed_rank_row(
    n: int,
    k: int,
    alpha: float,
    result: qc.FixedRankFeasibilityResult,
) -> dict[str, object]:
    return {
        "n": n,
        "k": k,
        "alpha": alpha,
        "method": "fixed_rank_nonsymmetric",
        "rank_bound": result.rank_bound,
        "rank_X_i": result.rank_bound if result.feasible else "",
        "coordinate_ranks": ";".join(
            str(qc.numerical_psd_rank(gram)) for gram in result.grams
        ),
        "trace": result.trace_value,
        "feasible": result.feasible,
        "max_equality_residual": result.max_equality_residual,
        "max_cap_violation": result.max_cap_violation,
        "status": result.status,
        "attempts": result.attempts,
        "total_nfev": result.total_nfev,
    }


def run_case(
    n: int,
    k: int,
    points: int,
    *,
    solver: str,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
) -> list[dict[str, object]]:
    inputs = qc.all_binary_inputs(n)
    outputs = [int(sum(x) >= k) for x in inputs]
    phase_adversary = qc.threshold_adversary_value(n, k) / 2.0
    rows: list[dict[str, object]] = []
    warm_starts: dict[int, list[np.ndarray]] = {}

    for alpha in alpha_grid(n, k, points):
        symmetric = qc.solve_symmetric_trace_minimized_dual(
            n,
            qc.THRESHOLD_layers(n, k),
            alpha=alpha,
            adversary_value=phase_adversary,
            epsilon=epsilon,
            solver=solver,
        )
        symmetric_rank = qc.symmetric_witness_rank(
            symmetric,
            relative_tol=1e-7,
            absolute_tol=1e-7,
        )["rank_X_i"]
        rows.append(
            {
                "n": n,
                "k": k,
                "alpha": alpha,
                "method": "convex_symmetric",
                "rank_bound": "",
                "rank_X_i": symmetric_rank,
                "coordinate_ranks": str(symmetric_rank),
                "trace": symmetric.trace_value,
                "feasible": True,
                "max_equality_residual": "",
                "max_cap_violation": "",
                "status": symmetric.status,
                "attempts": "",
                "total_nfev": "",
            }
        )

        full = qc.solve_trace_minimized_dual(
            inputs,
            outputs,
            alpha=alpha,
            adversary_value=phase_adversary,
            epsilon=epsilon,
            solver=solver,
            psd_tol=1e-7,
        )
        full_rank = max(full.witness_dims)
        rows.append(
            {
                "n": n,
                "k": k,
                "alpha": alpha,
                "method": "convex_nonsymmetric",
                "rank_bound": "",
                "rank_X_i": full_rank,
                "coordinate_ranks": ";".join(map(str, full.witness_dims)),
                "trace": full.trace_value,
                "feasible": True,
                "max_equality_residual": "",
                "max_cap_violation": "",
                "status": full.status,
                "attempts": "",
                "total_nfev": "",
            }
        )

        # Rank 1 is already optimal for OR. For the nontrivial case, search
        # every rank up through the rank selected by convex trace minimization.
        for rank in range(1, full_rank + 1):
            initial_factors = warm_starts.get(rank)
            initial_solution = None if initial_factors is not None else full
            # Failed rank-1 searches are comparatively expensive; one random
            # restart plus the projected convex start is enough for the sweep.
            rank_restarts = 1 if rank == 1 and full_rank > 1 else restarts
            result = qc.solve_fixed_rank_dual_feasibility(
                inputs,
                outputs,
                rank=rank,
                alpha=alpha,
                adversary_value=phase_adversary,
                epsilon=epsilon,
                initial_solution=initial_solution,
                initial_factors=initial_factors,
                restarts=rank_restarts,
                random_seed=1000 * k + 100 * rank + round(1000 * alpha),
                max_nfev=max_nfev,
                feasibility_tol=feasibility_tol,
            )
            if result.feasible:
                warm_starts[rank] = result.z
            rows.append(fixed_rank_row(n, k, alpha, result))
            print(
                f"THRESHOLD({n},{k}) alpha={alpha:.6g}, rank<={rank}: "
                f"{result.status}; equality={result.max_equality_residual:.3e}, "
                f"cap={result.max_cap_violation:.3e}"
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, object]], path: Path, feasibility_tol: float) -> None:
    fig, (rank_ax, residual_ax) = plt.subplots(1, 2, figsize=(12, 4.7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ks = sorted({int(row["k"]) for row in rows})
    for color, k in zip(colors, ks):
        case = [row for row in rows if int(row["k"]) == k]
        symmetric = [row for row in case if row["method"] == "convex_symmetric"]
        full = [row for row in case if row["method"] == "convex_nonsymmetric"]
        fixed = [row for row in case if row["method"] == "fixed_rank_nonsymmetric"]
        alphas = sorted({float(row["alpha"]) for row in case})
        minimum_fixed = []
        for alpha in alphas:
            feasible_ranks = [
                int(row["rank_bound"])
                for row in fixed
                if float(row["alpha"]) == alpha and str(row["feasible"]) == "True"
            ]
            minimum_fixed.append(min(feasible_ranks) if feasible_ranks else np.nan)

        rank_ax.plot(
            [float(row["alpha"]) for row in symmetric],
            [int(row["rank_X_i"]) for row in symmetric],
            "--",
            color=color,
            label=rf"symmetric convex, $k={k}$",
        )
        rank_ax.plot(
            [float(row["alpha"]) for row in full],
            [int(row["rank_X_i"]) for row in full],
            ":",
            linewidth=2.2,
            color=color,
            label=rf"full convex, $k={k}$",
        )
        rank_ax.plot(
            alphas,
            minimum_fixed,
            "o-",
            color=color,
            label=rf"fixed-rank feasible, $k={k}$",
        )

        rank_bounds = sorted({int(row["rank_bound"]) for row in fixed})
        convex_rank = max(int(row["rank_X_i"]) for row in full)
        for rank in rank_bounds:
            if rank >= convex_rank:
                continue
            selected = [row for row in fixed if int(row["rank_bound"]) == rank]
            scores = [
                max(
                    float(row["max_equality_residual"]),
                    float(row["max_cap_violation"]),
                    1e-16,
                )
                for row in selected
            ]
            residual_ax.plot(
                [float(row["alpha"]) for row in selected],
                scores,
                "o-",
                color=color,
                alpha=0.45 + 0.25 * rank,
                label=rf"$k={k}$, rank $\leq {rank}$",
            )

    rank_ax.set_xlabel(r"query relaxation $\alpha$")
    rank_ax.set_ylabel(r"rank bound per $X_i$")
    rank_ax.set_title("Convex rank vs. fixed-rank feasibility")
    rank_ax.grid(True, alpha=0.28)
    rank_ax.legend(fontsize=8)

    residual_ax.axhline(
        feasibility_tol,
        color="black",
        linestyle="--",
        linewidth=1,
        label="feasibility tolerance",
    )
    residual_ax.set_yscale("log")
    residual_ax.set_xlabel(r"query relaxation $\alpha$")
    residual_ax.set_ylabel("maximum constraint violation")
    residual_ax.set_title("Nonconvex search diagnostics")
    residual_ax.grid(True, which="both", alpha=0.28)
    residual_ax.legend(fontsize=8)

    n = int(rows[0]["n"])
    fig.suptitle(
        rf"THRESHOLD fixed-rank dual search, $n={n}$; "
        r"$X_i=Z_iZ_i^\mathsf{T}$ without permutation symmetry"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--ks", default="1,2")
    parser.add_argument("--points", type=int, default=7)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixed_rank_tradeoff.png"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("fixed_rank_tradeoff.csv"),
    )
    args = parser.parse_args()
    ks = sorted({int(value.strip()) for value in args.ks.split(",") if value.strip()})
    if args.n <= 0 or any(k < 1 or k > args.n for k in ks):
        parser.error("require n > 0 and every k in 1..n")
    if args.points < 2:
        parser.error("--points must be at least 2")

    rows: list[dict[str, object]] = []
    for k in ks:
        rows.extend(
            run_case(
                args.n,
                k,
                args.points,
                solver=args.solver,
                epsilon=args.epsilon,
                restarts=args.restarts,
                max_nfev=args.max_nfev,
                feasibility_tol=args.feasibility_tol,
            )
        )
    write_csv(rows, args.csv)
    plot_rows(rows, args.output, args.feasibility_tol)
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
