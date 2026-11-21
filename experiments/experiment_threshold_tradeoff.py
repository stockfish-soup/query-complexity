"""Plot the trace/rank tradeoff for trace-minimized THRESHOLD duals."""

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


def parse_ks(text: str, n: int) -> list[int]:
    if text.strip().lower() == "default":
        return sorted({1, 2, (n + 1) // 2})
    ks = sorted({int(value.strip()) for value in text.split(",") if value.strip()})
    if not ks or any(k < 1 or k > n for k in ks):
        raise ValueError("all threshold values k must lie in 1..n")
    return ks


def run_experiment(
    n: int,
    ks: list[int],
    points: int,
    *,
    solver: str | None,
    epsilon: float,
    rank_relative_tol: float,
    rank_absolute_tol: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in ks:
        usual_adversary = qc.threshold_adversary_value(n, k)
        phase_adversary = usual_adversary / 2.0
        alpha_max = n / usual_adversary
        alphas = np.linspace(1.0, alpha_max, points)
        for alpha in alphas:
            orbit = qc.solve_symmetric_trace_minimized_dual(
                n,
                qc.THRESHOLD_layers(n, k),
                alpha=float(alpha),
                adversary_value=phase_adversary,
                epsilon=epsilon,
                solver=solver,
            )
            rank_data = qc.symmetric_witness_rank(
                orbit,
                relative_tol=rank_relative_tol,
                absolute_tol=rank_absolute_tol,
            )
            realized_phase_cost = max(orbit.input_costs_by_layer.values())
            row = {
                "n": n,
                "k": k,
                "alpha": float(alpha),
                "usual_adversary": usual_adversary,
                "usual_objective_cap": 2.0 * orbit.cap,
                "realized_usual_objective": 2.0 * realized_phase_cost,
                "trace": orbit.trace_value,
                "rank_X_i": rank_data["rank_X_i"],
                "total_witness_dimension": rank_data["total_witness_dimension"],
                "block_ranks": ";".join(map(str, rank_data["block_ranks"])),
                "status": orbit.status,
            }
            rows.append(row)
            print(
                f"THRESHOLD({n},{k}) alpha={alpha:.6g}: "
                f"rank(X_i)={row['rank_X_i']}, trace={orbit.trace_value:.8g}, "
                f"realized objective={row['realized_usual_objective']:.8g}"
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, object]], path: Path) -> None:
    fig, (rank_ax, trace_ax) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ks = sorted({int(row["k"]) for row in rows})
    for k in ks:
        selected = [row for row in rows if int(row["k"]) == k]
        alphas = [float(row["alpha"]) for row in selected]
        ranks = [int(row["rank_X_i"]) for row in selected]
        traces = [float(row["trace"]) for row in selected]
        label = rf"$k={k}$"
        rank_ax.plot(alphas, ranks, "o-", linewidth=1.8, markersize=4, label=label)
        trace_ax.plot(alphas, traces, "o-", linewidth=1.8, markersize=4, label=label)

    rank_ax.set_xlabel(r"query relaxation $\alpha$")
    rank_ax.set_ylabel(r"numerical rank of $X_i$")
    rank_ax.set_title("Witness rank")
    rank_ax.set_yscale("log", base=2)
    rank_ticks = sorted({int(row["rank_X_i"]) for row in rows})
    rank_ax.set_yticks(rank_ticks, labels=[str(rank) for rank in rank_ticks])
    rank_ax.grid(True, which="both", alpha=0.28)
    rank_ax.legend()

    trace_ax.set_xlabel(r"query relaxation $\alpha$")
    trace_ax.set_ylabel(r"minimum $\frac{1}{2}\sum_i\mathrm{Tr}(X_i)$")
    trace_ax.set_title("Trace proxy for witness size")
    trace_ax.grid(True, alpha=0.28)
    trace_ax.legend()

    n = int(rows[0]["n"])
    fig.suptitle(
        rf"THRESHOLD$(n,k)$ trace-minimized duals, $n={n}$; "
        r"$\mathrm{OBJ}\leq\alpha\,\mathrm{Adv}^{\pm}$"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--ks",
        default="default",
        help="comma-separated thresholds, or 'default' for 1,2,ceil(n/2)",
    )
    parser.add_argument("--points", type=int, default=9)
    parser.add_argument("--solver", default=None)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--rank-relative-tol", type=float, default=1e-7)
    parser.add_argument("--rank-absolute-tol", type=float, default=1e-7)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("threshold_tradeoff.png"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("threshold_tradeoff.csv"),
    )
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.points < 2:
        parser.error("--points must be at least 2")

    try:
        ks = parse_ks(args.ks, args.n)
    except ValueError as exc:
        parser.error(str(exc))
    rows = run_experiment(
        args.n,
        ks,
        args.points,
        solver=args.solver,
        epsilon=args.epsilon,
        rank_relative_tol=args.rank_relative_tol,
        rank_absolute_tol=args.rank_absolute_tol,
    )
    write_csv(rows, args.csv)
    plot_rows(rows, args.output)
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
