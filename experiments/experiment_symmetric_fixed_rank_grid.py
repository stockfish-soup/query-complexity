"""Grid plot for symmetric fixed-rank nonconvex threshold searches."""

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
    for value in (1.01, 1.02, 1.03, 1.04, 1.06, 1.10, 1.20):
        if value <= alpha_max:
            values.add(value)
    return sorted(float(value) for value in values)


def run_case(
    n: int,
    k: int,
    *,
    points: int,
    max_block_rank: int,
    solver: str,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
) -> list[dict[str, object]]:
    phase_adversary = qc.threshold_adversary_value(n, k) / 2.0
    layers = qc.THRESHOLD_layers(n, k)
    rows: list[dict[str, object]] = []
    warm_starts: dict[int, qc.SymmetricFixedRankFeasibilityResult] = {}

    for alpha in alpha_grid(n, k, points):
        convex = qc.solve_symmetric_trace_minimized_dual(
            n,
            layers,
            alpha=alpha,
            adversary_value=phase_adversary,
            epsilon=epsilon,
            solver=solver,
        )
        convex_rank_info = qc.symmetric_witness_rank(
            convex,
            relative_tol=1e-7,
            absolute_tol=1e-7,
        )
        rows.append(
            {
                "n": n,
                "k": k,
                "alpha": alpha,
                "method": "convex_symmetric",
                "block_rank_bound": "",
                "full_rank_bound": "",
                "rank_X_i": convex_rank_info["rank_X_i"],
                "block_ranks": ";".join(map(str, convex_rank_info["block_ranks"])),
                "trace": convex.trace_value,
                "feasible": True,
                "max_equality_residual": "",
                "max_block_residual": "",
                "max_cap_violation": "",
                "status": convex.status,
                "attempts": "",
                "total_nfev": "",
            }
        )

        for block_rank in range(1, max_block_rank + 1):
            initial = warm_starts.get(block_rank, convex)
            result = qc.solve_symmetric_fixed_rank_dual_feasibility(
                n,
                layers,
                block_rank=block_rank,
                alpha=alpha,
                adversary_value=phase_adversary,
                epsilon=epsilon,
                initial_solution=initial,
                restarts=restarts,
                random_seed=10_000 * n + 1_000 * k + 100 * block_rank + round(1000 * alpha),
                max_nfev=max_nfev,
                feasibility_tol=feasibility_tol,
            )
            if result.feasible:
                warm_starts[block_rank] = result
            actual_block_ranks = [
                qc.numerical_psd_rank(block, relative_tol=1e-7, absolute_tol=1e-7)
                for block in result.block_values
            ]
            actual_full_rank = int(
                sum(rank * mult for rank, mult in zip(actual_block_ranks, result.block_multiplicities))
            )
            rows.append(
                {
                    "n": n,
                    "k": k,
                    "alpha": alpha,
                    "method": "fixed_rank_symmetric",
                    "block_rank_bound": block_rank,
                    "full_rank_bound": result.full_rank_bound,
                    "rank_X_i": actual_full_rank if result.feasible else "",
                    "block_ranks": ";".join(map(str, actual_block_ranks)),
                    "trace": result.trace_value,
                    "feasible": result.feasible,
                    "max_equality_residual": result.max_equality_residual,
                    "max_block_residual": result.max_block_residual,
                    "max_cap_violation": result.max_cap_violation,
                    "status": result.status,
                    "attempts": result.attempts,
                    "total_nfev": result.total_nfev,
                }
            )
            print(
                f"sym THRESHOLD({n},{k}) alpha={alpha:.6g}, "
                f"block-rank<={block_rank} full-rank<={result.full_rank_bound}: "
                f"{result.status}; eq={result.max_equality_residual:.2e}, "
                f"block={result.max_block_residual:.2e}, cap={result.max_cap_violation:.2e}"
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_grid(rows: list[dict[str, object]], path: Path) -> None:
    fixed = [row for row in rows if row["method"] == "fixed_rank_symmetric"]
    cases = sorted({(int(row["n"]), int(row["k"])) for row in fixed})
    cols = min(3, len(cases))
    rows_count = int(np.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows_count, cols, figsize=(4.8 * cols, 3.9 * rows_count), squeeze=False)

    for ax in axes.ravel()[len(cases):]:
        ax.axis("off")

    for ax, (n, k) in zip(axes.ravel(), cases):
        case_rows = [row for row in fixed if int(row["n"]) == n and int(row["k"]) == k]
        alphas = sorted({float(row["alpha"]) for row in case_rows})
        ranks = sorted({int(row["full_rank_bound"]) for row in case_rows})
        grid = np.zeros((len(ranks), len(alphas)), dtype=float)
        labels: dict[tuple[int, int], str] = {}
        for row in case_rows:
            r = ranks.index(int(row["full_rank_bound"]))
            c = alphas.index(float(row["alpha"]))
            feasible = str(row["feasible"]) == "True"
            grid[r, c] = 1.0 if feasible else 0.0
            block_rank = int(row["block_rank_bound"])
            status = "ok" if feasible else "no"
            labels[(r, c)] = f"a={float(row['alpha']):.3g}\nbr={block_rank}\n{status}"

        ax.imshow(
            grid,
            aspect="auto",
            interpolation="nearest",
            cmap=matplotlib.colors.ListedColormap(["#e57373", "#66bb6a"]),
            vmin=0,
            vmax=1,
        )
        ax.set_title(rf"THRESHOLD({n},{k})")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"full rank bound for $X_i$")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{alpha:.3g}" for alpha in alphas], rotation=45, ha="right")
        ax.set_yticks(range(len(ranks)))
        ax.set_yticklabels([str(rank) for rank in ranks])
        for (r, c), text in labels.items():
            ax.text(c, r, text, ha="center", va="center", fontsize=7, color="black")

    fig.suptitle("Symmetric nonconvex fixed-rank feasibility grid")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_cases(value: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        n_str, k_str = item.split(":")
        cases.append((int(n_str), int(k_str)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="4:1,4:2,5:1,5:2,5:3,6:1,6:2,6:3")
    parser.add_argument("--points", type=int, default=5)
    parser.add_argument("--max-block-rank", type=int, default=3)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=600)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, default=Path("symmetric_fixed_rank_grid.png"))
    parser.add_argument("--csv", type=Path, default=Path("symmetric_fixed_rank_grid.csv"))
    args = parser.parse_args()

    cases = parse_cases(args.cases)
    if not cases:
        parser.error("--cases must contain at least one n:k pair")
    for n, k in cases:
        if n <= 0 or k < 1 or k > n:
            parser.error("each case must satisfy n > 0 and 1 <= k <= n")
    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.max_block_rank < 1:
        parser.error("--max-block-rank must be positive")

    rows: list[dict[str, object]] = []
    for n, k in cases:
        rows.extend(
            run_case(
                n,
                k,
                points=args.points,
                max_block_rank=args.max_block_rank,
                solver=args.solver,
                epsilon=args.epsilon,
                restarts=args.restarts,
                max_nfev=args.max_nfev,
                feasibility_tol=args.feasibility_tol,
            )
        )
    write_csv(rows, args.csv)
    plot_grid(rows, args.output)
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
