"""Search which symmetric THRESHOLD Terwilliger blocks can be zero."""

from __future__ import annotations

import argparse
import csv
import itertools
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
    return [float(value) for value in np.linspace(1.0, alpha_max, points)]


def block_multiplicities(n: int) -> list[int]:
    N = n - 1
    return [qc.binom_safe(N, r) - qc.binom_safe(N, r - 1) for r in range(N // 2 + 1)]


def rank_patterns(num_blocks: int, max_active_r: int) -> list[tuple[int, ...]]:
    variable_count = min(num_blocks, max_active_r + 1)
    fixed_zero_count = num_blocks - variable_count
    patterns = [
        tuple(prefix) + (0,) * fixed_zero_count
        for prefix in itertools.product((0, 1), repeat=variable_count)
    ]
    return [pattern for pattern in patterns if any(pattern)]


def weighted_rank(pattern: tuple[int, ...], multiplicities: list[int]) -> int:
    return int(sum(rank * mult for rank, mult in zip(pattern, multiplicities)))


def run_case(
    n: int,
    k: int,
    *,
    points: int,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    solver: str,
) -> list[dict[str, object]]:
    layers = qc.THRESHOLD_layers(n, k)
    phase_adversary = qc.threshold_adversary_value(n, k) / 2.0
    multiplicities = block_multiplicities(n)
    patterns = sorted(
        rank_patterns(len(multiplicities), k - 1),
        key=lambda pattern: (weighted_rank(pattern, multiplicities), pattern),
    )
    rows: list[dict[str, object]] = []

    for alpha in alpha_grid(n, k, points):
        convex = qc.solve_symmetric_trace_minimized_dual(
            n,
            layers,
            alpha=alpha,
            adversary_value=phase_adversary,
            epsilon=epsilon,
            solver=solver,
        )
        for pattern in patterns:
            result = qc.solve_symmetric_fixed_rank_dual_feasibility(
                n,
                layers,
                block_rank=1,
                block_ranks=pattern,
                alpha=alpha,
                adversary_value=phase_adversary,
                epsilon=epsilon,
                initial_solution=convex,
                restarts=restarts,
                random_seed=70_000 + 1_000 * n + 100 * k + 10 * weighted_rank(pattern, multiplicities) + round(1000 * alpha),
                max_nfev=max_nfev,
                feasibility_tol=feasibility_tol,
            )
            actual_block_ranks = [
                qc.numerical_psd_rank(block, relative_tol=1e-7, absolute_tol=1e-7)
                for block in result.block_values
            ]
            actual_rank = weighted_rank(tuple(actual_block_ranks), multiplicities)
            rows.append(
                {
                    "n": n,
                    "k": k,
                    "alpha": alpha,
                    "pattern": "".join(map(str, pattern)),
                    "multiplicities": ";".join(map(str, multiplicities)),
                    "rank_bound": weighted_rank(pattern, multiplicities),
                    "actual_block_ranks": ";".join(map(str, actual_block_ranks)),
                    "actual_rank": actual_rank if result.feasible else "",
                    "feasible": result.feasible,
                    "trace": result.trace_value,
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
                f"pattern={''.join(map(str, pattern))}, "
                f"rank={weighted_rank(pattern, multiplicities)}: {result.status}; "
                f"eq={result.max_equality_residual:.2e}, "
                f"block={result.max_block_residual:.2e}, cap={result.max_cap_violation:.2e}"
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

    cases = sorted({(int(row["n"]), int(row["k"])) for row in rows})
    for color, (n, k) in zip(colors, cases):
        case = [row for row in rows if int(row["n"]) == n and int(row["k"]) == k]
        alphas = sorted({float(row["alpha"]) for row in case})
        minimums = []
        labels = []
        for alpha in alphas:
            feasible = [
                row for row in case
                if float(row["alpha"]) == alpha and str(row["feasible"]) == "True"
            ]
            if feasible:
                best = min(feasible, key=lambda row: int(row["rank_bound"]))
                minimums.append(int(best["rank_bound"]))
                labels.append(str(best["pattern"]))
            else:
                minimums.append(np.nan)
                labels.append("")

        ax.plot(alphas, minimums, "o-", color=color, label=rf"$n={n}, k={k}$")
        for alpha, value, label in zip(alphas, minimums, labels):
            if not np.isnan(value):
                ax.text(alpha, value + 0.08, label, ha="center", va="bottom", fontsize=7, color=color)

    ax.set_title("Symmetric zero-block search")
    ax.set_xlabel(r"query relaxation $\alpha$")
    ax.set_ylabel(r"minimal $\sum_r m_r\,\mathrm{rank}(B_r)$")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_cases(value: str) -> list[tuple[int, int]]:
    cases = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        n_str, k_str = item.split(":")
        cases.append((int(n_str), int(k_str)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="5:2,6:2,6:3,7:2,7:3")
    parser.add_argument("--points", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=400)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--output", type=Path, default=Path("symmetric_zero_block_rank.png"))
    parser.add_argument("--csv", type=Path, default=Path("symmetric_zero_block_rank.csv"))
    args = parser.parse_args()

    cases = parse_cases(args.cases)
    if not cases:
        parser.error("--cases must contain at least one n:k pair")
    if args.points < 2:
        parser.error("--points must be at least 2")

    rows: list[dict[str, object]] = []
    for n, k in cases:
        if n <= 0 or k < 1 or k > n:
            parser.error("each case must satisfy n > 0 and 1 <= k <= n")
        if 2 * k > n:
            parser.error("use only k <= n/2; larger k is complement-symmetric")
        rows.extend(
            run_case(
                n,
                k,
                points=args.points,
                epsilon=args.epsilon,
                restarts=args.restarts,
                max_nfev=args.max_nfev,
                feasibility_tol=args.feasibility_tol,
                solver=args.solver,
            )
        )
    write_csv(rows, args.csv)
    plot_min_rank(rows, args.output)
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
