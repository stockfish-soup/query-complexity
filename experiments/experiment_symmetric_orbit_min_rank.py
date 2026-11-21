"""Minimum-rank search in the orbit-reduced symmetric dual.

For each symmetric Boolean instance and alpha, enumerate Terwilliger block-rank
patterns and search for a feasible symmetric fixed-rank solution.  The reported
rank is the symmetric total rank

    sum_r multiplicity(r) * rank(B_r).

This is intended for small n first; n=5 is quite manageable.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "query-complexity-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import query_complexity as qc


FIELDS = [
    "kind",
    "n",
    "k",
    "alpha",
    "phase_adversary",
    "multiplicities",
    "convex_block_ranks",
    "pattern",
    "rank_bound",
    "actual_block_ranks",
    "actual_rank",
    "feasible",
    "trace",
    "max_equality_residual",
    "max_block_residual",
    "max_cap_violation",
    "status",
    "attempts",
    "total_nfev",
]


def parse_float_list(spec: str) -> list[float]:
    return sorted({float(part.strip()) for part in spec.split(",") if part.strip()})


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


def all_patterns(max_block_ranks: list[int], multiplicities: list[int], upper_rank: int) -> list[tuple[int, ...]]:
    choices = [range(rank + 1) for rank in max_block_ranks]
    patterns = [
        tuple(pattern)
        for pattern in itertools.product(*choices)
        if any(pattern) and weighted_rank(tuple(pattern), multiplicities) <= upper_rank
    ]
    return sorted(patterns, key=lambda pattern: (weighted_rank(pattern, multiplicities), pattern))


def targeted_patterns(
    convex_block_ranks: list[int],
    max_block_ranks: list[int],
    multiplicities: list[int],
    upper_rank: int,
) -> list[tuple[int, ...]]:
    active = [idx for idx, rank in enumerate(convex_block_ranks) if rank > 0]
    patterns: list[tuple[int, ...]] = []
    full = tuple(min(rank, max_block_ranks[idx]) for idx, rank in enumerate(convex_block_ranks))
    if any(full):
        patterns.append(full)
    if active:
        patterns.append(tuple(1 if idx in active else 0 for idx in range(len(max_block_ranks))))
    for idx in active:
        lowered = list(full)
        lowered[idx] = max(0, lowered[idx] - 1)
        if any(lowered):
            patterns.append(tuple(lowered))
        zeroed = list(full)
        zeroed[idx] = 0
        if any(zeroed):
            patterns.append(tuple(zeroed))
    for idx in active:
        singleton = [0] * len(max_block_ranks)
        singleton[idx] = 1
        patterns.append(tuple(singleton))
    unique = list(dict.fromkeys(patterns))
    unique = [
        pattern
        for pattern in unique
        if weighted_rank(pattern, multiplicities) <= upper_rank
    ]
    return sorted(unique, key=lambda pattern: (weighted_rank(pattern, multiplicities), pattern))


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def run_case(
    kind: str,
    n: int,
    k: int,
    *,
    alphas: list[float],
    max_block_rank: int,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    solver: str,
    pattern_mode: str,
    csv_path: Path,
) -> None:
    layers = layers_for(kind, n, k)
    base = qc.solve_symmetric_trace_minimized_dual(
        n,
        layers,
        alpha=1.0,
        epsilon=epsilon,
        solver=solver,
    )
    phase_adversary = base.objective_value
    num_blocks = len(base.block_values)
    multiplicities = multiplicities_for(n, num_blocks)
    upper_rank = sum(max_block_rank * mult for mult in multiplicities)

    for alpha in alphas:
        convex = qc.solve_symmetric_trace_minimized_dual(
            n,
            layers,
            alpha=alpha,
            adversary_value=phase_adversary,
            epsilon=epsilon,
            solver=solver,
        )
        convex_block_ranks = [
            qc.numerical_psd_rank(block, relative_tol=1e-7, absolute_tol=1e-7)
            for block in convex.block_values
        ]
        max_ranks = [
            min(max_block_rank, len(block))
            for block in convex.block_values
        ]
        if pattern_mode == "all":
            patterns = all_patterns(max_ranks, multiplicities, upper_rank)
        elif pattern_mode == "targeted":
            patterns = targeted_patterns(convex_block_ranks, max_ranks, multiplicities, upper_rank)
        else:
            raise ValueError("pattern_mode must be 'all' or 'targeted'")
        best_rank: int | None = None
        best_result = None
        print(
            f"{kind.upper()}({n},{k}) alpha={alpha:.8g}: "
            f"testing {len(patterns)} patterns up to rank {upper_rank}",
            flush=True,
        )
        for pattern in patterns:
            rank_bound = weighted_rank(pattern, multiplicities)
            if best_rank is not None and rank_bound >= best_rank:
                break
            print(f"  pattern={''.join(map(str, pattern))} rank={rank_bound}: starting", flush=True)
            result = qc.solve_symmetric_fixed_rank_dual_feasibility(
                n,
                layers,
                block_rank=max_block_rank,
                block_ranks=pattern,
                alpha=alpha,
                adversary_value=phase_adversary,
                epsilon=epsilon,
                initial_solution=convex,
                restarts=restarts,
                random_seed=130_000 + 1_000 * k + 10 * rank_bound + round(1000 * alpha),
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
                    "alpha": alpha,
                    "phase_adversary": phase_adversary,
                    "multiplicities": ";".join(map(str, multiplicities)),
                    "convex_block_ranks": ";".join(map(str, convex_block_ranks)),
                    "pattern": ";".join(map(str, pattern)),
                    "rank_bound": rank_bound,
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
                },
            )
            print(
                f"  pattern={''.join(map(str, pattern))} rank={rank_bound}: "
                f"{result.status}; eq={result.max_equality_residual:.2e}, "
                f"block={result.max_block_residual:.2e}, cap={result.max_cap_violation:.2e}",
                flush=True,
            )
            if result.feasible:
                best_rank = actual_rank
                best_result = result
                upper_rank = min(upper_rank, rank_bound)
                break
        if best_result is None:
            print(f"  no feasible pattern found up to rank {upper_rank}", flush=True)
        else:
            print(f"  minimum symmetric rank found: {best_rank}", flush=True)


def plot_csv(csv_path: Path, output: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cases = sorted({(row["kind"], int(row["n"]), int(row["k"])) for row in rows})
    for color, (kind, n, k) in zip(colors, cases):
        case = [row for row in rows if row["kind"] == kind and int(row["n"]) == n and int(row["k"]) == k]
        xs = sorted({float(row["alpha"]) for row in case})
        ys = []
        labels = []
        for alpha in xs:
            feasible = [
                row for row in case
                if float(row["alpha"]) == alpha and row["feasible"] == "True"
            ]
            if feasible:
                best = min(feasible, key=lambda row: int(row["actual_rank"]))
                ys.append(int(best["actual_rank"]))
                labels.append(best["pattern"])
            else:
                ys.append(np.nan)
                labels.append("")
        ax.step(xs, ys, where="post", marker="o", color=color, label=f"{kind.upper()}({n},{k})")
        for x, y, label in zip(xs, ys, labels):
            if not np.isnan(y):
                ax.text(x, y, label, fontsize=7, color=color, ha="center", va="bottom")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"minimum symmetric rank $\sum_r m_r\,rank(B_r)$")
    ax.set_title("Orbit-reduced symmetric BM rank search")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("exact", "threshold"), default="exact")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--ks", default="1,2")
    parser.add_argument("--alphas", default="1,1.005,1.01,1.02,1.05,1.1,1.25,1.5,2,3,5")
    parser.add_argument("--max-block-rank", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--restarts", type=int, default=0)
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--pattern-mode", choices=("targeted", "all"), default="targeted")
    parser.add_argument("--csv", type=Path, default=Path("symmetric_orbit_min_rank.csv"))
    parser.add_argument("--output", type=Path, default=Path("symmetric_orbit_min_rank.png"))
    args = parser.parse_args()

    if args.csv.exists():
        args.csv.unlink()
    alphas = parse_float_list(args.alphas)
    for k in sorted({int(value.strip()) for value in args.ks.split(",") if value.strip()}):
        run_case(
            args.kind,
            args.n,
            k,
            alphas=alphas,
            max_block_rank=args.max_block_rank,
            epsilon=args.epsilon,
            restarts=args.restarts,
            max_nfev=args.max_nfev,
            feasibility_tol=args.feasibility_tol,
            solver=args.solver,
            pattern_mode=args.pattern_mode,
            csv_path=args.csv,
        )
    plot_csv(args.csv, args.output)


if __name__ == "__main__":
    main()
