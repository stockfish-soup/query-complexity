"""Burer-Monteiro fixed-rank search for THRESHOLD(n,k)."""

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


def alpha_grid(
    n: int,
    k: int,
    points: int,
    adversary_value: float,
    alphas: str | None = None,
) -> list[float]:
    if alphas:
        return sorted(
            {
                float(value.strip())
                for value in alphas.split(",")
                if value.strip()
            }
        )
    alpha_max = n / (2.0 * adversary_value)
    near_one = [1.0, 1.01, 1.02, 1.03, 1.04, 1.06, 1.08, 1.10, 1.15, 1.20]
    values = {value for value in near_one if value <= alpha_max}
    values.update(float(value) for value in np.linspace(1.0, alpha_max, points))
    return sorted(values)


def parse_rank_spec(spec: str, k: int, benchmark: int) -> list[int]:
    if spec == "beat_down":
        return list(range(benchmark - 1, 0, -1))
    if spec == "benchmark_down":
        return list(range(benchmark, 0, -1))
    if spec == "below":
        return list(range(benchmark - 1, 0, -1))
    ranges: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ranges.extend(range(int(lo), int(hi) + 1))
        else:
            ranges.append(int(part))
    return sorted({rank for rank in ranges if 1 <= rank <= benchmark}, reverse=True)


def run_case(
    kind: str,
    n: int,
    k: int,
    ranks: list[int],
    *,
    points: int,
    alphas: str | None,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    stop_after_first_infeasible: bool,
) -> list[dict[str, object]]:
    if kind == "threshold":
        layers = qc.THRESHOLD_layers(n, k)
        benchmark = comb(n - 1, k - 1)
    elif kind == "exact":
        layers = qc.EXACT_layers(n, k)
        benchmark = comb(n, k)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    inputs = qc.all_binary_inputs(n)
    outputs = [layers[sum(x)] for x in inputs]
    phase_adversary = qc.solve_symmetric_trace_minimized_dual(
        n,
        layers,
        alpha=1.0,
        epsilon=epsilon,
    ).objective_value
    warm_starts: dict[int, list[np.ndarray]] = {}
    rows: list[dict[str, object]] = []

    def test_rank(
        alpha: float,
        rank: int,
        symmetric_start: qc.AdversarySolution,
        previous_feasible: list[np.ndarray] | None,
    ) -> tuple[qc.FixedRankFeasibilityResult, list[np.ndarray] | None]:
        initial_factors = warm_starts.get(rank)
        if initial_factors is None:
            initial_factors = previous_feasible
        print(
            f"{kind.upper()}({n},{k}) alpha={alpha:.6g}, rank<={rank}: starting",
            flush=True,
        )
        result = qc.solve_fixed_rank_dual_feasibility(
            inputs,
            outputs,
            rank=rank,
            alpha=alpha,
            adversary_value=phase_adversary,
            epsilon=epsilon,
            initial_solution=symmetric_start if initial_factors is None else None,
            initial_factors=initial_factors,
            restarts=restarts,
            random_seed=60_000 + 1_000 * k + 100 * rank + round(1000 * alpha),
            max_nfev=max_nfev,
            feasibility_tol=feasibility_tol,
        )
        next_previous_feasible = previous_feasible
        if result.feasible:
            warm_starts[rank] = result.z
            next_previous_feasible = result.z
        rows.append(
            {
                "kind": kind,
                "n": n,
                "k": k,
                "alpha": alpha,
                "benchmark_rank": benchmark,
                "threshold_rank": benchmark,
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
            f"{kind.upper()}({n},{k}) alpha={alpha:.6g}, rank<={rank} "
            f"(beat {benchmark}): {result.status}; "
            f"eq={result.max_equality_residual:.2e}, cap={result.max_cap_violation:.2e}"
        )
        return result, next_previous_feasible

    initial_upper = max(rank for rank in ranks if rank < benchmark)
    upper_rank = initial_upper
    for alpha in alpha_grid(n, k, points, phase_adversary, alphas):
        orbit = qc.solve_symmetric_trace_minimized_dual(
            n,
            layers,
            alpha=alpha,
            adversary_value=phase_adversary,
            epsilon=epsilon,
        )
        symmetric_start = qc.expand_orbit_solution(orbit, psd_tol=1e-7)
        previous_feasible: list[np.ndarray] | None = None
        lo = 1
        hi = upper_rank
        best_feasible: int | None = None
        while lo <= hi:
            rank = (lo + hi) // 2
            result, previous_feasible = test_rank(
                alpha,
                rank,
                symmetric_start,
                previous_feasible,
            )
            if result.feasible:
                best_feasible = rank
                hi = rank - 1
            else:
                lo = rank + 1
        if best_feasible is not None:
            upper_rank = best_feasible
            print(
                f"{kind.upper()}({n},{k}) alpha={alpha:.6g}: "
                f"binary-search minimum rank<={best_feasible}"
            )
        else:
            print(
                f"{kind.upper()}({n},{k}) alpha={alpha:.6g}: "
                f"no feasible rank found in [1,{upper_rank}]"
            )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, object]], path: Path) -> None:
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
        ax.plot(alphas, minimums, "o-", color=color, label=rf"$k={k}$ BM min found")
        ax.axhline(
            benchmark,
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.5,
            label=rf"$k={k}$ symmetric rank {benchmark}",
        )
    ns = sorted({int(row["n"]) for row in rows})
    title_n = str(ns[0]) if len(ns) == 1 else ",".join(str(n) for n in ns)
    kinds = sorted({str(row.get("kind", "threshold")) for row in rows})
    title_kind = kinds[0].upper() if len(kinds) == 1 else "mixed"
    ax.set_title(rf"Burer-Monteiro search for $\mathrm{{{title_kind}}}(n,k)$, $n={title_n}$")
    ax.set_xlabel(r"query relaxation $\alpha$")
    ax.set_ylabel(r"minimum feasible rank bound per $X_i$")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("threshold", "exact"), default="threshold")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--ks", default="2,3")
    parser.add_argument("--ranks-k2", default="beat_down")
    parser.add_argument("--ranks-k3", default="beat_down")
    parser.add_argument("--ranks-k4", default="beat_down")
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=180)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument(
        "--no-stop-after-first-infeasible",
        action="store_true",
        help="continue testing smaller ranks after the first not_found result",
    )
    parser.add_argument("--output", type=Path, default=Path("bm_threshold_n6.png"))
    parser.add_argument("--csv", type=Path, default=Path("bm_threshold_n6.csv"))
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for k in sorted({int(value.strip()) for value in args.ks.split(",") if value.strip()}):
        if k < 1 or k > args.n:
            parser.error(f"k={k} is outside 1..n")
        benchmark = comb(args.n - 1, k - 1) if args.kind == "threshold" else comb(args.n, k)
        spec = {
            2: args.ranks_k2,
            3: args.ranks_k3,
            4: args.ranks_k4,
        }.get(k, "beat_down")
        ranks = parse_rank_spec(spec, k, benchmark)
        if not ranks:
            parser.error(f"no ranks selected below benchmark for k={k}")
        rows.extend(
            run_case(
                args.kind,
                args.n,
                k,
                ranks,
                points=args.points,
                alphas=args.alphas,
                epsilon=args.epsilon,
                restarts=args.restarts,
                max_nfev=args.max_nfev,
                feasibility_tol=args.feasibility_tol,
                stop_after_first_infeasible=not args.no_stop_after_first_infeasible,
            )
        )
    write_csv(rows, args.csv)
    plot_rows(rows, args.output)
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
