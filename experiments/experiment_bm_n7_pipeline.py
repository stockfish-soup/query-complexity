"""One-command BM rank/alpha pipeline for EXACT or THRESHOLD instances.

The pipeline has two stages:

1. Fast alpha sweep.  For each alpha it binary-searches the smallest feasible
   rank, using the previous alpha's best rank as the next upper bound.
2. Refined alpha bisection.  For each rank drop seen in the fast sweep it
   bisects alpha to estimate the jump location.

All CSV files are checkpointed incrementally after every rank test.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from math import comb
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


SWEEP_FIELDS = [
    "stage",
    "kind",
    "n",
    "k",
    "alpha",
    "benchmark_rank",
    "rank_bound",
    "feasible",
    "coordinate_ranks",
    "trace",
    "max_equality_residual",
    "max_cap_violation",
    "status",
    "attempts",
    "total_nfev",
]

BISECT_FIELDS = [
    "stage",
    "kind",
    "n",
    "k",
    "rank",
    "alpha",
    "lo",
    "hi",
    "feasible",
    "max_equality_residual",
    "max_cap_violation",
    "attempts",
    "total_nfev",
]


@dataclass
class Instance:
    kind: str
    n: int
    k: int
    layers: dict[int, int]
    inputs: list[tuple[int, ...]]
    outputs: list[int]
    benchmark_rank: int
    adversary_value: float


class CsvCheckpoint:
    def __init__(self, path: Path, fields: list[str]):
        self.path = path
        self.fields = fields
        self.rows: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        exists = self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()


def parse_float_list(spec: str) -> list[float]:
    return sorted({float(part.strip()) for part in spec.split(",") if part.strip()})


def make_instance(kind: str, n: int, k: int, epsilon: float) -> Instance:
    if kind == "exact":
        layers = qc.EXACT_layers(n, k)
        benchmark = comb(n, k)
    elif kind == "threshold":
        layers = qc.THRESHOLD_layers(n, k)
        benchmark = comb(n - 1, k - 1)
    else:
        raise ValueError(kind)
    inputs = qc.all_binary_inputs(n)
    outputs = [layers[sum(x)] for x in inputs]
    adversary = qc.solve_symmetric_trace_minimized_dual(
        n,
        layers,
        alpha=1.0,
        epsilon=epsilon,
    ).objective_value
    return Instance(kind, n, k, layers, inputs, outputs, benchmark, adversary)


def symmetric_start(instance: Instance, alpha: float, epsilon: float) -> qc.AdversarySolution:
    orbit = qc.solve_symmetric_trace_minimized_dual(
        instance.n,
        instance.layers,
        alpha=alpha,
        adversary_value=instance.adversary_value,
        epsilon=epsilon,
    )
    return qc.expand_orbit_solution(orbit, psd_tol=1e-7)


def test_rank(
    instance: Instance,
    rank: int,
    alpha: float,
    *,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    seed: int,
    initial_solution: qc.AdversarySolution | None,
    initial_factors,
):
    return qc.solve_fixed_rank_dual_feasibility(
        instance.inputs,
        instance.outputs,
        rank=rank,
        alpha=alpha,
        adversary_value=instance.adversary_value,
        epsilon=epsilon,
        initial_solution=initial_solution if initial_factors is None else None,
        initial_factors=initial_factors,
        restarts=restarts,
        random_seed=seed,
        max_nfev=max_nfev,
        feasibility_tol=feasibility_tol,
    )


def append_sweep_row(
    checkpoint: CsvCheckpoint,
    instance: Instance,
    alpha: float,
    rank: int,
    result,
) -> None:
    checkpoint.append(
        {
            "stage": "fast",
            "kind": instance.kind,
            "n": instance.n,
            "k": instance.k,
            "alpha": alpha,
            "benchmark_rank": instance.benchmark_rank,
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


def fast_sweep(
    instance: Instance,
    alphas: list[float],
    *,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    checkpoint: CsvCheckpoint,
) -> dict[float, int]:
    warm_starts: dict[int, list[np.ndarray]] = {}
    min_rank_by_alpha: dict[float, int] = {}
    upper_rank = instance.benchmark_rank - 1

    for alpha in alphas:
        start = symmetric_start(instance, alpha, epsilon)
        lo, hi = 1, upper_rank
        best: int | None = None
        previous_feasible = None
        print(
            f"{instance.kind.upper()}({instance.n},{instance.k}) alpha={alpha:.8g}: "
            f"rank search in [1,{upper_rank}]",
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
                seed=70_000 + 1_000 * instance.k + 100 * rank + round(1000 * alpha),
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
        if best is not None:
            min_rank_by_alpha[alpha] = best
            upper_rank = best
            print(f"  minimum rank found: {best}", flush=True)
        else:
            print("  no feasible rank found in current interval", flush=True)
    return min_rank_by_alpha


def drop_cases_from_sweep(min_rank_by_alpha: dict[float, int]) -> list[tuple[int, float, float]]:
    """Return (rank, lo_alpha, hi_alpha) for each observed rank drop."""
    if not min_rank_by_alpha:
        return []
    pairs = sorted(min_rank_by_alpha.items())
    cases = []
    previous_alpha, previous_rank = pairs[0]
    for alpha, rank in pairs[1:]:
        if rank < previous_rank:
            for target_rank in range(rank, previous_rank):
                cases.append((target_rank, previous_alpha, alpha))
        previous_alpha, previous_rank = alpha, rank
    return cases


def refine_alpha(
    instance: Instance,
    rank: int,
    lo: float,
    hi: float,
    *,
    iters: int,
    epsilon: float,
    restarts: int,
    max_nfev: int,
    feasibility_tol: float,
    checkpoint: CsvCheckpoint,
) -> None:
    warm_factors = None
    best_factors = None
    for step in range(iters):
        mid = 0.5 * (lo + hi)
        start = symmetric_start(instance, mid, epsilon)
        print(
            f"{instance.kind.upper()}({instance.n},{instance.k}) rank<={rank}: "
            f"alpha={mid:.10g} in [{lo:.10g},{hi:.10g}]",
            flush=True,
        )
        result = test_rank(
            instance,
            rank,
            mid,
            epsilon=epsilon,
            restarts=restarts,
            max_nfev=max_nfev,
            feasibility_tol=feasibility_tol,
            seed=90_000 + 1_000 * instance.k + 100 * rank + step,
            initial_solution=start,
            initial_factors=warm_factors,
        )
        checkpoint.append(
            {
                "stage": "refine",
                "kind": instance.kind,
                "n": instance.n,
                "k": instance.k,
                "rank": rank,
                "alpha": mid,
                "lo": lo,
                "hi": hi,
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
            best_factors = result.z
            warm_factors = result.z
        else:
            lo = mid
            warm_factors = best_factors
    print(
        f"{instance.kind.upper()}({instance.n},{instance.k}) rank<={rank}: "
        f"final bracket ({lo:.10g}, {hi:.10g}]",
        flush=True,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def min_rank_summary(sweep_csv: Path) -> dict[tuple[str, int, int], dict[float, int]]:
    grouped: dict[tuple[str, int, int, float], list[int]] = defaultdict(list)
    for row in read_rows(sweep_csv):
        if row.get("feasible") == "True":
            grouped[
                (row["kind"], int(row["n"]), int(row["k"]), float(row["alpha"]))
            ].append(int(row["rank_bound"]))
    out: dict[tuple[str, int, int], dict[float, int]] = defaultdict(dict)
    for (kind, n, k, alpha), ranks in grouped.items():
        out[(kind, n, k)][alpha] = min(ranks)
    return out


def refined_thresholds(refine_csv: Path) -> dict[tuple[str, int, int], dict[int, float]]:
    grouped: dict[tuple[str, int, int, int], list[float]] = defaultdict(list)
    for row in read_rows(refine_csv):
        if row.get("feasible") == "True":
            grouped[(row["kind"], int(row["n"]), int(row["k"]), int(row["rank"]))].append(
                float(row["alpha"])
            )
    out: dict[tuple[str, int, int], dict[int, float]] = defaultdict(dict)
    for (kind, n, k, rank), alphas in grouped.items():
        out[(kind, n, k)][rank] = min(alphas)
    return out


def plot_tradeoff(sweep_csv: Path, refine_csv: Path, prefix: Path) -> None:
    sweep = min_rank_summary(sweep_csv)
    refined = refined_thresholds(refine_csv)
    for key, rank_to_alpha in refined.items():
        if key not in sweep:
            sweep[key] = {}
        # Add refined step points to the summary.
        for rank, alpha in rank_to_alpha.items():
            sweep[key][alpha] = min(rank, sweep[key].get(alpha, rank))

    if not sweep:
        return
    for suffix, xlim in [
        ("full", None),
        ("near1", (1.0, 1.15)),
        ("rank2", (1.0, 2.0)),
        ("rank1", (2.0, 8.0)),
    ]:
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for idx, key in enumerate(sorted(sweep)):
            kind, n, k = key
            points = sorted(sweep[key].items())
            if not points:
                continue
            xs = [points[0][0]]
            ys = [points[0][1]]
            current = points[0][1]
            for alpha, rank in points[1:]:
                current = min(current, rank)
                xs.append(alpha)
                ys.append(current)
            ax.step(xs, ys, where="post", marker="o", color=colors[idx % len(colors)], label=f"{kind} n={n} k={k}")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("lowest feasible rank found")
        ax.set_title(f"BM tradeoff {prefix.name} ({suffix})")
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
    parser.add_argument("--alphas", default="1,1.01,1.02,1.04,1.08,1.15,1.25,1.5,2,3,4,6,8,10")
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--fast-restarts", type=int, default=0)
    parser.add_argument("--fast-max-nfev", type=int, default=220)
    parser.add_argument("--refine-restarts", type=int, default=0)
    parser.add_argument("--refine-max-nfev", type=int, default=500)
    parser.add_argument("--refine-iters", type=int, default=7)
    parser.add_argument("--skip-refine", action="store_true")
    parser.add_argument("--out-prefix", type=Path, default=Path("bm_n7"))
    args = parser.parse_args()

    alphas = parse_float_list(args.alphas)
    ks = sorted({int(part.strip()) for part in args.ks.split(",") if part.strip()})
    sweep_csv = args.out_prefix.with_name(f"{args.out_prefix.name}_fast.csv")
    refine_csv = args.out_prefix.with_name(f"{args.out_prefix.name}_refined.csv")
    for path in [sweep_csv, refine_csv]:
        if path.exists():
            path.unlink()

    sweep_checkpoint = CsvCheckpoint(sweep_csv, SWEEP_FIELDS)
    refine_checkpoint = CsvCheckpoint(refine_csv, BISECT_FIELDS)

    for k in ks:
        instance = make_instance(args.kind, args.n, k, args.epsilon)
        min_rank_by_alpha = fast_sweep(
            instance,
            alphas,
            epsilon=args.epsilon,
            restarts=args.fast_restarts,
            max_nfev=args.fast_max_nfev,
            feasibility_tol=args.feasibility_tol,
            checkpoint=sweep_checkpoint,
        )
        if args.skip_refine:
            continue
        for rank, lo, hi in drop_cases_from_sweep(min_rank_by_alpha):
            refine_alpha(
                instance,
                rank,
                lo,
                hi,
                iters=args.refine_iters,
                epsilon=args.epsilon,
                restarts=args.refine_restarts,
                max_nfev=args.refine_max_nfev,
                feasibility_tol=args.feasibility_tol,
                checkpoint=refine_checkpoint,
            )

    plot_tradeoff(sweep_csv, refine_csv, args.out_prefix)
    print(f"Wrote {sweep_csv}")
    if not args.skip_refine:
        print(f"Wrote {refine_csv}")


if __name__ == "__main__":
    main()
