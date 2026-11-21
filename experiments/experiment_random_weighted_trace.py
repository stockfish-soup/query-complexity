"""Random weighted diagonal objectives for nonsymmetric dual witnesses."""

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


def layer_values(kind: str, n: int, k: int) -> dict[int, int]:
    if kind == "exact":
        return qc.EXACT_layers(n, k)
    if kind == "threshold":
        return qc.THRESHOLD_layers(n, k)
    raise ValueError(f"unknown function kind {kind!r}")


def make_instance(kind: str, n: int, k: int) -> tuple[list[tuple[int, ...]], list[int]]:
    layers = layer_values(kind, n, k)
    inputs = qc.all_binary_inputs(n)
    outputs = [layers[sum(x)] for x in inputs]
    return inputs, outputs


def random_weights(rng: np.random.Generator, m: int, distribution: str, spread: float) -> np.ndarray:
    if distribution == "lognormal":
        weights = rng.lognormal(mean=0.0, sigma=spread, size=m)
    elif distribution == "uniform":
        weights = rng.uniform(1.0, spread, size=m)
    elif distribution == "normal":
        weights = 1.0 + spread * rng.standard_normal(m)
        weights = weights - np.min(weights) + 1e-3
    else:
        raise ValueError(f"unknown distribution {distribution!r}")
    return weights / float(np.mean(weights))


def run_case(
    kind: str,
    n: int,
    k: int,
    alpha: float,
    *,
    samples: int,
    distribution: str,
    spread: float,
    epsilon: float,
    solver: str,
    psd_tol: float,
    seed: int,
) -> list[dict[str, object]]:
    inputs, outputs = make_instance(kind, n, k)
    value = qc.solve_adversary_value_sdp(inputs, outputs, solver=solver).objective_value
    rows: list[dict[str, object]] = []

    baseline = qc.solve_trace_minimized_dual(
        inputs,
        outputs,
        alpha=alpha,
        adversary_value=value,
        epsilon=epsilon,
        solver=solver,
        psd_tol=psd_tol,
    )
    rows.append(
        {
            "kind": kind,
            "n": n,
            "k": k,
            "alpha": alpha,
            "sample": "unweighted",
            "seed": "",
            "distribution": "",
            "spread": "",
            "objective": baseline.trace_value,
            "max_rank": max(baseline.witness_dims),
            "total_rank": sum(baseline.witness_dims),
            "coordinate_ranks": ";".join(map(str, baseline.witness_dims)),
            "max_input_cost": float(np.max(baseline.input_costs)),
            "weight_min": "",
            "weight_max": "",
            "weight_std": "",
            "status": baseline.status,
        }
    )
    print(
        f"{kind.upper()}({n},{k}) alpha={alpha:.6g} unweighted: "
        f"max_rank={max(baseline.witness_dims)}, ranks={baseline.witness_dims}"
    )

    rng = np.random.default_rng(seed)
    for sample in range(samples):
        weights = random_weights(rng, len(inputs), distribution, spread)
        solution = qc.solve_trace_minimized_dual(
            inputs,
            outputs,
            alpha=alpha,
            adversary_value=value,
            diagonal_weights=weights,
            epsilon=epsilon,
            solver=solver,
            psd_tol=psd_tol,
        )
        rows.append(
            {
                "kind": kind,
                "n": n,
                "k": k,
                "alpha": alpha,
                "sample": sample,
                "seed": seed + sample,
                "distribution": distribution,
                "spread": spread,
                "objective": solution.trace_value,
                "max_rank": max(solution.witness_dims),
                "total_rank": sum(solution.witness_dims),
                "coordinate_ranks": ";".join(map(str, solution.witness_dims)),
                "max_input_cost": float(np.max(solution.input_costs)),
                "weight_min": float(np.min(weights)),
                "weight_max": float(np.max(weights)),
                "weight_std": float(np.std(weights)),
                "status": solution.status,
            }
        )
        print(
            f"{kind.upper()}({n},{k}) alpha={alpha:.6g} sample={sample}: "
            f"max_rank={max(solution.witness_dims)}, ranks={solution.witness_dims}, "
            f"weights=[{np.min(weights):.3g},{np.max(weights):.3g}]"
        )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_rows(rows: list[dict[str, object]], path: Path) -> None:
    weighted = [row for row in rows if row["sample"] != "unweighted"]
    fig, (max_ax, total_ax) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    labels = []
    max_ranks = []
    total_ranks = []
    colors = []
    for idx, row in enumerate(rows):
        labels.append(str(row["sample"]))
        max_ranks.append(int(row["max_rank"]))
        total_ranks.append(int(row["total_rank"]))
        colors.append("#555555" if row["sample"] == "unweighted" else "#1f77b4")

    x = np.arange(len(rows))
    max_ax.bar(x, max_ranks, color=colors)
    total_ax.bar(x, total_ranks, color=colors)
    for ax, values, title in (
        (max_ax, max_ranks, "maximum coordinate rank"),
        (total_ax, total_ranks, "total private rank"),
    ):
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.28)
        for xpos, value in zip(x, values):
            ax.text(xpos, value + 0.05, str(value), ha="center", va="bottom", fontsize=8)

    first = rows[0]
    best = min(weighted, key=lambda row: (int(row["max_rank"]), int(row["total_rank"]))) if weighted else first
    fig.suptitle(
        rf"{str(first['kind']).upper()}({first['n']},{first['k']}), "
        rf"$\alpha={float(first['alpha']):.3g}$; "
        rf"best weighted max rank {best['max_rank']}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("exact", "threshold"), default="exact")
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--distribution", choices=("lognormal", "uniform", "normal"), default="lognormal")
    parser.add_argument("--spread", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--psd-tol", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, default=Path("random_weighted_trace.png"))
    parser.add_argument("--csv", type=Path, default=Path("random_weighted_trace.csv"))
    args = parser.parse_args()

    if args.n <= 0 or args.k < 0 or args.k > args.n:
        parser.error("require n > 0 and 0 <= k <= n")
    if args.alpha < 1.0:
        parser.error("--alpha must be at least 1")
    if args.samples < 0:
        parser.error("--samples must be non-negative")
    if args.spread <= 0:
        parser.error("--spread must be positive")

    rows = run_case(
        args.kind,
        args.n,
        args.k,
        args.alpha,
        samples=args.samples,
        distribution=args.distribution,
        spread=args.spread,
        epsilon=args.epsilon,
        solver=args.solver,
        psd_tol=args.psd_tol,
        seed=args.seed,
    )
    write_csv(rows, args.csv)
    plot_rows(rows, args.output)
    print(f"Wrote {args.output} and {args.csv}")


if __name__ == "__main__":
    main()
