"""Plot alpha-vs-minimum-rank curves from BM bisection summaries."""

from __future__ import annotations

import csv
from collections import defaultdict
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSV_FILES = [
    Path("exact_alpha_bisect_n5_refined.csv"),
    Path("exact_alpha_intermediate_n5.csv"),
    Path("exact_alpha_bisect_n6_refined.csv"),
    Path("exact_alpha_intermediate_n6_k1.csv"),
    Path("exact_alpha_intermediate_n6_k2.csv"),
    Path("exact_alpha_intermediate_n6_k3.csv"),
    Path("threshold_alpha_bisect_n5_refined.csv"),
    Path("threshold_alpha_bisect_n5_k3_refined.csv"),
    Path("threshold_alpha_intermediate_n5.csv"),
    Path("threshold_alpha_bisect_n6_refined.csv"),
    Path("threshold_alpha_intermediate_n6.csv"),
]


def benchmark_rank(kind: str, n: int, k: int) -> int:
    if kind == "exact":
        return comb(n, k)
    if kind == "threshold":
        return comb(n - 1, k - 1)
    raise ValueError(kind)


def load_thresholds():
    groups: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for path in CSV_FILES:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                groups[
                    (
                        row["kind"],
                        int(row["n"]),
                        int(row["k"]),
                        int(row["rank"]),
                    )
                ].append(row)

    thresholds: dict[tuple[str, int, int], dict[int, float]] = defaultdict(dict)
    raw: dict[tuple[str, int, int], dict[int, float]] = defaultdict(dict)
    for (kind, n, k, rank), rows in groups.items():
        feasible = [float(row["alpha"]) for row in rows if row["feasible"] == "True"]
        if feasible:
            raw[(kind, n, k)][rank] = min(feasible)
    for key, rank_to_alpha in raw.items():
        # Enforce monotonicity: if rank r is feasible, all larger ranks are feasible
        # at the same alpha. This smooths out nonconvex optimizer misses.
        best = float("inf")
        for rank in sorted(rank_to_alpha):
            best = min(best, rank_to_alpha[rank])
            thresholds[key][rank] = best
    return thresholds


def curve_from_thresholds(kind: str, n: int, k: int, rank_to_alpha: dict[int, float]):
    base = benchmark_rank(kind, n, k)
    events = sorted((alpha, rank) for rank, alpha in rank_to_alpha.items())
    xs = [1.0]
    ys = [base]
    current = base
    eps = 1e-6
    for alpha, rank in events:
        if rank >= current:
            continue
        xs.extend([alpha - eps, alpha])
        ys.extend([current, rank])
        current = rank
    xmax = max([alpha for alpha, _ in events] + [1.0])
    xs.append(xmax * 1.05)
    ys.append(current)
    return xs, ys


def make_plot(path: Path, *, xlim: tuple[float, float] | None = None, title_suffix: str = "") -> None:
    thresholds = load_thresholds()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = {5: "-", 6: "--"}

    for ax, kind in zip(axes, ["exact", "threshold"]):
        cases = sorted(key for key in thresholds if key[0] == kind)
        for idx, (_, n, k) in enumerate(cases):
            xs, ys = curve_from_thresholds(kind, n, k, thresholds[(kind, n, k)])
            ax.step(
                xs,
                ys,
                where="post",
                color=colors[idx % len(colors)],
                linestyle=linestyles.get(n, "-"),
                linewidth=2,
                label=rf"$n={n}, k={k}$",
            )
            ax.scatter(
                [thresholds[(kind, n, k)][rank] for rank in sorted(thresholds[(kind, n, k)])],
                sorted(thresholds[(kind, n, k)]),
                color=colors[idx % len(colors)],
                s=28,
                zorder=3,
            )
        ax.set_title(kind.upper())
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("minimum feasible rank found")
        if xlim is not None:
            ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"BM tradeoff curves reconstructed from alpha bisection{title_suffix}")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def main() -> None:
    make_plot(Path("alpha_min_rank_curves.png"))
    make_plot(Path("alpha_min_rank_curves_zoom_near1.png"), xlim=(1.0, 1.12), title_suffix="; zoom near alpha=1")
    make_plot(Path("alpha_min_rank_curves_zoom_rank2.png"), xlim=(1.0, 1.75), title_suffix="; zoom to rank-2 drops")
    make_plot(Path("alpha_min_rank_curves_zoom_rank1.png"), xlim=(2.0, 4.8), title_suffix="; rank-1 region")


if __name__ == "__main__":
    main()
