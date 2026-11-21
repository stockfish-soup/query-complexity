"""Plot BM alpha-threshold/rank patterns from refined bisection CSV files."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSV_FILES = [
    Path("exact_alpha_bisect_n5_refined.csv"),
    Path("exact_alpha_bisect_n6_refined.csv"),
    Path("threshold_alpha_bisect_n5_refined.csv"),
    Path("threshold_alpha_bisect_n6_refined.csv"),
]


def load_summaries():
    groups: dict[tuple[str, int, int, int], list[dict[str, str]]] = defaultdict(list)
    for path in CSV_FILES:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (
                    row["kind"],
                    int(row["n"]),
                    int(row["k"]),
                    int(row["rank"]),
                )
                groups[key].append(row)

    out = []
    for (kind, n, k, rank), rows in sorted(groups.items()):
        feasible = [float(row["alpha"]) for row in rows if row["feasible"] == "True"]
        not_found = [float(row["alpha"]) for row in rows if row["feasible"] == "False"]
        if not feasible:
            continue
        hi = min(feasible)
        lo = max((value for value in not_found if value < hi), default=np.nan)
        out.append(
            {
                "kind": kind,
                "n": n,
                "k": k,
                "rank": rank,
                "alpha_hi": hi,
                "alpha_lo": lo,
                "width": hi - lo if np.isfinite(lo) else np.nan,
            }
        )
    return out


def main() -> None:
    rows = load_summaries()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    markers = {5: "o", 6: "s"}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, kind in zip(axes, ["exact", "threshold"]):
        kind_rows = [row for row in rows if row["kind"] == kind]
        for idx, (n, k) in enumerate(sorted({(row["n"], row["k"]) for row in kind_rows})):
            case = sorted(
                [row for row in kind_rows if row["n"] == n and row["k"] == k],
                key=lambda row: row["rank"],
            )
            xs = [row["rank"] for row in case]
            ys = [row["alpha_hi"] for row in case]
            lower = [
                row["alpha_hi"] - row["alpha_lo"] if np.isfinite(row["alpha_lo"]) else 0.0
                for row in case
            ]
            ax.errorbar(
                xs,
                ys,
                yerr=[lower, [0.0] * len(xs)],
                marker=markers.get(n, "o"),
                linestyle="-",
                capsize=4,
                color=colors[idx % len(colors)],
                label=rf"$n={n}, k={k}$",
            )
        ax.set_title(kind.upper())
        ax.set_xlabel("target rank bound")
        ax.set_ylabel(r"smallest feasible $\alpha$ found")
        ax.set_xticks(sorted({row["rank"] for row in kind_rows}))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.invert_xaxis()

    fig.suptitle("Burer-Monteiro rank/query tradeoff: alpha jump thresholds")
    fig.tight_layout()
    out = Path("alpha_rank_patterns.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
