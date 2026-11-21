"""Plot n=5 and n=6 Burer-Monteiro threshold rank curves side by side."""

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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def plot_panel(ax, rows: list[dict[str, str]], title: str) -> None:
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
        ax.plot(alphas, minimums, "o-", color=color, label=rf"$k={k}$ BM min")
        ax.axhline(
            benchmark,
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.5,
            label=rf"$k={k}$ sym rank {benchmark}",
        )
    ax.set_title(title)
    ax.set_xlabel(r"query relaxation $\alpha$")
    ax.set_ylabel(r"minimum feasible rank per $X_i$")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n5-csv", type=Path, default=Path("nonsymmetric_n5_min_rank.csv"))
    parser.add_argument("--n6-csv", type=Path, default=Path("bm_threshold_n6.csv"))
    parser.add_argument("--output", type=Path, default=Path("bm_threshold_n5_n6.png"))
    args = parser.parse_args()

    n5_rows = read_rows(args.n5_csv)
    n6_rows = read_rows(args.n6_csv)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=False)
    plot_panel(axes[0], n5_rows, r"$n=5$")
    plot_panel(axes[1], n6_rows, r"$n=6$")
    fig.suptitle("Burer-Monteiro minimum rank found for THRESHOLD")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
