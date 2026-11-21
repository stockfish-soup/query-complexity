#!/usr/bin/env python3
from pathlib import Path
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = (Path(__file__).resolve().parent / "results")


def read_balanced_max_t():
    rows = []
    with open(ROOT / "balanced_max_t_surgery_values.csv") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "n": int(r["n"]),
                    "t": int(r["t"]),
                    "max_c": int(r["max_c"]),
                    "value": float(r["value"]),
                    "inputs": int(r["inputs"]),
                    "pairs": int(r["pair_orbits"]),
                }
            )
    return sorted(rows, key=lambda r: r["n"])


def read_positive_adversary():
    path = ROOT / "positive_adversary_benchmarks.csv"
    if not path.exists():
        return {}
    by_family = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_family.setdefault(r["family"], []).append((int(r["n"]), float(r["value"])))
    return {name: sorted(vals) for name, vals in by_family.items()}


def main():
    surgery = read_balanced_max_t()
    positive = read_positive_adversary()

    complete_bipartite = {
        4: 2.794191,
        5: 3.663603,
        6: 4.457446,
        7: 5.179195,
        8: 5.894469,
    }
    best_known = {
        4: 3.000000,
        5: 4.063267,
        6: 5.219912,
        7: 6.231963,
        8: 6.911788,
    }

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.4))

    ns = [r["n"] for r in surgery]
    vals = [r["value"] for r in surgery]
    ax.plot(ns, vals, marker="o", linewidth=2.5, color="#16a34a", label="balanced max-t surgery")

    for r in surgery:
        ax.annotate(
            f"t={r['t']}, c={r['max_c']}\n{r['value']:.3f}",
            (r["n"], r["value"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )

    c_ns = sorted(n for n in complete_bipartite if min(ns) <= n <= max(ns))
    ax.plot(
        c_ns,
        [complete_bipartite[n] for n in c_ns],
        marker="s",
        linewidth=2.2,
        color="#2563eb",
        label="complete bipartite cumulative",
    )

    if "balanced max-t surgery" in positive:
        p_ns, p_vals = zip(*positive["balanced max-t surgery"])
        ax.plot(
            p_ns,
            p_vals,
            marker="o",
            linestyle=":",
            linewidth=2.2,
            color="#15803d",
            label="positive ADV: max-t surgery",
        )

    if "complete bipartite" in positive:
        p_ns, p_vals = zip(*positive["complete bipartite"])
        ax.plot(
            p_ns,
            p_vals,
            marker="s",
            linestyle=":",
            linewidth=2.2,
            color="#1d4ed8",
            label="positive ADV: complete bipartite",
        )

    b_ns = sorted(n for n in best_known if min(ns) <= n <= max(ns))
    ax.plot(
        b_ns,
        [best_known[n] for n in b_ns],
        marker="D",
        linestyle="--",
        linewidth=1.8,
        color="#111827",
        label="best known restricted",
    )

    barrier_ns = list(range(min(ns), max(max(ns), max(complete_bipartite)) + 1))
    ax.plot(
        barrier_ns,
        [math.sqrt(3.0 * n * (n - 1) / 2.0) for n in barrier_ns],
        linestyle="-.",
        linewidth=1.8,
        color="#dc2626",
        label=r"certificate barrier $\sqrt{3\binom{n}{2}}$",
    )

    ax.set_title("Balanced Max-t Surgery vs Complete Bipartite")
    ax.set_xlabel("n")
    ax.set_ylabel("restricted adversary value")
    ax.legend(loc="lower right")
    fig.tight_layout()

    svg = ROOT / "t_surgery_vs_complete_bipartite.svg"
    png = ROOT / "t_surgery_vs_complete_bipartite.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    print(svg)
    print(png)


if __name__ == "__main__":
    main()
