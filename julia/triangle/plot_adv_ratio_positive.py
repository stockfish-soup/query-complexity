#!/usr/bin/env python3
from pathlib import Path
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = (Path(__file__).resolve().parent / "results")


GENERAL_COMPLETE_BIPARTITE = {
    4: 2.794191,
    5: 3.663603,
    6: 4.457446,
    7: 5.179195,
    8: 5.894469,
}


def read_positive():
    by_family = {}
    with open(ROOT / "positive_adversary_benchmarks.csv") as f:
        for row in csv.DictReader(f):
            by_family.setdefault(row["family"], {})[int(row["n"])] = float(row["value"])
    return by_family


def read_balanced_max_t_general():
    vals = {}
    with open(ROOT / "balanced_max_t_surgery_values.csv") as f:
        for row in csv.DictReader(f):
            vals[int(row["n"])] = float(row["value"])
    return vals


def read_greedy_n5():
    rows = []
    with open(ROOT / "positive_almost_forward_n5.csv") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "round": int(row["round"]),
                    "rep": int(row["rep"]),
                    "c": int(row["c"]),
                    "ratio": float(row["dual"]) / float(row["positive"]),
                }
            )
    return rows


def read_requested_ratios():
    path = ROOT / "requested_positive_ratios.csv"
    if not path.exists():
        return {}
    by_family = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            by_family.setdefault(row["family"], []).append(
                (int(row["n"]), float(row["ratio"]))
            )
    return {name: sorted(vals) for name, vals in by_family.items()}


def ratio_points(general, positive):
    points = []
    for n in sorted(set(general) & set(positive)):
        points.append((n, general[n] / positive[n]))
    return points


def main():
    positive = read_positive()
    max_t_general = read_balanced_max_t_general()
    greedy = read_greedy_n5()
    requested = read_requested_ratios()

    series = {
        "balanced max-t surgery": ratio_points(
            max_t_general,
            positive.get("balanced max-t surgery", {}),
        ),
        "complete bipartite": ratio_points(
            GENERAL_COMPLETE_BIPARTITE,
            positive.get("complete bipartite", {}),
        ),
    }
    for name, pts in requested.items():
        series[f"{name} (requested)"] = pts

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))

    ax = axes[0]
    styles = {
        "balanced max-t surgery": ("#16a34a", "o"),
        "complete bipartite": ("#2563eb", "s"),
        "complete bipartite (requested)": ("#1d4ed8", "P"),
        "max-t surgery no base (requested)": ("#dc2626", "X"),
    }
    for name, pts in series.items():
        if not pts:
            continue
        color, marker = styles.get(name, ("#64748b", "o"))
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, marker=marker, linewidth=2.4, color=color, label=name)
        for x, y in pts:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.2)
    ax.set_title(r"Promise Families: $\mathrm{Adv}^{\pm}/\mathrm{Adv}^{+}$")
    ax.set_xlabel("n")
    ax.set_ylabel("ratio")
    ax.legend(loc="upper left")

    ax = axes[1]
    xs = [r["round"] for r in greedy]
    ys = [r["ratio"] for r in greedy]
    ax.plot(xs, ys, marker="D", linewidth=2.4, color="#9333ea", label="n=5 greedy by round")
    for r in greedy:
        ax.annotate(
            f"rep {r['rep']}\nc={r['c']}\n{r['ratio']:.3f}",
            (r["round"], r["ratio"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.2)
    ax.set_title("Greedy Search, n=5")
    ax.set_xlabel("round")
    ax.set_ylabel("ratio")
    ax.legend(loc="upper left")

    fig.tight_layout()
    svg = ROOT / "adv_ratio_positive_vs_general.svg"
    png = ROOT / "adv_ratio_positive_vs_general.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=180)

    csv_path = ROOT / "adv_ratio_positive_vs_general.csv"
    with open(csv_path, "w") as f:
        f.write("family,x,ratio\n")
        for name, pts in series.items():
            for x, y in pts:
                f.write(f"{name},{x},{y}\n")
        for r in greedy:
            f.write(f"n=5 greedy round,{r['round']},{r['ratio']}\n")

    print(svg)
    print(png)
    print(csv_path)


if __name__ == "__main__":
    main()
