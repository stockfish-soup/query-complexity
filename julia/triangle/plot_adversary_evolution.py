#!/usr/bin/env python3
"""Plot measured triangle adversary values for the restricted input families."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTDIR = (Path(__file__).resolve().parent / "results")


SERIES = {
    "best MOSEK known": [
        (4, 3.000000),
        (5, 4.063267),
        (6, 5.219912),
        (7, 6.231963),
        (8, 6.911788),
    ],
    "s(G) greedy max round": [
        (4, 3.000000),
        (5, 4.063258),
        (6, 5.219912),
        (7, 6.231963),
        (8, 6.911788),
    ],
    "fixed complete |A|=1": [
        (4, 2.299322),
        (5, 2.990721),
        (6, 3.691801),
        (7, 4.395720),
        (8, 5.100853),
    ],
    "fixed complete |A|=2": [
        (4, 2.000000),
        (5, 2.958640),
        (6, 3.646483),
        (7, 4.297688),
        (8, 4.962089),
    ],
    "fixed complete |A|=3": [
        (6, 3.292819),
        (7, 4.105505),
        (8, 4.802603),
    ],
    "fixed complete |A|=4": [
        (8, 4.457027),
    ],
    "cumulative complete best": [
        (4, 2.794191),
        (5, 3.663603),
        (6, 4.457446),
        (7, 5.179195),
        (8, 5.894469),
    ],
    "1-surgery family": [
        (5, 3.881000),
        (6, 5.053339),
        (7, 6.153507),
    ],
    "2-surgery cumulative": [
        (6, 5.127890),
    ],
    "preserved bipartite best": [
        (5, 3.844139),
        (6, 4.380269),
        (7, 4.366601),
    ],
    "bipartite plus one": [
        (5, 4.005939),
    ],
}


ROUND_SERIES = {
    "n=4": [
        (1, 2.299322012),
        (2, 2.794191494),
        (3, 2.794191506),
        (4, 2.999999983),
    ],
    "n=5": [
        (1, 2.990720828),
        (2, 3.581834803),
        (3, 3.880999526),
        (4, 3.884557563),
        (5, 3.986348676),
        (6, 3.986349121),
        (7, 4.046403855),
        (8, 4.046403690),
        (9, 4.046403724),
        (10, 4.059807641),
        (11, 4.063257833),
    ],
    "n=6": [
        (1, 3.691801236),
        (2, 4.892707854),
        (3, 4.954884428),
        (4, 4.963194568),
        (5, 5.045457182),
        (6, 5.143366121),
        (7, 5.148552744),
        (8, 5.174055266),
        (16, 5.219911657),
    ],
    "n=7": [
        (1, 4.395719970),
        (2, 5.753855378),
        (3, 5.957143645),
        (4, 6.005144072),
        (5, 6.230222663),
        (6, 6.231962798),
    ],
    "n=8": [
        (1, 5.100853431),
        (2, 6.577611537),
        (3, 6.911788102),
    ],
}


MAX_ROUNDS = {
    4: (4, 2.999999983),
    5: (11, 4.063257833),
    6: (16, 5.219911657),
    7: (6, 6.231962798),
    8: (3, 6.911788102),
}

BENCHMARK_SERIES = {
    "best obtained": [
        (4, 3.000000),
        (5, 4.063267),
        (6, 5.219912),
        (7, 6.231963),
        (8, 6.911788),
    ],
    "complete bipartite": [
        (4, 2.794191),
        (5, 3.663603),
        (6, 4.457446),
        (7, 5.179195),
        (8, 5.894469),
    ],
    "1-surgery family": [
        (5, 3.881000),
        (6, 5.053339),
        (7, 6.153507),
    ],
    "2-surgery cumulative": [
        (6, 5.127890),
    ],
    "preserved bipartite": [
        (5, 3.844139),
        (6, 4.380269),
        (7, 4.366601),
    ],
    "heuristic round 2": [
        (4, 2.794191),
        (5, 3.581835),
        (6, 4.892708),
        (7, 5.753855),
        (8, 6.577612),
    ],
    "heuristic round 6": [
        (5, 3.986349),
        (6, 5.143366),
        (7, 6.231963),
    ],
    "heuristic round 16": [
        (6, 5.219912),
    ],
    "heuristic round 3": [
        (5, 3.880000),
        (6, 4.954884),
        (7, 5.957144),
        (8, 6.911788),
    ],
}


def scaled_power_curve(ns, exponent, anchor_n=6.0, anchor_value=5.219911656744336):
    return anchor_value * (ns / anchor_n) ** exponent


def plot_by_n():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )

    fig, ax = plt.subplots(figsize=(9.5, 6.2))

    colors = {
        "best MOSEK known": "#111827",
        "s(G) greedy max round": "#dc2626",
        "fixed complete |A|=1": "#2563eb",
        "fixed complete |A|=2": "#0f766e",
        "fixed complete |A|=3": "#6b7280",
        "fixed complete |A|=4": "#a16207",
        "cumulative complete best": "#0891b2",
        "1-surgery family": "#16a34a",
        "2-surgery cumulative": "#84cc16",
        "preserved bipartite best": "#0d9488",
        "bipartite plus one": "#9333ea",
    }
    markers = {
        "best MOSEK known": "o",
        "s(G) greedy max round": "D",
        "fixed complete |A|=1": "s",
        "fixed complete |A|=2": "^",
        "fixed complete |A|=3": "D",
        "fixed complete |A|=4": "v",
        "cumulative complete best": "X",
        "1-surgery family": "h",
        "2-surgery cumulative": "*",
        "preserved bipartite best": "p",
        "bipartite plus one": "P",
    }

    for name, pts in SERIES.items():
        pts = sorted(pts)
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        linestyle = "-" if len(pts) > 1 else "None"
        linewidth = 2.6 if name in {"best MOSEK known", "s(G) greedy max round"} else 1.8
        ax.plot(
            x,
            y,
            marker=markers[name],
            markersize=8,
            linewidth=linewidth,
            linestyle=linestyle,
            color=colors[name],
            label=name,
        )

    ns = np.linspace(4, 8, 200)
    for exponent, color in [(1.00, "#94a3b8"), (1.10, "#64748b"), (1.20, "#475569"), (1.25, "#1e293b")]:
        ax.plot(
            ns,
            scaled_power_curve(ns, exponent),
            linestyle="--",
            linewidth=1.4,
            color=color,
            alpha=0.85,
            label=rf"$5.220(n/6)^{{{exponent:.2f}}}$",
        )

    for n, (round_id, value) in MAX_ROUNDS.items():
        label = "full" if round_id is None else f"r{round_id}"
        ax.annotate(
            label,
            xy=(n, value),
            xytext=(4, 7),
            textcoords="offset points",
            fontsize=9,
            color="#111827",
        )

    ax.set_title("Triangle adversary lower bounds from restricted input domains")
    ax.set_xlabel("number of vertices n")
    ax.set_ylabel("adversary value")
    ax.set_xticks([4, 5, 6, 7, 8])
    ax.set_xlim(3.85, 8.15)
    ax.set_ylim(1.7, 7.9)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()

    svg = OUTDIR / "triangle_adversary_evolution.svg"
    png = OUTDIR / "triangle_adversary_evolution.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    print(svg)
    print(png)


def plot_by_round():
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    colors = {
        "n=4": "#64748b",
        "n=5": "#9333ea",
        "n=6": "#2563eb",
        "n=7": "#dc2626",
        "n=8": "#0f766e",
    }

    for name, pts in ROUND_SERIES.items():
        pts = sorted(pts)
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax.plot(x, y, marker="o", linewidth=2.3, markersize=6, color=colors[name], label=name)
        last_x, last_y = x[-1], y[-1]
        ax.annotate(
            f"r{last_x}: {last_y:.3f}",
            xy=(last_x, last_y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=colors[name],
        )

    ax.set_title("Incremental s(G) greedy domains")
    ax.set_xlabel("greedy round")
    ax.set_ylabel("adversary value")
    ax.set_xlim(0.5, 16.8)
    ax.set_ylim(2.0, 7.0)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()

    svg = OUTDIR / "triangle_greedy_round_evolution.svg"
    png = OUTDIR / "triangle_greedy_round_evolution.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    print(svg)
    print(png)


def plot_benchmark_comparison():
    fig, ax = plt.subplots(figsize=(9.3, 5.8))
    styles = {
        "best obtained": ("#111827", "o", "-", 2.8),
        "complete bipartite": ("#0891b2", "X", "-", 2.4),
        "1-surgery family": ("#16a34a", "h", "-", 2.3),
        "2-surgery cumulative": ("#84cc16", "*", "None", 0.0),
        "preserved bipartite": ("#0d9488", "p", "-", 2.2),
        "heuristic round 2": ("#f97316", "s", "--", 2.0),
        "heuristic round 3": ("#16a34a", "D", "--", 2.0),
        "heuristic round 6": ("#dc2626", "^", "--", 2.0),
        "heuristic round 16": ("#7c3aed", "*", "None", 0.0),
    }

    for name, pts in BENCHMARK_SERIES.items():
        color, marker, linestyle, linewidth = styles[name]
        pts = sorted(pts)
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax.plot(
            x,
            y,
            marker=marker,
            markersize=11 if marker == "*" else 8,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=name,
        )

    checkpoint_labels = {
        (6, 5.219912): "r16",
        (7, 6.231963): "r6",
        (8, 6.911788): "r3",
    }
    for (n, value), label in checkpoint_labels.items():
        ax.annotate(
            label,
            xy=(n, value),
            xytext=(5, 7),
            textcoords="offset points",
            fontsize=9,
            color="#111827",
        )

    ax.set_title("Complete-bipartite benchmark vs greedy heuristic checkpoints")
    ax.set_xlabel("number of vertices n")
    ax.set_ylabel("adversary value")
    ax.set_xticks([4, 5, 6, 7, 8])
    ax.set_xlim(3.85, 8.15)
    ax.set_ylim(2.45, 7.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    svg = OUTDIR / "triangle_benchmark_comparison.svg"
    png = OUTDIR / "triangle_benchmark_comparison.png"
    fig.savefig(svg)
    fig.savefig(png, dpi=180)
    print(svg)
    print(png)


def main():
    plot_by_n()
    plot_by_round()
    plot_benchmark_comparison()


if __name__ == "__main__":
    main()
