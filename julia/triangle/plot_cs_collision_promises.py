#!/usr/bin/env python3
import csv
import sys

import matplotlib.pyplot as plt


def load(path):
    with open(path, newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "inputs": [int(row["inputs"]) for row in rows],
        "fraction": [100.0 * float(row["fraction_full"]) for row in rows],
        "value": [float(row["value"]) for row in rows],
    }


def main():
    output = sys.argv[1] if len(sys.argv) >= 2 else "cs_collision_promise_growth.svg"
    series = [
        ("CS(8), standalone ordering", load("cs_collision_promise_n8.csv"), "#7a7a7a", "--"),
        ("CS(8), marginal ordering", load("cs_collision_marginal_n8.csv"), "#087e8b", "-"),
        ("CS(9), marginal ordering", load("cs_collision_marginal_n9.csv"), "#d1495b", "-"),
    ]

    fig, axis = plt.subplots(figsize=(8.2, 5.2))
    for label, data, color, style in series:
        axis.plot(
            data["inputs"],
            data["fraction"],
            marker="o",
            linewidth=2.0,
            linestyle=style,
            color=color,
            label=label,
        )
    for threshold in (90, 95, 99):
        axis.axhline(threshold, color="#bbbbbb", linewidth=0.8, linestyle=":")
        axis.text(0.995, threshold + 0.25, f"{threshold}%", transform=axis.get_yaxis_transform(),
                  ha="right", va="bottom", color="#666666", fontsize=8)
    axis.set_xlabel("explicit inputs in the restricted promise")
    axis.set_ylabel("percentage of the full adversary value")
    axis.set_title("Compact graph-collision promises for CS(n)")
    axis.set_ylim(58, 101)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
