#!/usr/bin/env python3
import csv
import math
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    by_n = {}
    with open(path, newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            if row["status"] not in {"OPTIMAL", "ALMOST_OPTIMAL"}:
                continue
            by_n[int(row["n"])] = row
    return [by_n[n] for n in sorted(by_n)]


def fit_power(ns, values):
    exponent, log_constant = np.polyfit(np.log(ns), np.log(values), 1)
    return math.exp(log_constant), exponent


def main():
    input_path = sys.argv[1] if len(sys.argv) >= 2 else "cs_graph_collision_values.csv"
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "cs_graph_collision_growth.svg"
    rows = load_rows(input_path)
    if len(rows) < 2:
        raise SystemExit("need at least two solved rows")

    ns = np.array([int(row["n"]) for row in rows], dtype=float)
    values = np.array([float(row["value"]) for row in rows])
    constant, exponent = fit_power(ns, values)
    fit = constant * ns**exponent

    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(8.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )
    top.plot(ns, values, "o-", color="#087e8b", linewidth=2.2, label=r"$\mathrm{Adv}^{\pm}(\mathrm{COL}_{CS(n)})$")
    top.plot(ns, np.sqrt(ns), "--", color="#d1495b", linewidth=1.8, label=r"$\sqrt{n}$")
    top.plot(ns, fit, ":", color="#3c3c3c", linewidth=1.8, label=rf"fit ${constant:.3f}n^{{{exponent:.3f}}}$")
    if 8 in ns:
        top.axvline(8, color="#777777", linestyle="--", linewidth=1.0, alpha=0.65)
        top.annotate(
            "square offset 4 enters",
            xy=(8, values[np.where(ns == 8)[0][0]]),
            xytext=(7.75, max(values) * 0.84),
            ha="right",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 1.0},
        )
    top.set_ylabel("adversary value")
    top.grid(alpha=0.25)
    top.legend(frameon=False)
    top.set_title("Graph collision on the square-difference graph CS(n)")

    ratios = values / np.sqrt(ns)
    bottom.plot(ns, ratios, "o-", color="#087e8b", linewidth=2.2)
    bottom.axhline(1.0, color="#d1495b", linestyle="--", linewidth=1.3)
    bottom.set_xlabel("number of host vertices n")
    bottom.set_ylabel(r"$\mathrm{Adv}^{\pm}/\sqrt{n}$")
    bottom.grid(alpha=0.25)
    bottom.set_xticks(ns.astype(int))

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"power_fit_constant={constant:.12g}")
    print(f"power_fit_exponent={exponent:.12g}")
    print(f"ratio_first={ratios[0]:.12g} ratio_last={ratios[-1]:.12g}")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
