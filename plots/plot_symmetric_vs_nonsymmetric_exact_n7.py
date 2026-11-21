"""Compare symmetric rank-2 pattern search with nonsymmetric BM for EXACT(7,k)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NON_SYM = Path("bm_exact_n7_adaptive_summary.csv")
SYM = Path("symmetric_exact_n7_pattern2000_alpha_bisect.csv")
WARMSTART = Path("exact_n7_k2_rank2_from_symmetric_warmstart.csv")


def nonsym_steps():
    rows = list(csv.DictReader(NON_SYM.open(newline="", encoding="utf-8")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["k"])].append(row)
    out = {}
    for k, case in grouped.items():
        case = sorted(case, key=lambda row: float(row["alpha"]))
        xs = [float(row["alpha"]) for row in case]
        ys = [int(row["min_rank"]) for row in case]
        out[k] = (xs, ys)
    return out


def symmetric_rank2_thresholds():
    rows = list(csv.DictReader(SYM.open(newline="", encoding="utf-8")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["k"])].append(row)
    out = {}
    for k, case in grouped.items():
        feas = [float(row["alpha"]) for row in case if row["feasible"] == "True"]
        infeas = [float(row["alpha"]) for row in case if row["feasible"] == "False"]
        hi = min(feas) if feas else None
        lo = max([alpha for alpha in infeas if hi is not None and alpha < hi], default=None)
        out[k] = (lo, hi)
    return out


def warmstart_rank2_thresholds():
    if not WARMSTART.exists():
        return {}
    rows = list(csv.DictReader(WARMSTART.open(newline="", encoding="utf-8")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["k"])].append(row)
    out = {}
    for k, case in grouped.items():
        feas = [
            float(row["alpha"])
            for row in case
            if row["nonsymmetric_feasible"] == "True"
        ]
        infeas = [
            float(row["alpha"])
            for row in case
            if row["nonsymmetric_feasible"] == "False"
        ]
        hi = min(feas) if feas else None
        lo = max([alpha for alpha in infeas if hi is not None and alpha < hi], default=None)
        out[k] = (lo, hi)
    return out


def main() -> None:
    steps = nonsym_steps()
    sym = symmetric_rank2_thresholds()
    warm = warmstart_rank2_thresholds()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=False)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, k in zip(axes, sorted(steps)):
        xs, ys = steps[k]
        ax.step(xs, ys, where="post", marker="o", color=colors[0], label="nonsymmetric min rank")
        lo, hi = sym.get(k, (None, None))
        if hi is not None:
            ax.axvline(hi, color=colors[1], linestyle="--", linewidth=2, label=r"symmetric $2;0;0;0$ feasible")
            if lo is not None:
                ax.axvspan(lo, hi, color=colors[1], alpha=0.16, label="symmetric bracket")
        wlo, whi = warm.get(k, (None, None))
        if whi is not None:
            ax.axvline(whi, color=colors[2], linestyle="-.", linewidth=2, label="nonsym warm-start rank 2")
            if wlo is not None:
                ax.axvspan(wlo, whi, color=colors[2], alpha=0.12, label="warm-start bracket")
        ax.axhline(2, color="0.35", linestyle=":", linewidth=1.4)
        ax.set_title(rf"$\mathrm{{EXACT}}(7,{k})$")
        ax.set_xlabel(r"$\alpha$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("rank")
    fig.suptitle(r"EXACT(7,k): nonsymmetric BM rank curve vs symmetric orbit rank-2 pattern")
    fig.tight_layout()
    out = Path("exact_n7_symmetric_vs_nonsymmetric_rank2.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    width = 0.34
    ks = sorted(steps)
    nonsym_rank2 = []
    sym_rank2 = []
    warm_rank2 = []
    for k in ks:
        xs, ys = steps[k]
        rank2 = min((x for x, y in zip(xs, ys) if y <= 2), default=float("nan"))
        nonsym_rank2.append(rank2)
        sym_rank2.append(sym.get(k, (None, None))[1] or float("nan"))
        warm_rank2.append(warm.get(k, (None, None))[1] or float("nan"))
    pos = list(range(len(ks)))
    width = 0.25
    ax.bar([p - width for p in pos], nonsym_rank2, width, label="old nonsym adaptive rank ≤2")
    ax.bar(pos, sym_rank2, width, label=r"symmetric $2;0;0;0$")
    ax.bar([p + width for p in pos], warm_rank2, width, label="nonsym from symmetric warm start")
    ax.set_xticks(pos, [f"k={k}" for k in ks])
    ax.set_ylabel(r"$\alpha$ threshold")
    ax.set_title("Rank-2 alpha thresholds for EXACT(7,k)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = Path("exact_n7_rank2_threshold_comparison.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
