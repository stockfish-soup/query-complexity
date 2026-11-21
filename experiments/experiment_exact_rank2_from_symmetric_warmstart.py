"""Nonsymmetric rank-2 BM bisection warm-started from symmetric rank-2 orbit solution."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import query_complexity as qc


FIELDS = [
    "n",
    "k",
    "alpha",
    "lo",
    "hi",
    "symmetric_feasible",
    "symmetric_eq",
    "symmetric_block",
    "symmetric_cap",
    "expanded_ranks",
    "nonsymmetric_feasible",
    "nonsymmetric_eq",
    "nonsymmetric_cap",
    "nonsymmetric_nfev",
]


def append(path: Path, row: dict) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def test_alpha(
    n: int,
    k: int,
    alpha: float,
    *,
    pattern: tuple[int, ...],
    epsilon: float,
    feasibility_tol: float,
    sym_max_nfev: int,
    nonsym_max_nfev: int,
    restarts: int,
    solver: str,
):
    layers = qc.EXACT_layers(n, k)
    inputs = qc.all_binary_inputs(n)
    outputs = [layers[sum(x)] for x in inputs]
    base = qc.solve_symmetric_trace_minimized_dual(
        n,
        layers,
        alpha=1.0,
        epsilon=epsilon,
        solver=solver,
    )
    adversary = base.objective_value
    convex = qc.solve_symmetric_trace_minimized_dual(
        n,
        layers,
        alpha=alpha,
        adversary_value=adversary,
        epsilon=epsilon,
        solver=solver,
    )
    sym = qc.solve_symmetric_fixed_rank_dual_feasibility(
        n,
        layers,
        block_rank=max(pattern),
        block_ranks=pattern,
        alpha=alpha,
        adversary_value=adversary,
        epsilon=epsilon,
        initial_solution=convex,
        restarts=restarts,
        random_seed=230_000 + 1000 * k + round(10000 * alpha),
        max_nfev=sym_max_nfev,
        feasibility_tol=feasibility_tol,
        solver=solver,
    )
    if not sym.feasible:
        return sym, None, None
    orbit_view = qc.OrbitSolution(
        n=sym.n,
        layer_values=sym.layer_values,
        allowed_layers=sym.allowed_layers,
        objective_value=sym.adversary_value,
        trace_value=sym.trace_value,
        cap=sym.cap,
        status=sym.status,
        orbit_values=sym.orbit_values,
        block_values=sym.block_values,
        block_labels=sym.block_labels,
        input_costs_by_layer=sym.input_costs_by_layer,
    )
    expanded = qc.expand_orbit_solution(orbit_view, psd_tol=1e-7)
    nonsym = qc.solve_fixed_rank_dual_feasibility(
        inputs,
        outputs,
        rank=2,
        alpha=alpha,
        adversary_value=adversary,
        epsilon=epsilon,
        initial_solution=expanded,
        restarts=0,
        random_seed=240_000 + 1000 * k + round(10000 * alpha),
        max_nfev=nonsym_max_nfev,
        feasibility_tol=feasibility_tol,
    )
    return sym, expanded, nonsym


def parse_pattern(spec: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in spec.split(";"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=7)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--pattern", default="2;0;0;0")
    parser.add_argument("--alpha-lo", type=float, default=1.2)
    parser.add_argument("--alpha-hi", type=float, default=1.4)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--sym-max-nfev", type=int, default=1200)
    parser.add_argument("--nonsym-max-nfev", type=int, default=50)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--solver", default="MOSEK")
    parser.add_argument("--csv", type=Path, default=Path("exact_rank2_from_symmetric_warmstart.csv"))
    args = parser.parse_args()

    if args.csv.exists():
        args.csv.unlink()
    pattern = parse_pattern(args.pattern)
    lo = args.alpha_lo
    hi = args.alpha_hi
    for step in range(args.iters + 1):
        alpha = hi if step == 0 else 0.5 * (lo + hi)
        print(f"EXACT({args.n},{args.k}) alpha={alpha:.10g}: symmetric pattern {args.pattern} -> nonsymmetric rank 2", flush=True)
        sym, expanded, nonsym = test_alpha(
            args.n,
            args.k,
            alpha,
            pattern=pattern,
            epsilon=args.epsilon,
            feasibility_tol=args.feasibility_tol,
            sym_max_nfev=args.sym_max_nfev,
            nonsym_max_nfev=args.nonsym_max_nfev,
            restarts=args.restarts,
            solver=args.solver,
        )
        feasible = bool(nonsym and nonsym.feasible)
        append(
            args.csv,
            {
                "n": args.n,
                "k": args.k,
                "alpha": alpha,
                "lo": lo,
                "hi": hi,
                "symmetric_feasible": sym.feasible,
                "symmetric_eq": sym.max_equality_residual,
                "symmetric_block": sym.max_block_residual,
                "symmetric_cap": sym.max_cap_violation,
                "expanded_ranks": ";".join(map(str, expanded.witness_dims)) if expanded else "",
                "nonsymmetric_feasible": feasible,
                "nonsymmetric_eq": nonsym.max_equality_residual if nonsym else "",
                "nonsymmetric_cap": nonsym.max_cap_violation if nonsym else "",
                "nonsymmetric_nfev": nonsym.total_nfev if nonsym else "",
            },
        )
        print(
            f"  symmetric={sym.feasible} eq={sym.max_equality_residual:.3e} block={sym.max_block_residual:.3e}; "
            f"expanded_ranks={expanded.witness_dims if expanded else None}; "
            f"nonsymmetric={feasible}",
            flush=True,
        )
        if step == 0:
            if not feasible:
                print(f"alpha_hi={hi} was not feasible; increase --alpha-hi", flush=True)
                break
            continue
        if feasible:
            hi = alpha
        else:
            lo = alpha
        print(f"  bracket ({lo:.10g}, {hi:.10g}]", flush=True)


if __name__ == "__main__":
    main()
