import itertools
import math
from typing import Dict, Tuple, List, Optional

import numpy as np
import scipy.sparse as sp
import cvxpy as cp


# ============================
# Basic tuple/index utilities
# ============================

def tuples_of_len(m: int, s: int) -> List[Tuple[int, ...]]:
    """All tuples in [m]^s, using 0-based indexing."""
    if s == 0:
        return [()]
    return list(itertools.product(range(m), repeat=s))


def monomial_features(z: np.ndarray, tuple_arr: np.ndarray) -> np.ndarray:
    """
    Given z in R^m and tuple_arr of shape (K, ell),
    return feature vector phi where
        phi[r] = prod_j z[tuple_arr[r, j]].
    Works also when ell = 0.
    """
    if tuple_arr.shape[1] == 0:
        return np.ones(tuple_arr.shape[0], dtype=float)
    return np.prod(z[tuple_arr], axis=1)


# ====================================================
# Sparse symmetric matrices for equality operator A_s
# ====================================================

def _add_sym_entry(rows, cols, data, i: int, j: int, coeff: float) -> None:
    """
    Add coeff * E_{ij} to a symmetric linear form <A, X>.
    For off-diagonal entries, use coeff/2 at (i,j) and (j,i)
    so that <A, X> = coeff * X_{ij} for symmetric X.
    """
    if i == j:
        rows.append(i)
        cols.append(j)
        data.append(coeff)
    else:
        rows.extend([i, j])
        cols.extend([j, i])
        half = 0.5 * coeff
        data.extend([half, half])


def make_entry_diff_matrix(
    N: int,
    p: int,
    q: int,
    r: int,
    s: int,
) -> sp.csr_matrix:
    """
    Return sparse symmetric A such that for symmetric X:
        <A, X> = X[p,q] - X[r,s].
    """
    rows, cols, data = [], [], []
    _add_sym_entry(rows, cols, data, p, q, +1.0)
    _add_sym_entry(rows, cols, data, r, s, -1.0)
    return sp.csr_matrix((data, (rows, cols)), shape=(N, N))


# =================================================
# Structure builder for the reduced-size paper SDP
# =================================================

def build_paper_structure(n: int, degree: int, split: Optional[int] = None):
    """
    Build all static data for the reduced-size SDP from the paper.

    Parameters
    ----------
    n : int
        Input dimension of f : D -> {-1,1}.
        The ambient tensor dimension is m = n + 1.
    degree : int
        Tensor degree d (paper notation).
    split : int or None
        Split parameter s in [0, degree]. If None, use floor(degree/2).

    Returns
    -------
    structure : dict
        Contains tuple lists, tuple arrays, positions, sparse A_s matrices, etc.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")

    if split is None:
        split = degree // 2
    if not (0 <= split <= degree):
        raise ValueError("split must satisfy 0 <= split <= degree")

    m = n + 1
    s = split
    r = degree - s

    a_tuples = tuples_of_len(m, s)
    b_tuples = tuples_of_len(m, r)

    a_pos = {alpha: k for k, alpha in enumerate(a_tuples)}
    b_pos = {beta: k for k, beta in enumerate(b_tuples)}

    a_arr = np.array(a_tuples, dtype=int)
    b_arr = np.array(b_tuples, dtype=int)

    Na = len(a_tuples)
    Nb = len(b_tuples)
    N = Na + Nb

    A_sparse = []

    # --------------------------------------------------------
    # U-block equalities: condition (8) from the paper
    # <u_{j i}, u_{k i}> = <u_{j i'}, u_{k i'}>
    # for ell=1,...,s-1; i,i' in [m]^ell; j,k in [m]^(s-ell)
    # --------------------------------------------------------
    for ell in range(1, s):
        suffixes = tuples_of_len(m, ell)          # i, i'
        prefixes = tuples_of_len(m, s - ell)      # j, k

        ref_suffix = suffixes[0]

        for cur_suffix in suffixes[1:]:
            for j_idx, j in enumerate(prefixes):
                for k in prefixes[j_idx:]:
                    ref_row = a_pos[j + ref_suffix]
                    ref_col = a_pos[k + ref_suffix]
                    cur_row = a_pos[j + cur_suffix]
                    cur_col = a_pos[k + cur_suffix]

                    A_sparse.append(
                        make_entry_diff_matrix(N, ref_row, ref_col, cur_row, cur_col)
                    )

    # --------------------------------------------------------
    # V-block equalities: condition (7) from the paper
    # <v_{i j}, v_{i k}> = <v_{i' j}, v_{i' k}>
    # for ell=1,...,r-1; i,i' in [m]^ell; j,k in [m]^(r-ell)
    # --------------------------------------------------------
    for ell in range(1, r):
        prefixes = tuples_of_len(m, ell)          # i, i'
        suffixes = tuples_of_len(m, r - ell)      # j, k

        ref_prefix = prefixes[0]

        for cur_prefix in prefixes[1:]:
            for j_idx, j in enumerate(suffixes):
                for k in suffixes[j_idx:]:
                    ref_row = Na + b_pos[ref_prefix + j]
                    ref_col = Na + b_pos[ref_prefix + k]
                    cur_row = Na + b_pos[cur_prefix + j]
                    cur_col = Na + b_pos[cur_prefix + k]

                    A_sparse.append(
                        make_entry_diff_matrix(N, ref_row, ref_col, cur_row, cur_col)
                    )

    A_consts = [cp.Constant(A) for A in A_sparse]

    return {
        "n": n,
        "m": m,
        "degree": degree,
        "split": s,
        "r": r,
        "Na": Na,
        "Nb": Nb,
        "N": N,
        "a_tuples": a_tuples,
        "b_tuples": b_tuples,
        "a_arr": a_arr,
        "b_arr": b_arr,
        "A_sparse": A_sparse,
        "A_consts": A_consts,
    }


# ============================================================
# Static matrix builders
# ============================================================

def C_s_of_M(M: cp.Expression, Na: int, Nb: int) -> cp.Expression:
    """
    Build C_s(T) = 1/2 [[0, M], [M^T, 0]]
    where M[a,b] = T_{a b}.
    """
    zero_aa = cp.Constant(sp.csr_matrix((Na, Na)))
    zero_bb = cp.Constant(sp.csr_matrix((Nb, Nb)))
    return cp.bmat([
        [zero_aa, 0.5 * M],
        [0.5 * M.T, zero_bb],
    ])


def A_star_of_y(y: Optional[cp.Variable], A_consts: List[cp.Constant], N: int) -> cp.Expression:
    """
    Build A_s^*(y) = sum_k y_k A_k.
    """
    if y is None or len(A_consts) == 0:
        return cp.Constant(sp.csr_matrix((N, N)))

    expr = cp.Constant(sp.csr_matrix((N, N)))
    for k, A in enumerate(A_consts):
        expr = expr + y[k] * A
    return expr


# ============================================================
# Precompute approximation features
# ============================================================

def precompute_split_features(
    n: int,
    structure,
    inputs,
):
    """
    For each x in inputs, build:
        alpha_x[a] = prod_{i in a} z_i
        beta_x[b]  = prod_{i in b} z_i
    where z = (x, 1) in R^(n+1).
    """
    a_arr = structure["a_arr"]
    b_arr = structure["b_arr"]

    features = {}
    for x in inputs:
        x = tuple(int(v) for v in x)
        z = np.concatenate([np.array(x, dtype=float), np.array([1.0])])

        alpha = monomial_features(z, a_arr)
        beta = monomial_features(z, b_arr)
        features[x] = (alpha, beta)

    return features


# ===
# SDP
# ===

def paper_cbdeg_problem(
    n: int,
    degree: int,
    f_values: Dict[Tuple[int, ...], float],
    split: Optional[int] = None,
    structure=None,
):
    """
    Build the reduced-size paper SDP for fixed degree d.

    This implements the query-complexity reformulation based on:
      - the reduced tensor norm SDP (Theorem 3.4 / eq. (12))
      - the cb-degree feasibility SDP (eq. (18))

    Variables:
      eps_alg >= 0
      M in R^(m^s x m^(d-s))       representing tensor coefficients T_{a,b}
      lambda in R^(m^s + m^(d-s))
      y in R^(# equality constraints)

    Constraints:
      | sum_{a,b} M[a,b] alpha_x[a] beta_x[b] - f(x) | <= 2 eps_alg
      sum(lambda) <= 1
      Diag(lambda) + A_s^*(y) - C_s(M) >= 0
    """
    if structure is None:
        structure = build_paper_structure(n=n, degree=degree, split=split)
    else:
        if structure["n"] != n or structure["degree"] != degree:
            raise ValueError("Provided structure does not match (n, degree).")

    Na = structure["Na"]
    Nb = structure["Nb"]
    N = structure["N"]
    A_consts = structure["A_consts"]

    eps_alg = cp.Variable(nonneg=True)
    M = cp.Variable((Na, Nb))         # tensor flattened as split matrix
    lam = cp.Variable(N)              # free variable
    y = cp.Variable(len(A_consts)) if len(A_consts) > 0 else None

    constraints = []

    # Approximation constraints on the supplied domain only
    feature_cache = precompute_split_features(
        n=n,
        structure=structure,
        inputs=f_values.keys(),
    )

    for x, fx in f_values.items():
        x = tuple(int(v) for v in x)
        fx = float(fx)

        alpha, beta = feature_cache[x]
        approx = alpha @ M @ beta

        constraints.append(approx - fx <= 2 * eps_alg)
        constraints.append(fx - approx <= 2 * eps_alg)

    # <e, lambda> <= 1
    constraints.append(cp.sum(lam) <= 1)

    # Diag(lambda) + A_s^*(y) - C_s(M) >= 0
    Csm = C_s_of_M(M, Na=Na, Nb=Nb)
    Astar = A_star_of_y(y, A_consts=A_consts, N=N)
    constraints.append(cp.diag(lam) + Astar - Csm >> 0)

    prob = cp.Problem(cp.Minimize(eps_alg), constraints)

    return prob, {
        "eps_alg": eps_alg,
        "M": M,
        "lambda": lam,
        "y": y,
        "structure": structure,
        "feature_cache": feature_cache,
    }


# ========================
# Query complexity search
# ========================

def query_upper_bound_from_degree(degree: int) -> int:
    """
    Paper / ABP relation:
        Q_eps(f) = ceil(cb-deg_eps(f) / 2)
    """
    return math.ceil(degree / 2)


def tensor2(
    n: int,
    f_values: Dict[Tuple[int, ...], float],
    eps_target: float = 1/3,
    max_degree: int = 6,
    solver=cp.SCS,
    verbose: bool = False,
    solver_opts: Optional[dict] = None,
):
    """
    Scan degrees d = 1,...,max_degree and solve the paper-style SDP
    until eps_alg <= eps_target.

    Returns a summary dict with per-degree results.
    """
    if solver_opts is None:
        solver_opts = {}

    results = []

    for degree in range(1, max_degree + 1):
        split = degree // 2
        structure = build_paper_structure(n=n, degree=degree, split=split)
        prob, vars_ = paper_cbdeg_problem(
            n=n,
            degree=degree,
            f_values=f_values,
            structure=structure,
        )

        # Default SCS options
        local_opts = dict(solver_opts)
        if solver == cp.SCS:
            local_opts.setdefault("eps", 1e-4)
            local_opts.setdefault("max_iters", 10000)
            local_opts.setdefault("normalize", True)
            local_opts.setdefault("acceleration_lookback", 10)

        prob.solve(solver=solver, verbose=verbose, **local_opts)

        eps_val = vars_["eps_alg"].value
        eps_val = None if eps_val is None else float(eps_val)

        sm = prob.size_metrics
        row = {
            "degree": degree,
            "split": split,
            "query_upper_bound": query_upper_bound_from_degree(degree),
            "status": prob.status,
            "eps_alg": eps_val,
            "objective": None if prob.value is None else float(prob.value),
            "num_scalar_variables": sm.num_scalar_variables,
            "num_scalar_eq": sm.num_scalar_eq_constr,
            "num_scalar_ineq": sm.num_scalar_leq_constr,
            "largest_psd_block": structure["N"],
            "A_constraints": len(structure["A_sparse"]),
        }
        results.append(row)

        if eps_val is not None and eps_val <= eps_target:
            return {
                "cb_degree": degree,
                "query_complexity_upper_bound": query_upper_bound_from_degree(degree),
                "results": results,
            }

    return {
        "cb_degree": None,
        "query_complexity_upper_bound": None,
        "results": results,
    }
