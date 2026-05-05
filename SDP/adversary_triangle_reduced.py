"""
Specht-block-diagonalised primal adversary SDP for triangle finding.

This is the "second reduction" from Section 2.3 of Gribling-Polak applied
to the triangle adversary. Building on triangle_adversary.py (which
gives the orbit reduction of variables), here we additionally
block-diagonalise the PSD matrix

    M = diag(beta) - Gamma o Delta_{e0}

using the symmetry group  Stab(e_0) = S_2 x S_{n-2}  (the subgroup of
S_n fixing the canonical edge e_0 = {0, 1} as a set).

The plan
--------
1. Variables (beta, gamma) remain S_n-invariant -- one beta per graph
   iso-class and one gamma per pair iso-class (from triangle_adversary).
2. The matrix M is Stab(e_0)-invariant. Decompose
       C^D = bigoplus_{rho} (S^rho)^{oplus M_rho}
   under Stab(e_0), where rho = (lambda_1, lambda_2) ranges over irreps
   of S_2 x S_{n-2}.
3. For each rho, compute the Young symmetrizer
       c_rho = sum_{p in R_t} sum_{q in C_t} sgn(q) * pq
   in C[Stab(e_0)] for the canonical tableau t of shape rho. Apply
   c_rho to C^D, take an orthonormal basis U_rho of its image. Then
       dim(im c_rho) = M_rho
   and {U_rho} is a representative set in the sense of the paper.
4. Replace  M >= 0  by  U_rho^T M U_rho >= 0  for each rho.

For n = 4 this turns one 64 x 64 PSD constraint into four blocks of
sizes 28, 12, 12, 12 (sum = 64, matches dim C^D). For n = 5 the
biggest block is much smaller than 1024.
"""

import itertools
import datetime
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import cvxpy as cp
import math

from adversary_triangle import (
    precompute_perm_table,
    compute_orbits,
)


# ---------------------------------------------------------------------------
# 1. Partition / tableau / Young symmetrizer machinery
# ---------------------------------------------------------------------------

def partitions(n):
    """All partitions of n as tuples in lex-decreasing order. p(0) = ().
    """
    if n == 0:
        yield ()
        return
    def _gen(remaining, max_part):
        if remaining == 0:
            yield ()
            return
        for part in range(min(remaining, max_part), 0, -1):
            for tail in _gen(remaining - part, part):
                yield (part,) + tail
    yield from _gen(n, n)


def standard_tableau(lam, start=0):
    """Fill the shape `lam` row-by-row with consecutive integers from `start`."""
    rows, cnt = [], start
    for r in lam:
        rows.append(list(range(cnt, cnt + r)))
        cnt += r
    return rows


def perms_preserving_blocks(blocks, n):
    """Generate all sigma in S_n that permute each block (a list of integers
    in [n]) to itself, fixing positions outside the union of the blocks."""
    from itertools import permutations as P, product
    blocks = [list(b) for b in blocks if len(b) > 0]
    perms_per_block = [list(P(b)) for b in blocks]
    if not blocks:
        yield tuple(range(n))
        return
    for combo in product(*perms_per_block):
        sigma = list(range(n))
        for orig, new in zip(blocks, combo):
            for p_old, p_new in zip(orig, new):
                sigma[p_old] = p_new
        yield tuple(sigma)


def row_stab(tab, n):
    """Permutations preserving each row of `tab` as a set."""
    return list(perms_preserving_blocks(tab, n))


def col_stab(tab, n):
    """Permutations preserving each column of `tab` as a set."""
    if not tab:
        return [tuple(range(n))]
    max_len = max(len(r) for r in tab)
    cols = [[r[c] for r in tab if len(r) > c] for c in range(max_len)]
    return list(perms_preserving_blocks(cols, n))


def perm_sign(sigma):
    """Sign of a permutation given as a tuple of length n."""
    n = len(sigma)
    sign = 1
    seen = [False] * n
    for i in range(n):
        if seen[i]:
            continue
        j, cyc = i, 0
        while not seen[j]:
            seen[j] = True
            j = sigma[j]
            cyc += 1
        if cyc % 2 == 0:
            sign = -sign
    return sign


def young_symmetrizer_e0(lam1, lam2, n):
    """Young symmetrizer  c = sum_{p in R} sum_{q in C} sgn(q) * pq
    for the irrep rho = (lam1, lam2) of  S_2 x S_{n-2}, where:
        S_2     acts on the endpoints {0, 1} of the canonical edge,
        S_{n-2} acts on {2, ..., n-1}.

    The canonical tableau is  lam1  filled with {0, 1}  ROW-WISE, then
    lam2 filled with {2, ..., n-1} row-wise.

    Returns a list of (sigma, coeff) pairs (with coeff != 0).
    """
    tab1 = standard_tableau(lam1, start=0)
    tab2 = standard_tableau(lam2, start=2)
    R1, R2 = row_stab(tab1, n), row_stab(tab2, n)
    C1, C2 = col_stab(tab1, n), col_stab(tab2, n)

    # Compose r1 (acts on {0,1}) with r2 (acts on {2,...,n-1}); they
    # commute since they touch disjoint indices.
    R = []
    for r1 in R1:
        for r2 in R2:
            sigma = tuple(r1[i] if i < 2 else r2[i] for i in range(n))
            R.append(sigma)
    C = []
    for c1 in C1:
        for c2 in C2:
            sigma = tuple(c1[i] if i < 2 else c2[i] for i in range(n))
            C.append(sigma)

    coeffs = defaultdict(int)
    for p in R:
        for q in C:
            sigma = tuple(p[q[i]] for i in range(n))
            coeffs[sigma] += perm_sign(q)
    return [(s, c) for s, c in coeffs.items() if c != 0]


# ---------------------------------------------------------------------------
# 2. Apply the Young symmetrizer to C^D
# ---------------------------------------------------------------------------

def apply_young_symmetrizer(young_sym, perm_table, perms_Sn_list):
    """Build the N x N matrix C with  C v = (c_rho . v)  for v in C^D.

    P_sigma is the permutation matrix on C^D acting as
        (P_sigma v)[x] = v[sigma^{-1} . x],
    equivalently (P_sigma e_y) = e_{sigma . y}, so
        C[sigma . y, y] += sign(sigma in c_rho).
    """
    perm_to_idx = {sigma: k for k, sigma in enumerate(perms_Sn_list)}
    N = perm_table.shape[1]
    C = np.zeros((N, N), dtype=np.float64)
    ys = np.arange(N)
    for sigma, coeff in young_sym:
        k = perm_to_idx[sigma]
        xs = perm_table[k]            # (N,)  xs[y] = sigma . y
        np.add.at(C, (xs, ys), coeff)
    return C


def column_basis(C, tol=1e-10):
    """Orthonormal basis of column-space of C (via SVD)."""
    if not np.any(C):
        return np.zeros((C.shape[0], 0))
    U, s, _ = np.linalg.svd(C, full_matrices=False)
    threshold = tol * (s[0] if len(s) > 0 else 1.0)
    rank = int((s > threshold).sum())
    return U[:, :rank]


def compute_specht_blocks(n, verbose=True):
    """For each irrep rho = (lam1, lam2) of S_2 x S_{n-2}, return the
    representative matrix U_rho (N x M_rho).
    """
    perm_table, m, N = precompute_perm_table(n)
    perms_Sn_list = list(itertools.permutations(range(n)))

    blocks = {}
    if verbose:
        print(f"Computing Specht blocks for n={n}, "
              f"|Stab(e0)|=2 * {(n-2)}!={2 * math.factorial(n-2)}, N={N}")
    total_size = 0
    for lam1 in partitions(2):
        for lam2 in partitions(n - 2):
            t0 = datetime.datetime.now()
            young_sym = young_symmetrizer_e0(lam1, lam2, n)
            C = apply_young_symmetrizer(young_sym, perm_table, perms_Sn_list)
            U = column_basis(C)
            blocks[(lam1, lam2)] = U
            total_size += U.shape[1]
            if verbose:
                print(f"  rho=({lam1}, {lam2}):  M_rho = {U.shape[1]:4d}   "
                      f"(c_rho has {len(young_sym)} terms; "
                      f"took {datetime.datetime.now() - t0})")
    if verbose:
        print(f"  total of all block sizes: {total_size}  "
              f"(should equal N = {N})")
    return blocks, perm_table, perms_Sn_list


# ---------------------------------------------------------------------------
# 3. The Specht-blocked SDP
# ---------------------------------------------------------------------------

def adversary_blocked_triangle(
        n, solver=cp.MOSEK, verbose=False, orb=None, blocks=None):
    """Solve the orbit-reduced primal adversary SDP for triangle finding,
    with the PSD constraint block-diagonalised via Stab(e_0)-Specht
    decomposition.
    """
    # ---- 1. Variables on S_n-orbits ------------------------------------
    if orb is None:
        orb = compute_orbits(n, verbose=verbose)
    f               = orb["f"]
    N               = orb["N"]
    graph_reps      = orb["graph_reps"]
    graph_size      = orb["graph_orbit_size"]
    graph_orbit_idx = orb["graph_orbit_idx"]
    pair_reps       = orb["pair_reps"]
    pair_size       = orb["pair_orbit_size"]
    pair_canon_idx  = orb["pair_canon_idx"]
    e0              = orb["canonical_edge"]

    graph_f  = np.array([int(f[r]) for r in graph_reps], dtype=np.int8)
    pair_f_x = np.array([int(f[r // N]) for r in pair_reps], dtype=np.int8)
    pair_f_y = np.array([int(f[r %  N]) for r in pair_reps], dtype=np.int8)

    n_g, n_p = len(graph_reps), len(pair_reps)
    beta  = cp.Variable(n_g, nonneg=True, name="beta")
    gamma = cp.Variable(n_p, name="gamma")

    constraints = []

    same_f_mask = (pair_f_x == pair_f_y)
    if same_f_mask.any():
        constraints.append(gamma[np.where(same_f_mask)[0]] == 0)

    zero_idx = np.where(graph_f == 0)[0]
    one_idx  = np.where(graph_f == 1)[0]
    constraints.append(graph_size[zero_idx].astype(float) @ beta[zero_idx]
                       == 0.5)
    constraints.append(graph_size[one_idx ].astype(float) @ beta[one_idx ]
                       == 0.5)

    # ---- 2. Specht blocks ----------------------------------------------
    if blocks is None:
        blocks, _, _ = compute_specht_blocks(n, verbose=verbose)

    # ---- 3. Per-orbit sparse matrices for the diagonal-beta and
    #         off-diagonal-gamma contributions to M = diag(beta) - G o D
    #-------------------------------------------------------------------
    D_a_list = []
    for a in range(n_g):
        rows = np.where(graph_orbit_idx == a)[0]
        D_a = sp.coo_matrix(
            (np.ones(len(rows)), (rows, rows)), shape=(N, N)).tocsr()
        D_a_list.append(D_a)

    x_bit = ((np.arange(N, dtype=np.int64) >> e0) & 1).astype(np.int8)
    flat_idx = pair_canon_idx.ravel()
    ij = np.arange(N * N, dtype=np.int64)
    ii = (ij // N).astype(np.int64)
    jj = (ij %  N).astype(np.int64)
    delta_keep = (x_bit[ii] != x_bit[jj])
    keep_orbit = flat_idx[delta_keep]
    keep_i = ii[delta_keep]
    keep_j = jj[delta_keep]
    order = np.argsort(keep_orbit, kind="stable")
    keep_orbit_s = keep_orbit[order]
    keep_i_s = keep_i[order]
    keep_j_s = keep_j[order]
    boundaries = np.concatenate((
        [0],
        np.where(np.diff(keep_orbit_s) != 0)[0] + 1,
        [len(keep_orbit_s)],
    ))

    A_O_list = {}
    for k in range(len(boundaries) - 1):
        s, e = boundaries[k], boundaries[k + 1]
        if s == e:
            continue
        O = int(keep_orbit_s[s])
        rows = keep_i_s[s:e]
        cols = keep_j_s[s:e]
        A_O = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(N, N)).tocsr()
        A_O_list[O] = A_O

    # ---- 4. For each rho, project the variable contributions to M
    #         to the U_rho block.
    #-------------------------------------------------------------------
    if verbose:
        print("\nProjecting to Specht blocks...")
    block_count = 0
    for rho, U in blocks.items():
        if U.shape[1] == 0:
            continue
        t0 = datetime.datetime.now()
        block_terms = []
        # (a) diag(beta) part: add  beta_a * (U^T D_a U)
        for a in range(n_g):
            UDU = U.T @ (D_a_list[a] @ U)         # M_rho x M_rho
            if not np.any(UDU):
                continue
            block_terms.append(beta[a] * UDU)
        # (b) -gamma o Delta part:  -gamma_O * (U^T A_O U)
        for O, A_O in A_O_list.items():
            UAU = U.T @ (A_O @ U)
            if not np.any(UAU):
                continue
            block_terms.append(-gamma[O] * UAU)
        if not block_terms:
            continue
        block_expr = sum(block_terms)
        # Symmetrise to clean up numerical noise
        block_expr = (block_expr + block_expr.T) / 2.0
        constraints.append(block_expr >> 0)
        block_count += 1
        if verbose:
            print(f"  rho={rho}: built block of size {U.shape[1]}  "
                  f"({datetime.datetime.now() - t0})")

    # ---- 5. Objective and solve ---------------------------------------
    objective = cp.Maximize(pair_size.astype(float) @ gamma)
    problem = cp.Problem(objective, constraints)
    val = problem.solve(solver=solver, verbose=verbose)

    return {
        "value": val,
        "status": problem.status,
        "n": n,
        "block_sizes": {rho: U.shape[1] for rho, U in blocks.items()},
        "num_blocks": block_count,
        "num_graph_orbits": n_g,
        "num_pair_orbits": n_p,
        "num_pair_orbits_free": int((~same_f_mask).sum()),
        "beta":  np.array(beta.value)  if beta.value  is not None else None,
        "gamma": np.array(gamma.value) if gamma.value is not None else None,
    }
