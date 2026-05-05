"""
Orbit-reduced primal adversary SDP for the triangle finding problem.

Triangle finding: given x in {0,1}^{C(n,2)} encoding the adjacency matrix
of a graph on n vertices, decide whether the graph contains a triangle.

The function f : {0,1}^{C(n,2)} -> {0,1} is invariant under the natural
S_n action on edges induced by vertex permutations. We exploit this to
reduce variables (gamma, beta are constant on S_n-orbits) and to keep
exactly ONE PSD constraint of the form  diag(beta) - Gamma o Delta_{e0}
>= 0 (any single edge e0 will do, since S_n acts transitively on edges
and we restrict to S_n-invariant solutions).

Compare with the symmetric-function code (Hamming-weight orbits): there
the orbit invariants are (|x|, |y|, |x ^ y|); here the orbit invariant
is the iso-class of the edge-2-coloured graph (x, y), with the swap
(x, y) <-> (y, x) folded in (since Gamma is symmetric).

Implementation notes
--------------------
- Graphs encoded as Python ints (m bits, m = C(n, 2)).
- Canonicalisation is done by enumerating all sigma in S_n: for each
  sigma we precompute the bit-permutation, then min over sigma. Pairs
  are encoded into a single int p*N + q so we can take min lex-style
  with one numpy reduction.
- The PSD matrix is assembled as
       sum_a  beta_a * D_a   -   sum_O  gamma_O * A_O
  where D_a is sparse (just the diagonal indicator of orbit a) and A_O
  is sparse ({0,1} mask of orbit O intersected with Delta_{e0}). This
  is far cheaper to compile than an N x N cp.bmat over scalar entries.

For n in {3, 4, 5} this runs in seconds. For n >= 6 the dominant cost
is the N x N PSD constraint itself (N = 2^15 for n = 6); a further
Specht-block decomposition would be needed.
"""

import itertools
import datetime
from collections import defaultdict
from math import comb

import numpy as np
import scipy.sparse as sp
import cvxpy as cp


# ---------------------------------------------------------------------------
# 1. Triangle finding function
# ---------------------------------------------------------------------------

def edges_of_Kn(n):
    """Edges of K_n in lex order: [(0,1),(0,2),...,(n-2,n-1)]."""
    return list(itertools.combinations(range(n), 2))


def triangle_function(n):
    """f : {0,1}^{C(n,2)} -> {0,1},  f(x)=1 iff x contains a triangle.

    Returns (f_array, edge_list, edge_index) where f_array is a 1D numpy
    array of length 2^m with f_array[x_int] = f(x)."""
    E = edges_of_Kn(n)
    edge_index = {e: i for i, e in enumerate(E)}
    triangles = list(itertools.combinations(range(n), 3))
    m = len(E)
    N = 1 << m

    f = np.zeros(N, dtype=np.int8)
    arange = np.arange(N, dtype=np.int64)
    for (a, b, c) in triangles:
        i_ab = edge_index[(a, b)]
        i_ac = edge_index[(a, c)]
        i_bc = edge_index[(b, c)]
        mask = (1 << i_ab) | (1 << i_ac) | (1 << i_bc)
        f |= ((arange & mask) == mask).astype(np.int8)
    return f, E, edge_index


# ---------------------------------------------------------------------------
# 2. S_n action on edges (as bit-permutations of the m-bit graph integer)
# ---------------------------------------------------------------------------

def edge_perm_from_vertex_perm(sigma, E, edge_index):
    """Edge permutation induced by a vertex permutation sigma of [n]:
    a tuple pi of length m with pi[i] = index of the image of edge i."""
    pi = [0] * len(E)
    for i, (u, v) in enumerate(E):
        a, b = sigma[u], sigma[v]
        if a > b:
            a, b = b, a
        pi[i] = edge_index[(a, b)]
    return tuple(pi)


def precompute_perm_table(n):
    """Return (perm_table, m, N).

    perm_table : int64 array of shape (|S_n|, N) with
                 perm_table[k, x] = (pi_k . x) viewed as an int,
                 where pi_k is the bit-permutation induced by the k-th
                 element of S_n.
    """
    E = edges_of_Kn(n)
    edge_index = {e: i for i, e in enumerate(E)}
    m = len(E)
    N = 1 << m

    perms_edge = [edge_perm_from_vertex_perm(sigma, E, edge_index)
                  for sigma in itertools.permutations(range(n))]
    nperms = len(perms_edge)

    arange = np.arange(N, dtype=np.int64)
    table = np.zeros((nperms, N), dtype=np.int64)
    for k, pi in enumerate(perms_edge):
        result = np.zeros(N, dtype=np.int64)
        for i, pi_i in enumerate(pi):
            bit_i = (arange >> i) & 1
            result |= bit_i << pi_i
        table[k] = result
    return table, m, N


# ---------------------------------------------------------------------------
# 3. Canonical forms via numpy
# ---------------------------------------------------------------------------

def canonicalise_graphs(perm_table):
    """For every x in 0..N-1, return canonical(x) = min_k perm_table[k, x]."""
    return perm_table.min(axis=0)


def canonicalise_pairs(perm_table, N):
    """Return canon_pair: int64 array of shape (N, N) where
    canon_pair[x, y] = min_{k, swap} of (px, py)*N + py'  encoding the
    canonical (x', y') of the orbit under  (sigma, swap)  in
    S_n x Z_2.  Encoded as p*N + q so a numpy min suffices.

    OK as long as N*N fits in int64 (always true here)."""
    P = perm_table.shape[0]
    canon = np.empty((N, N), dtype=np.int64)
    for x in range(N):
        px = perm_table[:, x][:, None]            # (P, 1)
        py = perm_table                           # (P, N)
        code1 = px * N + py
        code2 = py * N + px
        codes = np.minimum(code1, code2)          # (P, N)
        canon[x] = codes.min(axis=0)              # (N,)
    return canon


# ---------------------------------------------------------------------------
# 4. Orbit data
# ---------------------------------------------------------------------------

def compute_orbits(n, verbose=True):
    t0 = datetime.datetime.now()
    perm_table, m, N = precompute_perm_table(n)
    f, _, _ = triangle_function(n)
    if verbose:
        print(f"n={n}: m={m} edges, N=2^m={N} graphs, "
              f"|S_n|={perm_table.shape[0]}.  "
              f"perm_table built in {datetime.datetime.now() - t0}.")

    t0 = datetime.datetime.now()
    graph_canon = canonicalise_graphs(perm_table)            # (N,)
    graph_reps, graph_orbit_size = np.unique(
        graph_canon, return_counts=True)
    if verbose:
        print(f"  {len(graph_reps)} graph orbits "
              f"(took {datetime.datetime.now() - t0}).")

    t0 = datetime.datetime.now()
    pair_canon = canonicalise_pairs(perm_table, N)           # (N, N)
    pair_reps, inverse, pair_orbit_size = np.unique(
        pair_canon.ravel(), return_inverse=True,
        return_counts=True)
    pair_canon_idx = inverse.reshape(N, N)
    if verbose:
        print(f"  {len(pair_reps)} pair  orbits "
              f"(took {datetime.datetime.now() - t0}).")

    graph_rep_to_idx = {int(r): i for i, r in enumerate(graph_reps)}
    graph_orbit_idx = np.array(
        [graph_rep_to_idx[int(c)] for c in graph_canon],
        dtype=np.int64)

    return {
        "n": n,
        "m": m,
        "N": N,
        "f": f,
        "graph_canon": graph_canon,
        "graph_reps": graph_reps,
        "graph_orbit_size": graph_orbit_size,
        "graph_orbit_idx": graph_orbit_idx,
        "pair_canon_idx": pair_canon_idx,
        "pair_reps": pair_reps,
        "pair_orbit_size": pair_orbit_size,
        "canonical_edge": 0,
    }


# ---------------------------------------------------------------------------
# 5. Build and solve the orbit-reduced primal adversary SDP
# ---------------------------------------------------------------------------

def adversary_primal_orbit_reduced_triangle(
        n, solver=cp.MOSEK, verbose=False, orb=None):
    """Solve the orbit-reduced primal adversary SDP for triangle finding.

    SDP:
        max  sum_{x,y} Gamma[x, y]
        s.t. diag(beta) - Gamma o Delta_{e0}   >= 0
             Gamma[x, y] = 0           if f(x) = f(y)
             sum_{x : f(x)=1} beta[x]  = 1/2
             sum_{y : f(y)=0} beta[y]  = 1/2
    """
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

    # ---- Variables -----------------------------------------------------
    n_g = len(graph_reps)
    n_p = len(pair_reps)
    beta  = cp.Variable(n_g, nonneg=True, name="beta")
    gamma = cp.Variable(n_p, name="gamma")

    # ---- Gamma[x,y] = 0 when f(x) = f(y) -------------------------------
    same_f_mask = (pair_f_x == pair_f_y)
    constraints = []
    if same_f_mask.any():
        constraints.append(gamma[np.where(same_f_mask)[0]] == 0)

    # ---- beta normalisations -------------------------------------------
    zero_idx = np.where(graph_f == 0)[0]
    one_idx  = np.where(graph_f == 1)[0]
    constraints.append(graph_size[zero_idx].astype(float) @ beta[zero_idx]
                       == 0.5)
    constraints.append(graph_size[one_idx ].astype(float) @ beta[one_idx ]
                       == 0.5)

    # ---- PSD matrix M = diag(beta) - Gamma o Delta_{e0} ---------------
    M_terms = []

    # (a) diag(beta) part: sparse diagonal indicator per graph orbit.
    for a in range(n_g):
        rows = np.where(graph_orbit_idx == a)[0]
        D_a = sp.coo_matrix(
            (np.ones(len(rows)), (rows, rows)), shape=(N, N))
        M_terms.append(beta[a] * D_a)

    # (b) Gamma o Delta_{e0} part: Delta_{e0}[x, y] = 1 iff x[e0] != y[e0].
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

    for k in range(len(boundaries) - 1):
        s, e = boundaries[k], boundaries[k + 1]
        if s == e:
            continue
        O = int(keep_orbit_s[s])
        rows = keep_i_s[s:e]
        cols = keep_j_s[s:e]
        A_O = sp.coo_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(N, N))
        M_terms.append(-gamma[O] * A_O)

    M = sum(M_terms)
    constraints.append(M >> 0)

    # ---- Objective: sum of Gamma over D x D = sum_O |O| * gamma_O -----
    objective = cp.Maximize(pair_size.astype(float) @ gamma)

    problem = cp.Problem(objective, constraints)
    val = problem.solve(solver=solver, verbose=verbose)

    return {
        "value": val,
        "status": problem.status,
        "n": n,
        "num_graphs": N,
        "num_graph_orbits": n_g,
        "num_pair_orbits": n_p,
        "num_pair_orbits_free": int((~same_f_mask).sum()),
        "beta":  np.array(beta.value),
        "gamma": np.array(gamma.value),
        "graph_reps": graph_reps,
        "pair_reps": pair_reps,
        "orb": orb,
    }


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    chosen_solver = None
    avail = cp.installed_solvers()
    for s in (cp.MOSEK, cp.CLARABEL, cp.SCS):
        if s in avail:
            chosen_solver = s
            break
    print(f"Using solver: {chosen_solver}")

    ns = [3, 4]
    if "--n5" in sys.argv:
        ns.append(5)

    for n in ns:
        print(f"\n=== triangle finding on K_{n} ===")
        t0 = datetime.datetime.now()
        result = adversary_primal_orbit_reduced_triangle(
            n, solver=chosen_solver, verbose=False)
        t1 = datetime.datetime.now()
        print(f"  status:               {result['status']}")
        print(f"  ADV_pm value:         {result['value']:.6f}")
        print(f"  full SDP would be:    {result['num_graphs']} x "
              f"{result['num_graphs']}")
        print(f"  graph orbits:         {result['num_graph_orbits']}")
        print(f"  pair  orbits (total): {result['num_pair_orbits']}")
        print(f"  pair  orbits (free):  {result['num_pair_orbits_free']}")
        print(f"  total time:           {t1 - t0}")
