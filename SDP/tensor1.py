import itertools
import numpy as np
import cvxpy as cp
import math


def tuples_of_len(m: int, s: int):
    """All tuples in [m]^s."""
    if s == 0:
        return [()]
    return list(itertools.product(range(m), repeat=s))


def build_index_data(m: int, q: int):
    """
    All tuples, positions, cached prefix row indices,
    and cached np.ix_ objects for fast block slicing.
    """
    tuples = {s: tuples_of_len(m, s) for s in range(q + 1)}
    pos = {s: {alpha: k for k, alpha in enumerate(tuples[s])} for s in range(q + 1)}

    prefix_rows = {}
    prefix_ix = {}

    for s in range(1, q + 1):
        prefix_rows[s] = {}
        prefix_ix[s] = {}

        if s == 1:
            for i in range(m):
                rows = [pos[s][(i,)]]
                prefix_rows[s][i] = rows
                prefix_ix[s][i] = np.ix_(rows, rows)
        else:
            suffixes = tuples[s - 1]
            for i in range(m):
                rows = [pos[s][(i,) + a] for a in suffixes]
                prefix_rows[s][i] = rows
                prefix_ix[s][i] = np.ix_(rows, rows)

    return tuples, pos, prefix_rows, prefix_ix


_ZERO_CACHE = {}


def zero_const(shape):
    """Cache zero Constant blocks by shape."""
    if shape not in _ZERO_CACHE:
        _ZERO_CACHE[shape] = cp.Constant(np.zeros(shape))
    return _ZERO_CACHE[shape]


def block_diag(blocks):
    """Block diagonal matrix with cached zero blocks."""
    rows = []
    for i, Bi in enumerate(blocks):
        row = []
        r_i = Bi.shape[0]
        for j, Bj in enumerate(blocks):
            c_j = Bj.shape[1]
            if i == j:
                row.append(Bi)
            else:
                row.append(zero_const((r_i, c_j)))
        rows.append(row)
    return cp.bmat(rows)


def prefix_block(Ys, s: int, i: int, prefix_ix):
    """
    Returns block (Y_{i j, i j'})_{j,j' in [2n]^{s-1}}
    using cached np.ix_ slicing.
    """
    return Ys[prefix_ix[s][i]]


def full_domain(n: int):
    return list(itertools.product([-1.0, 1.0], repeat=n))


def monomial_vector(y: np.ndarray, tuple_list):
    """
    Vector indexed by alpha in [2n]^2t:
        phi_y[alpha] = prod_r y[alpha_r]
    """
    out = np.empty(len(tuple_list), dtype=float)
    for k, alpha in enumerate(tuple_list):
        prod = 1.0
        for a in alpha:
            prod *= y[a]
        out[k] = prod
    return out


def precompute_monomials(n: int, tuple_list, inputs):
    tuple_arr = np.array(tuple_list, dtype=int)
    phi_cache = {}

    for x in inputs:
        x = tuple(int(v) for v in x)
        z = np.concatenate([np.array(x, dtype=float), np.array([1.0])])
        phi_cache[x] = np.prod(z[tuple_arr], axis=1)

    return phi_cache


def prepare_structure(n: int, t: int):
    """
    Build reusable structural data for fixed (n,t).
    Reuse this if you solve several problems with the same n,t.
    """
    nn = n + 1
    tt = 2 * t
    tuples, pos, prefix_rows, prefix_ix = build_index_data(nn, tt)

    return {
        "n": n,
        "t": t,
        "nn": nn,
        "tt": tt,
        "tuples": tuples,
        "pos": pos,
        "prefix_rows": prefix_rows,
        "prefix_ix": prefix_ix,
    }


def tensor1(
    n: int,
    t: int,
    f_values: dict,
    structure=None,
):
    """
    Build the SDP for E(f,t).

    Parameters
    ----------
    n : int
        Number of input bits.
    t : int
        Number of queries.
    f_values : dict
        Maps x in {-1,1}^n (as tuples) to f(x) in [-1,1].
        For promise problems, include only promised inputs.
    literal_printed_form : bool
        If True, omits w <= 1 to match the printed corollary literally.
    structure : dict or None
        Reusable structural cache from prepare_structure(n,t).
    """
    if structure is None:
        structure = prepare_structure(n, t)

    nn = structure["nn"]
    tt = structure["tt"]
    tuples = structure["tuples"]
    pos = structure["pos"]
    prefix_rows = structure["prefix_rows"]
    prefix_ix = structure["prefix_ix"]

    eps = cp.Variable(nonneg=True)
    w = cp.Variable(nonneg=True)

    # One PSD variable per level s = 1,...,2t
    Y = {
        s: cp.Variable((len(tuples[s]), len(tuples[s])), PSD=True)
        for s in range(1, tt + 1)
    }

    # Y' indexed by {0} U [m]^q
    Yp = cp.Variable((1 + len(tuples[tt]), 1 + len(tuples[tt])), PSD=True)

    constraints = []

    # Y'_{0,0} = w
    constraints.append(Yp[0, 0] == w)

    # ||T||_cb <= 1  <=>  w <= 1
    constraints.append(w <= 1)

    # sum_{i in [2n]} Y_{i,i} <= w
    constraints.append(cp.sum(cp.diag(Y[1])) <= w)

    # Recursive PSD constraints
    for s in range(1, tt):
        lhs_blocks = [prefix_block(Y[s + 1], s + 1, i, prefix_ix) for i in range(nn)]
        lhs = sum(lhs_blocks[1:], lhs_blocks[0])

        rhs_blocks = [prefix_block(Y[s], s, i, prefix_ix) for i in range(nn)]
        rhs = block_diag(rhs_blocks)

        constraints.append(rhs - lhs >> 0)

    # Final PSD constraint
    rhs_final = block_diag([prefix_block(Y[tt], tt, i, prefix_ix) for i in range(nn)])
    constraints.append(rhs_final - Yp[1:, 1:] >> 0) #main source of constraints

    # Precompute monomials only for supplied inputs
    tuples_q = tuples[tt]
    phi_cache = precompute_monomials(
        n=n,
        tuple_list=tuples_q,
        inputs=f_values.keys(),
    )

    # Approximation constraints over supplied domain only
    for x, fx in f_values.items():
        x = tuple(int(v) for v in x)
        fx = float(fx)

        phi = phi_cache[x]
        approx = Yp[0, 1:] @ phi

        constraints.append(approx - fx <= eps)
        constraints.append(fx - approx <= eps)

    prob = cp.Problem(cp.Minimize(eps), constraints)

    return prob, {
        "eps": eps,
        "w": w,
        "Y": Y,
        "Yp": Yp,
        "tuples": tuples,
        "pos": pos,
        "prefix_rows": prefix_rows,
        "prefix_ix": prefix_ix,
        "phi_cache": phi_cache,
        "structure": structure,
    }




