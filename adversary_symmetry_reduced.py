import cvxpy as cp
from math import comb
from collections import defaultdict, deque

# ============================================================
# Basic helpers
# ============================================================

def normalize_input_bits(x):
    """
    Convert input to a 0/1 tuple.
    Accepts:
      - tuple/list in {0,1}
      - tuple/list in {-1,1}
      - string like '0101'
    """
    if isinstance(x, str):
        bits = tuple(int(c) for c in x)
        if not set(bits).issubset({0, 1}):
            raise ValueError(f"String input {x} is not binary.")
        return bits

    x = tuple(int(v) for v in x)
    vals = set(x)
    if vals.issubset({0, 1}):
        return x
    if vals.issubset({-1, 1}):
        return tuple(1 if v == 1 else 0 for v in x)

    raise ValueError(f"Input {x} is not in {{0,1}}^n or {{-1,1}}^n.")


def normalize_output(v):
    return 0 if v in (-1, 0, False) else 1


def hamming_weight(x):
    return sum(1 for bit in x if int(bit) == 1)


def intersection_weight(x, y):
    return sum(1 for a, b in zip(x, y) if int(a) == 1 and int(b) == 1)


def layer_sizes_full(n, allowed_layers):
    return {k: comb(n, k) for k in allowed_layers}


# ============================================================
# Full S_n pair-orbit count
# ============================================================

def num_pairs_in_orbit(n, a, b, t):
    """
    Number of ordered pairs (x,y) in {0,1}^n x {0,1}^n
    with |x|=a, |y|=b, |x∩y|=t.
    """
    if not (0 <= a <= n and 0 <= b <= n):
        return 0
    if not (max(0, a + b - n) <= t <= min(a, b)):
        return 0
    return comb(n, a) * comb(a, t) * comb(n - a, b - t)


# ============================================================
# H = Stab(1) ordered pair orbit key:
# (x1, y1, |x|, |y|, |x∩y|)
# ============================================================

def valid_H_orbit_key(n, key):
    x1, y1, a, b, t = key
    if x1 not in (0, 1) or y1 not in (0, 1):
        return False
    if not (0 <= a <= n and 0 <= b <= n):
        return False

    xy1 = x1 * y1
    a_rest = a - x1
    b_rest = b - y1
    t_rest = t - xy1

    if a_rest < 0 or b_rest < 0 or t_rest < 0:
        return False

    return max(0, a_rest + b_rest - (n - 1)) <= t_rest <= min(a_rest, b_rest)


def num_H_pairs_in_orbit(n, key):
    """
    Number of ordered pairs (x,y) in {0,1}^n x {0,1}^n
    with given H-orbit key (x1,y1,a,b,t).
    """
    x1, y1, a, b, t = key
    if not valid_H_orbit_key(n, key):
        return 0

    xy1 = x1 * y1
    a_rest = a - x1
    b_rest = b - y1
    t_rest = t - xy1

    return (
        comb(n - 1, a_rest)
        * comb(a_rest, t_rest)
        * comb((n - 1) - a_rest, b_rest - t_rest)
    )


def enumerate_H_orbit_keys(n, allowed_layers):
    """
    Enumerate all ordered H-orbit keys present in a domain that is
    a union of full Hamming layers.
    """
    keys = []
    for x1 in (0, 1):
        for y1 in (0, 1):
            for a in allowed_layers:
                for b in allowed_layers:
                    for t in range(max(0, a + b - n), min(a, b) + 1):
                        key = (x1, y1, a, b, t)
                        if num_H_pairs_in_orbit(n, key) > 0:
                            keys.append(key)
    return sorted(keys)


# ============================================================
# Structural constants for ordered H-orbit basis
# p_{r,t}^s = #{z : (x,z) in R_r, (z,y) in R_s}
# for any fixed (x,y) in orbit R_t
# ============================================================

def structural_constant_H(n, r, s, tkey):
    x1r, z1r, a_r, c_r, u_r = r
    z1s, y1s, c_s, b_s, v_s = s
    x1t, y1t, a_t, b_t, w_t = tkey

    # Endpoint / middle compatibility
    if x1r != x1t or y1s != y1t:
        return 0
    if z1r != z1s:
        return 0
    if a_r != a_t or b_s != b_t or c_r != c_s:
        return 0

    x1, y1, z1 = x1t, y1t, z1r
    a, b, c = a_t, b_t, c_r
    u, v, w = u_r, v_s, w_t

    # Counts on coordinates 2..n for fixed (x,y) in orbit tkey
    n11 = w - x1 * y1
    n10 = (a - x1) - n11
    n01 = (b - y1) - n11
    n00 = (n - 1) - n11 - n10 - n01

    if min(n11, n10, n01, n00) < 0:
        return 0

    # Remove coordinate-1 contribution from x-z and z-y overlaps
    u_rem = u - x1 * z1
    v_rem = v - z1 * y1
    c_rem = c - z1

    if min(u_rem, v_rem, c_rem) < 0:
        return 0

    total = 0

    # alpha = number of (1,1) coordinates among 2..n where z=1
    alpha_min = max(
        0,
        u_rem - n10,
        v_rem - n01,
        u_rem + v_rem - c_rem,
    )
    alpha_max = min(
        n11,
        u_rem,
        v_rem,
        n00 + u_rem + v_rem - c_rem,
    )

    for alpha in range(alpha_min, alpha_max + 1):
        a11 = alpha
        a10 = u_rem - alpha
        a01 = v_rem - alpha
        a00 = c_rem - u_rem - v_rem + alpha

        if min(a11, a10, a01, a00) < 0:
            continue
        if a11 > n11 or a10 > n10 or a01 > n01 or a00 > n00:
            continue

        total += (
            comb(n11, a11)
            * comb(n10, a10)
            * comb(n01, a01)
            * comb(n00, a00)
        )

    return total


# ============================================================
# Connected components for the reduced regular-representation graph
# ============================================================

def connected_components_from_transitions(M, transitions):
    """
    Build a graph on reduced indices 0..M-1 where t is connected to s
    if some basis matrix has a nonzero (s,t) entry. Return connected comps.
    """
    adj = [set() for _ in range(M)]

    for r_trans in transitions:
        for t_idx, items in r_trans.items():
            for s_idx, coeff in items:
                if coeff != 0:
                    adj[t_idx].add(s_idx)
                    adj[s_idx].add(t_idx)

    seen = [False] * M
    comps = []

    for v in range(M):
        if seen[v]:
            continue
        q = deque([v])
        seen[v] = True
        comp = []
        while q:
            u = q.popleft()
            comp.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    q.append(w)
        comps.append(sorted(comp))

    return comps


def restrict_transitions_to_components(transitions, components):
    """
    Convert global sparse transitions into per-component sparse transitions.

    Output:
      block_transitions[cid][r_idx][t_local] = [(s_local, coeff), ...]
      block_sizes[cid]
    """
    M = len(transitions)
    comp_of = {}
    local_index = {}
    block_sizes = []

    for cid, comp in enumerate(components):
        block_sizes.append(len(comp))
        for j, idx in enumerate(comp):
            comp_of[idx] = cid
            local_index[idx] = j

    block_transitions = [
        [defaultdict(list) for _ in range(M)]
        for _ in components
    ]

    for r_idx, r_trans in enumerate(transitions):
        for t_idx, items in r_trans.items():
            cid = comp_of[t_idx]
            t_local = local_index[t_idx]

            for s_idx, coeff in items:
                if comp_of[s_idx] != cid:
                    raise ValueError("Transition crosses components; decomposition is inconsistent.")
                s_local = local_index[s_idx]
                block_transitions[cid][r_idx][t_local].append((s_local, coeff))

    return block_transitions, block_sizes


# ============================================================
# Precomputation for sparse Step 1 1/2 with connected-component splitting
# ============================================================

def precompute_step_one_half_sparse_blocks(n, allowed_layers):
    """
    Precompute sparse regular-representation transitions and split the reduced
    matrix into connected components.

    Returns a dict with:
      - orbit_keys
      - orbit_sizes
      - norms
      - transitions: global sparse regular-representation data
      - components
      - block_transitions
      - block_sizes
      - regular_rep_size
    """
    orbit_keys = enumerate_H_orbit_keys(n, allowed_layers)
    M = len(orbit_keys)

    orbit_sizes = [num_H_pairs_in_orbit(n, key) for key in orbit_keys]
    norms = [size ** 0.5 for size in orbit_sizes]

    # Candidate s-orbits indexed by (z1, y1, c, b)
    # s = (z1, y1, c, b, v)
    s_index = defaultdict(list)
    for s_idx, key in enumerate(orbit_keys):
        z1, y1, c, b, v = key
        s_index[(z1, y1, c, b)].append((v, s_idx))

    # transitions[r_idx][t_idx] = [(s_idx, coeff), ...]
    # coeff = (||C_s|| / ||C_t||) * p_{r,t}^s
    transitions = [defaultdict(list) for _ in range(M)]

    for r_idx, r in enumerate(orbit_keys):
        x1r, z1r, a_r, c_r, u_r = r

        for t_idx, tkey in enumerate(orbit_keys):
            x1t, y1t, a_t, b_t, w_t = tkey

            # Fast compatibility checks
            if x1r != x1t:
                continue
            if a_r != a_t:
                continue

            candidates = s_index.get((z1r, y1t, c_r, b_t), [])
            if not candidates:
                continue

            t_norm = norms[t_idx]
            if t_norm == 0:
                continue

            row_entries = []
            for _, s_idx in candidates:
                s = orbit_keys[s_idx]
                p_val = structural_constant_H(n, r, s, tkey)
                if p_val == 0:
                    continue
                coeff = (norms[s_idx] / t_norm) * p_val
                row_entries.append((s_idx, coeff))

            if row_entries:
                transitions[r_idx][t_idx] = row_entries

    # Coarse exact block decomposition from connectivity
    components = connected_components_from_transitions(M, transitions)
    block_transitions, block_sizes = restrict_transitions_to_components(transitions, components)

    return {
        "n": n,
        "allowed_layers": tuple(sorted(allowed_layers)),
        "orbit_keys": orbit_keys,
        "orbit_sizes": orbit_sizes,
        "norms": norms,
        "transitions": transitions,
        "components": components,
        "block_transitions": block_transitions,
        "block_sizes": block_sizes,
        "regular_rep_size": M,
        "num_blocks": len(components),
    }


# ============================================================
# Solver using sparse Step 1 1/2 + block splitting
# ============================================================

def adversary_symmetry_reduced(
    n,
    f_values,
    solver=None,
    verbose=False,
    precomp=None,
):
    """
    Sparse combinatorial Step 1 1/2 implementation for the primal adversary SDP,
    with exact block splitting of the reduced PSD matrix.

    Assumptions:
      - f is symmetric on its promise domain
      - the promise domain is a union of full Hamming layers
      - inputs may be in {0,1}^n or {-1,1}^n

    Parameters
    ----------
    half_objective : bool
        If True, optimize 0.5 * sum_{x,y} Gamma[x,y].
        If False, optimize the full symmetric sum.

    precomp : dict or None
        Output of precompute_step_one_half_sparse_blocks(n, allowed_layers).
        If provided, reuses the algebra precomputation.
    """
    # ----------------------------------------------------------
    # 1. Normalize domain and verify symmetric full-layer promise
    # ----------------------------------------------------------
    orig_keys = list(f_values.keys())
    domain = [normalize_input_bits(x) for x in orig_keys]
    values01 = [normalize_output(f_values[x]) for x in orig_keys]

    if not domain:
        raise ValueError("Empty domain.")

    for x in domain:
        if len(x) != n:
            raise ValueError(f"Input {x} has length {len(x)} but expected {n}.")

    layer_value = {}
    layer_count_seen = defaultdict(int)

    for x, fx in zip(domain, values01):
        k = hamming_weight(x)
        layer_count_seen[k] += 1
        if k in layer_value and layer_value[k] != fx:
            raise ValueError(
                f"Function is not symmetric on the given domain: layer {k} has both outputs."
            )
        layer_value[k] = fx

    allowed_layers = sorted(layer_value.keys())
    full_layer_sizes = layer_sizes_full(n, allowed_layers)

    for k in allowed_layers:
        if layer_count_seen[k] != full_layer_sizes[k]:
            raise ValueError(
                f"Layer {k} is incomplete: saw {layer_count_seen[k]} points but full layer has {full_layer_sizes[k]}. "
                "This implementation assumes the promise domain is a union of full Hamming layers."
            )

    zero_layers = sorted(k for k, v in layer_value.items() if v == 0)
    one_layers = sorted(k for k, v in layer_value.items() if v == 1)

    if not zero_layers or not one_layers:
        raise ValueError("The function must take both output values on the domain.")

    # ----------------------------------------------------------
    # 2. Reduced variables
    # ----------------------------------------------------------
    beta = {
        k: cp.Variable(nonneg=True, name=f"beta_{k}")
        for k in allowed_layers
    }

    gamma_keys = []
    for a in allowed_layers:
        for b in allowed_layers:
            if layer_value[a] == layer_value[b]:
                continue
            aa, bb = min(a, b), max(a, b)
            for t in range(max(0, aa + bb - n), min(aa, bb) + 1):
                key = (aa, bb, t)
                if key not in gamma_keys:
                    gamma_keys.append(key)
    gamma_keys = sorted(gamma_keys)

    gamma = {
        key: cp.Variable(name=f"gamma_{key[0]}_{key[1]}_{key[2]}")
        for key in gamma_keys
    }

    def gamma_var(a, b, t):
        aa, bb = min(a, b), max(a, b)
        key = (aa, bb, t)
        return gamma[key] if key in gamma else 0.0

    # ----------------------------------------------------------
    # 3. Precomputation
    # ----------------------------------------------------------
    if precomp is None:
        precomp = precompute_step_one_half_sparse_blocks(n, allowed_layers)
    else:
        if precomp["n"] != n:
            raise ValueError("Precomputation n does not match.")
        if tuple(sorted(allowed_layers)) != tuple(precomp["allowed_layers"]):
            raise ValueError("Precomputation allowed_layers do not match current function.")

    orbit_keys = precomp["orbit_keys"]
    block_transitions = precomp["block_transitions"]
    block_sizes = precomp["block_sizes"]

    # ----------------------------------------------------------
    # 4. Expand M_1 = sum_r z_r C_r in the ordered H-orbit basis
    # ----------------------------------------------------------
    z = []
    for key in orbit_keys:
        x1, y1, a, b, t = key

        # diagonal orbit iff x=y
        is_diagonal_orbit = (x1 == y1) and (a == b == t)
        diag_term = beta[a] if is_diagonal_orbit else 0.0

        delta1 = 1 if x1 != y1 else 0
        if layer_value[a] != layer_value[b] and delta1 == 1:
            gamma_term = gamma_var(a, b, t)
        else:
            gamma_term = 0.0

        z.append(diag_term - gamma_term)

    # ----------------------------------------------------------
    # 5. Constraints
    # ----------------------------------------------------------
    constraints = []

    constraints.append(
        cp.sum([full_layer_sizes[k] * beta[k] for k in zero_layers]) == 0.5
    )
    constraints.append(
        cp.sum([full_layer_sizes[k] * beta[k] for k in one_layers]) == 0.5
    )

    # Assemble one reduced PSD block per connected component
    for cid, bt in enumerate(block_transitions):
        m = block_sizes[cid]
        block_entries = [[0 for _ in range(m)] for __ in range(m)]

        for r_idx, zr in enumerate(z):
            r_trans = bt[r_idx]
            if not r_trans:
                continue
            for t_local, items in r_trans.items():
                for s_local, coeff in items:
                    block_entries[s_local][t_local] = block_entries[s_local][t_local] + zr * coeff

        Bred = cp.bmat(block_entries)
        constraints.append(Bred >> 0)

    # ----------------------------------------------------------
    # 6. Objective assembled combinatorially
    # ----------------------------------------------------------
    objective_expr = 0
    for (a, b, t), var in gamma.items():
        if a == b:
            mult = num_pairs_in_orbit(n, a, b, t)
        else:
            mult = num_pairs_in_orbit(n, a, b, t) + num_pairs_in_orbit(n, b, a, t)
        objective_expr += mult * var

    problem = cp.Problem(cp.Maximize(objective_expr), constraints)

    if solver is not None:
        value = problem.solve(solver=solver, verbose=verbose)
    else:
        try:
            value = problem.solve(solver=cp.MOSEK, verbose=verbose)
        except Exception:
            value = problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6)

    
    return {
        "value": value,
        "status": problem.status,
        "beta": {k: beta[k].value for k in beta},
        "gamma": {k: gamma[k].value for k in gamma},
        "allowed_layers": allowed_layers,
        "regular_rep_size": precomp["regular_rep_size"],
        "num_blocks": precomp["num_blocks"],
        "block_sizes": precomp["block_sizes"],
        "orbit_keys_H": orbit_keys,
        "precomp": precomp,
    }
