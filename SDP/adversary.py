import itertools
import numpy as np
import cvxpy as cp
import math
from collections import defaultdict
import datetime


#### PRIMAL ####

def adversary_primal(
    n: int,
    f_values: dict,
    solver=None, 
    verbose=False
):
    
    """
    
    Solves the reformulated adversary bound primal SDP, as in Arjan's thesis, section 6.2.4

    Parameters
    -------

    n : int
    Input dimension of f : D -> {-1,1}
    f_values : dict
    Domain points x -> f(x); total or promise-only

    Returns
    -------
    value : float
        Optimal SDP value.
    X_values : list[np.ndarray]
        Optimal matrices X_j.
    t_value : float
        Optimal t.
    
    """

    domain = []
    values = []
    f_0 = []
    f_1 = []

    for i, (x, f_x) in enumerate(f_values.items()):
        domain.append(x)
        values.append(f_x)
        if f_x == -1 or f_x == 0:
            f_0.append(i)
        else:
            f_1.append(i)

    m = len(domain)

    Gamma = cp.Variable((m, m), symmetric=True)
    beta = cp.Variable(m)

    # Define the Delta_j matrices

    deltas = []

    for j in range(n): 
        Dj = np.zeros((m, m), dtype=float)
        for a, xa in enumerate(domain):
            for b, xb in enumerate(domain):
                Dj[a, b] = 1.0 if xa[j] != xb[j] else 0.0
        deltas.append(Dj)

    ## CONSTRAINTS ##

    constraints = []

    # Optional 

    constraints.append(beta >= 0)

    # Impose Gamma_ab = 0 if f(a) = f(b)

    for a in range(m):
        for b in range(m):
            if values[a] == values[b]:
                constraints.append(Gamma[a, b] == 0)

    # First constraint

    for j in range(n):
        constraints.append(cp.diag(beta) - cp.multiply(Gamma, deltas[j]) >> 0)

    # Equality constraints

    constraints.append(cp.sum(beta[f_0]) == 0.5)
    constraints.append(cp.sum(beta[f_1]) == 0.5)

    objective = cp.Maximize(cp.sum(Gamma))
    problem = cp.Problem(objective, constraints)

    if solver is not None:
        value = problem.solve(solver=solver, verbose=verbose)
    else:
        value = problem.solve(solver=cp.SCS, verbose=verbose)

    return value, Gamma.value, beta.value, problem.status


#### DUAL ####

def adversary_dual(
    n: int,
    f_values: dict,
    solver=None,
    verbose=True
):
    """
    Solves the dual adversary bound SDP, as in Arjan's thesis, section 6.2.5.

    Parameters
    ----------
    n : int
        Input dimension of f : D -> {-1,1} or {0,1}.
    f_values : dict
        Map x -> f(x), for total or promise domain.
        Each x must be indexable with length n.

    Returns
    -------
    value : float
        Optimal SDP value.
    X_values : list[np.ndarray]
        Optimal matrices X_j.
    t_value : float
        Optimal t.
    status : str
        CVXPY status string.
    """

    domain = []
    values = []


    for x, f_x in f_values.items():
        if len(x) != n:
            raise ValueError(f"Input {x} has length {len(x)} but expected {n}.")
        domain.append(x)
        values.append(f_x)

    m = len(domain)

    X = [cp.Variable((m, m), PSD=True) for _ in range(n)]

    t = cp.Variable()

    constraints = []

    # For every x,y with f(x) != f(y), the first constraint sums to 1
    for a, xa in enumerate(domain):
        for b, xb in enumerate(domain):
            if values[a] != values[b]:
                differing_positions = [j for j in range(n) if xa[j] != xb[j]]
                constraints.append(
                    cp.sum([X[j][a, b] for j in differing_positions]) == 1
                )

    # t >= sum_j X_j[x,x] for every, a way of implementing max
    for a in range(m):
        constraints.append(
            cp.sum([X[j][a, a] for j in range(n)]) <= t
        )

    objective = cp.Minimize(t)
    problem = cp.Problem(objective, constraints)

    if solver is not None:
        value = problem.solve(solver=solver, verbose=verbose)
    else:
        value = problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6)

    return value, [Xj.value for Xj in X], t.value, problem.status
    
### SYMMETRY REDUCED FOR FUNCTIONS THAT ONLY DEPEND ON THE HAMMING WEIGHT ###


# STEP 1: still too slow

### Basically it reduces the number of variables by symmetry.


def hamming_weight(x):
    return sum(int(bit) for bit in x)


def intersection_weight(x, y):
    return sum(int(a and b) for a, b in zip(x, y))


def normalize_output(v):
    return 0 if v in (-1, 0, False) else 1


def adversary_primal_orbit_reduced(
    n: int,
    f_values: dict,
    solver=None,
    verbose=False,
):
    
    """
    Solves the dual adversary bound SDP, as in Arjan's thesis, section 6.2.5.
    Uses symmetry reduction (step 1 in the SDP paper) for symmetric functions.

    Parameters
    ----------
    n : int
        Input dimension of f : D -> {-1,1} or {0,1}.
    f_values : dict
        Map x -> f(x), for total or promise domain.
        Each x must be indexable with length n.

    Returns
    -------
    value : float
        Optimal SDP value.
    X_values : list[np.ndarray]
        Optimal matrices X_j.
    t_value : float
        Optimal t.
    status : str
        CVXPY status string.
    """

    domain = list(f_values.keys())
    if len(domain) == 0:
        raise ValueError("Empty domain.")

    orig_keys = list(f_values.keys())
    domain = [tuple(int(c) for c in x) if isinstance(x, str) else tuple(x) for x in orig_keys]
    values01 = [normalize_output(f_values[x]) for x in orig_keys]

    for x in domain:
        if len(x) != n:
            raise ValueError(f"Input {x} has length {len(x)} but expected {n}.")

    m = len(domain)

    layer_value = {}
    layer_size = defaultdict(int)

    for x, fx in zip(domain, values01):
        k = int(hamming_weight(x))
        layer_size[k] += 1
        if k in layer_value and layer_value[k] != fx:
            raise ValueError(
                f"Function is not symmetric on the given domain: layer {k} has both outputs."
            )
        layer_value[k] = fx

    zero_layers = sorted(int(k) for k, v in layer_value.items() if v == 0)
    one_layers = sorted(int(k) for k, v in layer_value.items() if v == 1)

    if not zero_layers or not one_layers:
        raise ValueError("The function must take both output values on the given domain.")

    beta = {k: cp.Variable(nonneg=True, name=f"beta_{k}") for k in sorted(layer_value)}

    orbit_keys = set()
    for x in domain:
        a = int(hamming_weight(x))
        for y in domain:
            b = int(hamming_weight(y))
            t = int(intersection_weight(x, y))
            orbit_keys.add((a, b, t))

    orbit_keys = sorted(orbit_keys)
    gamma = {
        key: cp.Variable(name=f"gamma_{key[0]}_{key[1]}_{key[2]}")
        for key in orbit_keys
    }

    Gamma = [[None for _ in range(m)] for _ in range(m)]
    beta_diag = [None for _ in range(m)]

    first_time = datetime.datetime.now()

    for i, x in enumerate(domain):
        a = int(hamming_weight(x))
        beta_diag[i] = beta[a]
        for j, y in enumerate(domain):
            b = int(hamming_weight(y))
            t = int(intersection_weight(x, y))
            Gamma[i][j] = gamma[(a, b, t)]

    later_time = datetime.datetime.now()

    print("time spent in the domain :",later_time - first_time)

    Gamma_expr = cp.bmat(Gamma)
    beta_diag_expr = cp.diag(cp.hstack(beta_diag))

    Delta1 = np.zeros((m, m), dtype=float)
    for i, x in enumerate(domain):
        for j, y in enumerate(domain):
            Delta1[i, j] = 1.0 if x[0] != y[0] else 0.0

    M1 = beta_diag_expr - cp.multiply(Gamma_expr, Delta1)

    constraints = []

    for (a, b, t), var in gamma.items():
        if a in layer_value and b in layer_value and layer_value[a] == layer_value[b]:
            constraints.append(var == 0)

    for (a, b, t) in orbit_keys:
        if (b, a, t) in gamma and (a, b, t) <= (b, a, t):
            constraints.append(gamma[(a, b, t)] == gamma[(b, a, t)])

    constraints.append(
        cp.sum([layer_size[k] * beta[k] for k in zero_layers]) == 0.5
    )
    constraints.append(
        cp.sum([layer_size[k] * beta[k] for k in one_layers]) == 0.5
    )

    constraints.append(M1 >> 0)

    objective = cp.Maximize(cp.sum(Gamma_expr))
    problem = cp.Problem(objective, constraints)

    if solver is not None:
        value = problem.solve(solver=solver, verbose=verbose)
    else:
        try:
            value = problem.solve(solver=cp.MOSEK, verbose=verbose)
        except Exception:
            value = problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6)

    beta_values = {k: beta[k].value for k in beta}
    gamma_values = {key: gamma[key].value for key in gamma}

    return value, beta_values, gamma_values, problem.status


## STEP 1 1/2

# ============================================================
# Basic helpers
# ============================================================

def normalize_input_bits(x):
    """
    Convert an input into a 0/1 tuple.

    Accepted formats:
      - tuple/list in {0,1}
      - tuple/list in {-1,1}
      - string like '0101'
    """
    if isinstance(x, str):
        bits = tuple(int(c) for c in x)
        vals = set(bits)
        if not vals.issubset({0, 1}):
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
    """
    Normalize function values to 0/1.
    """
    return 0 if v in (-1, 0, False) else 1


def hamming_weight(x):
    return sum(1 for bit in x if int(bit) == 1)


def intersection_weight(x, y):
    return sum(1 for a, b in zip(x, y) if int(a) == 1 and int(b) == 1)


# ============================================================
# H = Stab(1) orbit keys on ordered pairs
# Under H=S_{n-1}, orbit of (x,y) determined by:
#   x1, y1, |x|, |y|, |x ∩ y|
# ============================================================

def H_orbit_key(x, y):
    return (
        int(x[0]),
        int(y[0]),
        hamming_weight(x),
        hamming_weight(y),
        intersection_weight(x, y),
    )


def transpose_key(key):
    x1, y1, a, b, t = key
    return (y1, x1, b, a, t)


# ============================================================
# Ordered H-orbit basis C_r
# ============================================================

def build_H_ordered_orbit_basis(domain):
    """
    Build the ordered-orbit basis C_r for the H-invariant algebra on D x D.

    Returns:
      ordered_keys : list of orbit keys
      C            : list of 0/1 orbit matrices
      orbit_pairs  : dict key -> list[(i,j)]
    """
    m = len(domain)
    orbit_pairs = defaultdict(list)

    for i, x in enumerate(domain):
        for j, y in enumerate(domain):
            orbit_pairs[H_orbit_key(x, y)].append((i, j))

    ordered_keys = sorted(orbit_pairs.keys())
    C = []

    for key in ordered_keys:
        M = np.zeros((m, m), dtype=float)
        for i, j in orbit_pairs[key]:
            M[i, j] = 1.0
        C.append(M)

    return ordered_keys, C, orbit_pairs


# ============================================================
# Multiplication parameters for ordered orbit basis
# C_r C_s = sum_t p[r,s,t] C_t
# ============================================================

def multiplication_parameters(C, ordered_keys):
    """
    Compute p[r,s,t] from products of ordered orbit basis matrices.
    Since orbit basis matrices are disjoint 0/1 supports, the coefficient
    on orbit t is read from any representative entry in that orbit.
    """
    M = len(C)
    p = np.zeros((M, M, M), dtype=float)

    # representative entry for each orbit
    reps = []
    for Ct in C:
        idx = np.argwhere(Ct > 0.5)
        if len(idx) == 0:
            raise ValueError("Empty orbit basis element encountered.")
        reps.append(tuple(idx[0]))

    for r in range(M):
        for s in range(M):
            P = C[r] @ C[s]
            for t in range(M):
                i, j = reps[t]
                p[r, s, t] = P[i, j]

    return p


# ============================================================
# Regular *-representation matrices L(C_r)
# Following the paper:
#   L(C_r)_{st} = <C_r C_t, C_s> / (||C_t|| ||C_s||)
#               = ||C_s|| / ||C_t|| * p[r,t,s]
# ============================================================

def regular_representation(C, p):
    """
    Build L(C_r) matrices for the ordered orbit basis.
    """
    M = len(C)
    norms = np.array([np.sqrt(np.sum(Cr * Cr)) for Cr in C], dtype=float)

    L = []
    for r in range(M):
        Lr = np.zeros((M, M), dtype=float)
        for s in range(M):
            for t in range(M):
                if norms[t] == 0:
                    raise ValueError("Zero norm basis element encountered.")
                Lr[s, t] = (norms[s] / norms[t]) * p[r, t, s]
        L.append(Lr)

    return L, norms


# ============================================================
# Step 1 1/2 SDP for primal adversary bound
# ============================================================

def adversary_primal_step_one_half_symmetric(
    n,
    f_values,
    solver=None,
    verbose=False,
):
    """
    Step 1 1/2 implementation for the primal adversary SDP
    for a symmetric Boolean function f(x)=g(|x|).

    Uses:
      - Step 1 orbit reduction for beta and Gamma
      - Step 1 1/2 regular *-representation for the PSD constraint on M_1

    Returns a dictionary with the solution and algebra data.
    """

    # ----------------------------------------------------------
    # 1. Normalize domain and outputs
    # ----------------------------------------------------------
    orig_keys = list(f_values.keys())
    domain = [normalize_input_bits(x) for x in orig_keys]
    values01 = [normalize_output(f_values[x]) for x in orig_keys]

    if not domain:
        raise ValueError("Empty domain.")

    for x in domain:
        if len(x) != n:
            raise ValueError(f"Input {x} has length {len(x)} but expected {n}.")

    # ----------------------------------------------------------
    # 2. Check symmetry by Hamming layer
    # ----------------------------------------------------------
    layer_value = {}
    layer_size = defaultdict(int)

    for x, fx in zip(domain, values01):
        k = hamming_weight(x)
        layer_size[k] += 1
        if k in layer_value and layer_value[k] != fx:
            raise ValueError(
                f"Function is not symmetric on the given domain: "
                f"layer {k} has both outputs."
            )
        layer_value[k] = fx

    zero_layers = sorted(k for k, v in layer_value.items() if v == 0)
    one_layers = sorted(k for k, v in layer_value.items() if v == 1)

    if not zero_layers or not one_layers:
        raise ValueError("The function must take both output values on the domain.")

    # ----------------------------------------------------------
    # 3. Reduced variables
    # ----------------------------------------------------------
    beta = {
        k: cp.Variable(nonneg=True, name=f"beta_{k}")
        for k in sorted(layer_value)
    }

    # Keep only unordered layer pairs with opposite outputs
    gamma_keys = set()
    for x in domain:
        a = hamming_weight(x)
        for y in domain:
            b = hamming_weight(y)
            t = intersection_weight(x, y)
            aa, bb = min(a, b), max(a, b)
            if layer_value[a] != layer_value[b]:
                gamma_keys.add((aa, bb, t))

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
    # 4. Ordered H-orbit basis and multiplication tables
    # ----------------------------------------------------------
    ordered_keys, C, orbit_pairs = build_H_ordered_orbit_basis(domain)
    p = multiplication_parameters(C, ordered_keys)
    L, norms = regular_representation(C, p)

    # ----------------------------------------------------------
    # 5. Express M_1 in the ordered orbit basis:
    #    M_1 = sum_r z_r C_r
    # ----------------------------------------------------------
    z = []
    for key in ordered_keys:
        x1, y1, a, b, t = key

        # diagonal orbit iff x=y, for binary strings:
        is_diagonal_orbit = (x1 == y1) and (a == b == t)
        diag_term = beta[a] if is_diagonal_orbit else 0.0

        # Delta_1 contributes iff first coordinate differs
        delta1 = 1 if x1 != y1 else 0

        if layer_value[a] != layer_value[b] and delta1 == 1:
            gamma_term = gamma_var(a, b, t)
        else:
            gamma_term = 0.0

        z.append(diag_term - gamma_term)

    # ----------------------------------------------------------
    # 6. Constraints
    # ----------------------------------------------------------
    constraints = []

    # class-mass constraints
    constraints.append(
        cp.sum([layer_size[k] * beta[k] for k in zero_layers]) == 0.5
    )
    constraints.append(
        cp.sum([layer_size[k] * beta[k] for k in one_layers]) == 0.5
    )

    # Step 1 1/2 PSD constraint:
    #   sum_r z_r L(C_r) >= 0
    Mred = 0
    for zr, Lr in zip(z, L):
        Mred = Mred + zr * Lr
    constraints.append(Mred >> 0)

    # ----------------------------------------------------------
    # 7. Objective
    # maximize sum_{x,y} Gamma[x,y]
    # ----------------------------------------------------------
    objective = 0
    for x in domain:
        a = hamming_weight(x)
        for y in domain:
            b = hamming_weight(y)
            t = intersection_weight(x, y)
            if layer_value[a] != layer_value[b]:
                objective += gamma_var(a, b, t)

    problem = cp.Problem(cp.Maximize(objective), constraints)

    if solver is not None:
        value = problem.solve(solver=solver, verbose=verbose)
    else:
        try:
            value = problem.solve(solver=cp.MOSEK, verbose=verbose)
        except Exception:
            value = problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6)

    beta_values = {k: beta[k].value for k in beta}
    gamma_values = {key: gamma[key].value for key in gamma}

    return {
        "value": value,
        "status": problem.status,
        "beta": beta_values,
        "gamma": gamma_values,
        "ordered_orbit_keys": ordered_keys,
        "L_matrices": L,
        "norms": norms,
        "reduced_matrix_size": len(L),
        "domain_01": domain,
    }

## STEP 2

# ============================================================
# Basic helpers
# ============================================================

def normalize_input_bits(x):
    """
    Convert an input into a 0/1 tuple.

    Accepted formats:
      - tuple/list in {0,1}
      - tuple/list in {-1,1}
      - string like '0101'
    """
    if isinstance(x, str):
        bits = tuple(int(c) for c in x)
        vals = set(bits)
        if not vals.issubset({0, 1}):
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
    """
    Normalize function values to 0/1.
    """
    return 0 if v in (-1, 0, False) else 1


def hamming_weight(x):
    return sum(1 for bit in x if int(bit) == 1)


def intersection_weight(x, y):
    return sum(1 for a, b in zip(x, y) if int(a) == 1 and int(b) == 1)


# ============================================================
# H = Stab(1) orbit keys on ordered pairs
# For x,y in {0,1}^n, the H-orbit is determined by:
#   x_1, y_1, |x|, |y|, |x ∩ y|
# ============================================================

def H_orbit_key(x, y):
    return (
        int(x[0]),
        int(y[0]),
        hamming_weight(x),
        hamming_weight(y),
        intersection_weight(x, y),
    )


def transpose_key(key):
    x1, y1, a, b, t = key
    return (y1, x1, b, a, t)


# ============================================================
# Build Hermitian orbit basis for the H-invariant algebra
# ============================================================

def build_H_hermitian_orbit_basis(domain):
    """
    Returns a Hermitian basis B_u of the H-invariant matrix algebra
    acting on matrices indexed by domain.

    Construction:
      - build ordered-orbit matrices C_k for H-orbits on D x D
      - merge each orbit with its transpose orbit:
            B_u = C_k                    if key == transpose_key(key)
            B_u = C_k + C_tkey          otherwise
    """
    m = len(domain)

    ordered_orbit_pairs = defaultdict(list)
    for i, x in enumerate(domain):
        for j, y in enumerate(domain):
            ordered_orbit_pairs[H_orbit_key(x, y)].append((i, j))

    ordered_keys = sorted(ordered_orbit_pairs.keys())

    # Ordered orbit matrices
    C = {}
    for key in ordered_keys:
        M = np.zeros((m, m), dtype=float)
        for i, j in ordered_orbit_pairs[key]:
            M[i, j] = 1.0
        C[key] = M

    # Merge transpose-paired orbits into Hermitian basis
    visited = set()
    hermitian_keys = []
    B = []

    for key in ordered_keys:
        if key in visited:
            continue

        tkey = transpose_key(key)

        if tkey not in C:
            raise ValueError(
                f"Transpose orbit key {tkey} was not found. "
                f"This usually means the domain/orbit normalization is inconsistent."
            )

        if tkey == key:
            hermitian_keys.append(key)
            B.append(C[key])
            visited.add(key)
        else:
            rep = min(key, tkey)
            if rep == key:
                hermitian_keys.append(rep)
                B.append(C[key] + C[tkey])
            visited.add(key)
            visited.add(tkey)

    return hermitian_keys, B


# ============================================================
# Numerical Step 2: block diagonalize a Hermitian matrix *-algebra
# ============================================================

def numerical_block_diagonalize_algebra(B, eig_tol=1e-9, zero_tol=1e-8, seed=0):
    """
    Input:
      B : list of Hermitian basis matrices spanning the algebra

    Output:
      Q : orthogonal change of basis
      block_slices : list of slices for the blocks
      B_blocks : list over basis elements, where B_blocks[r][k] is the k-th block
    """
    rng = np.random.default_rng(seed)
    n = B[0].shape[0]

    # Generic Hermitian element
    coeffs = rng.standard_normal(len(B))
    H = sum(c * Br for c, Br in zip(coeffs, B))

    first_time = datetime.datetime.now()

    # Diagonalize
    evals, U = np.linalg.eigh(H)

    later_time = datetime.datetime.now()

    print("[BLOCK DIAGONALIZATION] Time spent diagonalizing the commutative subalgebra:",later_time - first_time)

    first_time = datetime.datetime.now()

    # Cluster nearly equal eigenvalues
    clusters = []
    current = [0]
    for i in range(1, n):
        if abs(evals[i] - evals[i - 1]) <= eig_tol:
            current.append(i)
        else:
            clusters.append(current)
            current = [i]
    clusters.append(current)

    later_time = datetime.datetime.now()

    # Transform basis into eigenbasis of H
    B_tilde = [U.T @ Br @ U for Br in B]

    print("[BLOCK DIAGONALIZATION] Time spent clustering the eigenvalues:",later_time - first_time)

    first_time = datetime.datetime.now()

    # Graph on clusters: two clusters are connected if some basis element
    # has a non-negligible block between them
    num_clusters = len(clusters)
    adjacency = [[False] * num_clusters for _ in range(num_clusters)]
    for i in range(num_clusters):
        adjacency[i][i] = True

    for i in range(num_clusters):
        Ii = clusters[i]
        for j in range(i + 1, num_clusters):
            Jj = clusters[j]
            coupled = False
            for Br in B_tilde:
                block = Br[np.ix_(Ii, Jj)]
                if np.linalg.norm(block, ord="fro") > zero_tol:
                    coupled = True
                    break
            if coupled:
                adjacency[i][j] = adjacency[j][i] = True

    # Connected components = final blocks
    seen = [False] * num_clusters
    components = []

    for i in range(num_clusters):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in range(num_clusters):
                if adjacency[v][w] and not seen[w]:
                    seen[w] = True
                    stack.append(w)
        components.append(sorted(comp))

    later_time = datetime.datetime.now()

    print("[BLOCK DIAGONALIZATION] Time spent finding building the equivalence relation:",later_time - first_time)

    first_time = datetime.datetime.now()

    # Reorder basis vectors to make blocks contiguous
    perm = []
    block_sizes = []
    for comp in components:
        idxs = []
        for c in comp:
            idxs.extend(clusters[c])
        perm.extend(idxs)
        block_sizes.append(len(idxs))

    P = np.eye(n)[:, perm]
    Q = U @ P

    later_time = datetime.datetime.now()

    print("[BLOCK DIAGONALIZATION] Time spent reordering:",later_time - first_time)

    first_time = datetime.datetime.now()

    B_final = [Q.T @ Br @ Q for Br in B]

    # Slices for each block
    block_slices = []
    start = 0
    for sz in block_sizes:
        block_slices.append(slice(start, start + sz))
        start += sz

    # Extract blocks
    B_blocks = []
    for Br in B_final:
        blocks_r = []
        for sl in block_slices:
            blocks_r.append(Br[sl, sl])
        B_blocks.append(blocks_r)

    later_time = datetime.datetime.now()

    print("[BLOCK DIAGONALIZATION] Time spent conjugating into block diagonal form:",later_time - first_time)

    return Q, block_slices, B_blocks


# ============================================================
# Full Step 2 SDP for symmetric Boolean functions
# ============================================================

def adversary_primal_step2_symmetric(
    n,
    f_values,
    solver=None,
    verbose=False,
    eig_tol=1e-9,
    zero_tol=1e-8,
    seed=0,
):
    """
    Numerical Step 2 implementation for the primal adversary SDP
    for a symmetric Boolean function f(x)=g(|x|).

    This:
      - reduces beta[x] -> beta_k
      - reduces Gamma[x,y] -> gamma_{a,b,t} with a<=b and g(a)!=g(b)
      - builds the H=Stab(1)-invariant Hermitian orbit basis
      - numerically block diagonalizes that algebra
      - imposes M_1 >= 0 blockwise
    """

    # ----------------------------------------------------------
    # 1. Normalize domain and outputs
    # ----------------------------------------------------------
    orig_keys = list(f_values.keys())
    domain = [normalize_input_bits(x) for x in orig_keys]
    values01 = [normalize_output(f_values[x]) for x in orig_keys]

    if not domain:
        raise ValueError("Empty domain.")

    for x in domain:
        if len(x) != n:
            raise ValueError(f"Input {x} has length {len(x)} but expected {n}.")

    m = len(domain)

    # ----------------------------------------------------------
    # 2. Layer data
    # ----------------------------------------------------------
    layer_value = {}
    layer_size = defaultdict(int)

    for x, fx in zip(domain, values01):
        k = hamming_weight(x)
        layer_size[k] += 1
        if k in layer_value and layer_value[k] != fx:
            raise ValueError(
                f"Function is not symmetric on the given domain: "
                f"layer {k} has both outputs."
            )
        layer_value[k] = fx

    zero_layers = sorted(k for k, v in layer_value.items() if v == 0)
    one_layers = sorted(k for k, v in layer_value.items() if v == 1)

    if not zero_layers or not one_layers:
        raise ValueError("The function must take both output values on the domain.")

    # ----------------------------------------------------------
    # 3. Reduced beta variables
    # ----------------------------------------------------------
    beta = {
        k: cp.Variable(nonneg=True, name=f"beta_{k}")
        for k in sorted(layer_value)
    }

    # ----------------------------------------------------------
    # 4. Reduced gamma variables
    # Keep only unordered layer pairs with opposite outputs
    # ----------------------------------------------------------

    first_time = datetime.datetime.now()
    gamma_keys = set()
    for x in domain:
        a = hamming_weight(x)
        for y in domain:
            b = hamming_weight(y)
            t = intersection_weight(x, y)
            aa, bb = min(a, b), max(a, b)
            if layer_value[a] != layer_value[b]:
                gamma_keys.add((aa, bb, t))

    gamma_keys = sorted(gamma_keys)
    gamma = {
        key: cp.Variable(name=f"gamma_{key[0]}_{key[1]}_{key[2]}")
        for key in gamma_keys
    }

    def gamma_var(a, b, t):
        aa, bb = min(a, b), max(a, b)
        key = (aa, bb, t)
        return gamma[key] if key in gamma else 0.0
    
    later_time = datetime.datetime.now()

    print("time spent in the domain:",later_time - first_time)

    # ----------------------------------------------------------
    # 5. Build Hermitian H-orbit basis and block diagonalize
    # ----------------------------------------------------------

    first_time = datetime.datetime.now()

    hermitian_keys, B = build_H_hermitian_orbit_basis(domain)

    later_time = datetime.datetime.now()

    print("time spent orbit reducing:",later_time - first_time)

    first_time = datetime.datetime.now()

    Q, block_slices, B_blocks = numerical_block_diagonalize_algebra(
        B,
        eig_tol=eig_tol,
        zero_tol=zero_tol,
        seed=seed,
    )

    later_time = datetime.datetime.now()

    print("time spent block-diagonalizing:",later_time - first_time)

    # ----------------------------------------------------------
    # 6. Express M_1 = sum_u z_u B_u
    # ----------------------------------------------------------
    z = []

    for key in hermitian_keys:
        x1, y1, a, b, t = key

        # Diagonal orbit iff x=y, which for binary strings means:
        # x1=y1 and a=b=t
        is_diagonal_orbit = (x1 == y1) and (a == b == t)
        diag_term = beta[a] if is_diagonal_orbit else 0.0

        # Delta_1 contributes iff first coordinate differs
        delta1 = 1 if x1 != y1 else 0

        if (a in layer_value and b in layer_value and layer_value[a] != layer_value[b] and delta1 == 1):
            gamma_term = gamma_var(a, b, t)
        else:
            gamma_term = 0.0

        z.append(diag_term - gamma_term)

    # ----------------------------------------------------------
    # 7. Constraints
    # ----------------------------------------------------------
    constraints = []

    # Class-mass constraints
    constraints.append(
        cp.sum([layer_size[k] * beta[k] for k in zero_layers]) == 0.5
    )
    constraints.append(
        cp.sum([layer_size[k] * beta[k] for k in one_layers]) == 0.5
    )

    # Blockwise PSD constraints
    num_blocks = len(block_slices)
    for k in range(num_blocks):
        Mk = 0
        for u, zu in enumerate(z):
            Mk = Mk + zu * B_blocks[u][k]
        constraints.append(Mk >> 0)

    # ----------------------------------------------------------
    # 8. Objective
    # maximize sum_{x,y} Gamma[x,y]
    # ----------------------------------------------------------
    objective = 0
    for x in domain:
        a = hamming_weight(x)
        for y in domain:
            b = hamming_weight(y)
            t = intersection_weight(x, y)
            if layer_value[a] != layer_value[b]:
                objective += gamma_var(a, b, t)

    problem = cp.Problem(cp.Maximize(objective), constraints)

    if solver is not None:
        value = problem.solve(solver=solver, verbose=verbose)
    else:
        try:
            value = problem.solve(solver=cp.MOSEK, verbose=verbose)
        except Exception:
            value = problem.solve(solver=cp.SCS, verbose=verbose, eps=1e-6)

    beta_values = {k: beta[k].value for k in beta}
    gamma_values = {key: gamma[key].value for key in gamma}

    return {
        "value": value,
        "status": problem.status,
        "beta": beta_values,
        "gamma": gamma_values,
        "num_blocks": num_blocks,
        "block_sizes": [sl.stop - sl.start for sl in block_slices],
        "Q": Q,
        "hermitian_orbit_keys": hermitian_keys,
        "domain_01": domain,
    }

from math import comb

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


# ============================================================
# Full-layer sizes
# ============================================================

def layer_sizes_full(n, allowed_layers):
    return {k: comb(n, k) for k in allowed_layers}


# ============================================================
# Full S_n pair-orbit count:
# number of ordered pairs (x,y) with |x|=a, |y|=b, |x∩y|=t
# ============================================================

def num_pairs_in_orbit(n, a, b, t):
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
    if not (max(0, a_rest + b_rest - (n - 1)) <= t_rest <= min(a_rest, b_rest)):
        return False
    return True


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


# ============================================================
# Enumerate all H-orbit keys present in a symmetric promise domain
# Domain = union of full Hamming layers in allowed_layers
# ============================================================

def enumerate_H_orbit_keys(n, allowed_layers):
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
# Structural constants p_{rt}^s for ordered H-orbit basis
#
# r = orbit of (x,z): (x1, z1, a, c, u)
# t = orbit of (x,y): (x1, y1, a, b, w)
# s = orbit of (z,y): (z1, y1, c, b, v)
#
# For fixed r,t, only a small family of s are even compatible.
# We exploit that sparsity.
# ============================================================

def structural_constant_H(n, r, s, tkey):
    """
    Return p_{r,t}^s = #{z : (x,z) in R_r and (z,y) in R_s}
    for any fixed (x,y) in orbit R_t.
    """
    x1r, z1r, a_r, c_r, u_r = r
    z1s, y1s, c_s, b_s, v_s = s
    x1t, y1t, a_t, b_t, w_t = tkey

    # Endpoint/middle compatibility
    if x1r != x1t or y1s != y1t:
        return 0
    if z1r != z1s:
        return 0
    if a_r != a_t or b_s != b_t or c_r != c_s:
        return 0

    x1, y1, z1 = x1t, y1t, z1r
    a, b, c = a_t, b_t, c_r
    u, v, w = u_r, v_s, w_t

    # Counts for coordinates 2..n, relative to fixed (x,y) in orbit tkey
    n11 = w - x1 * y1
    n10 = (a - x1) - n11
    n01 = (b - y1) - n11
    n00 = (n - 1) - n11 - n10 - n01

    if min(n11, n10, n01, n00) < 0:
        return 0

    # Remove coordinate 1 contribution from target overlaps
    u_rem = u - x1 * z1
    v_rem = v - z1 * y1
    c_rem = c - z1

    if min(u_rem, v_rem, c_rem) < 0:
        return 0

    total = 0

    # alpha = number of (1,1) positions among coords 2..n where z=1
    alpha_min = max(
        0,
        u_rem - n10,
        v_rem - n01,
        u_rem + v_rem - c_rem
    )
    alpha_max = min(
        n11,
        u_rem,
        v_rem,
        n00 + u_rem + v_rem - c_rem
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
# Sparse precomputation object for Step 1 1/2
# ============================================================

def precompute_step_one_half_sparse(n, allowed_layers):
    """
    Precompute the sparse regular-representation transitions for the H-orbit algebra.

    Returns a dict with:
      - orbit_keys
      - orbit_sizes
      - norms
      - transitions: transitions[r][t] = list of (s, coeff), where
            coeff = (norm_s / norm_t) * p_{r,t}^s
    """
    orbit_keys = enumerate_H_orbit_keys(n, allowed_layers)
    M = len(orbit_keys)

    orbit_sizes = [num_H_pairs_in_orbit(n, key) for key in orbit_keys]
    norms = [size ** 0.5 for size in orbit_sizes]

    # Index candidate s-orbits by (z1, y1, c, b)
    # s = (z1, y1, c, b, v)
    s_index = defaultdict(list)
    for s_idx, key in enumerate(orbit_keys):
        z1, y1, c, b, v = key
        s_index[(z1, y1, c, b)].append((v, s_idx))

    transitions = [defaultdict(list) for _ in range(M)]

    # For each (r,t), only a tiny family of s can be nonzero.
    for r_idx, r in enumerate(orbit_keys):
        x1r, z1r, a_r, c_r, u_r = r

        for t_idx, tkey in enumerate(orbit_keys):
            x1t, y1t, a_t, b_t, w_t = tkey

            # Quick compatibility checks
            if x1r != x1t:
                continue
            if a_r != a_t:
                continue

            # Candidate s must have (z1, y1, c, b) = (z1r, y1t, c_r, b_t)
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

    return {
        "n": n,
        "allowed_layers": tuple(sorted(allowed_layers)),
        "orbit_keys": orbit_keys,
        "orbit_sizes": orbit_sizes,
        "norms": norms,
        "transitions": transitions,
        "regular_rep_size": M,
    }


# ============================================================
# Solve SDP from sparse precomputation
# ============================================================

def adversary_primal_step_one_half_sparse(
    n,
    f_values,
    solver=None,
    verbose=False,
    precomp=None,
):
    """
    Sparse combinatorial Step 1 1/2 implementation for the primal adversary SDP.

    Assumptions:
      - f is symmetric on its promise domain
      - the promise domain is a union of full Hamming layers
      - inputs may be in {0,1}^n or {-1,1}^n

    Parameters
    ----------
    precomp : dict or None
        Output of precompute_step_one_half_sparse(n, allowed_layers).
        If provided, reuses the sparse regular-representation data.
    """

    # ----------------------------------------------------------
    # 1. Normalize domain and verify symmetry-by-layer
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

    # Verify the promise domain is a union of full layers
    for k in allowed_layers:
        if layer_count_seen[k] != full_layer_sizes[k]:
            raise ValueError(
                f"Layer {k} is incomplete: saw {layer_count_seen[k]} points but full layer has {full_layer_sizes[k]}. "
                "This sparse combinatorial implementation assumes the promise domain is a union of full Hamming layers."
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
    # 3. Sparse Step 1 1/2 precomputation
    # ----------------------------------------------------------

    first_time = datetime.datetime.now()

    if precomp is None:
        precomp = precompute_step_one_half_sparse(n, allowed_layers)
    else:
        if precomp["n"] != n:
            raise ValueError("Precomputation n does not match.")
        if tuple(sorted(allowed_layers)) != tuple(precomp["allowed_layers"]):
            raise ValueError("Precomputation allowed_layers do not match current function.")

    orbit_keys = precomp["orbit_keys"]
    transitions = precomp["transitions"]
    M = precomp["regular_rep_size"]

    later_time = datetime.datetime.now()

    print("time spent building the regular representation:",later_time - first_time)

    # ----------------------------------------------------------
    # 4. Expand M_1 = sum_r z_r C_r in ordered H-orbit basis
    # ----------------------------------------------------------

    first_time = datetime.datetime.now()

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

    later_time = datetime.datetime.now()

    print("time spent expanding the sum in ordered basis:",later_time - first_time)

    # ----------------------------------------------------------
    # 5. Constraints
    # ----------------------------------------------------------
    constraints = []

    # class-mass constraints
    constraints.append(cp.sum([full_layer_sizes[k] * beta[k] for k in zero_layers]) == 0.5)
    constraints.append(cp.sum([full_layer_sizes[k] * beta[k] for k in one_layers]) == 0.5)

    # Sparse assembly of reduced PSD matrix:
    # Mred[s,t] = sum_r z_r * (||C_s||/||C_t||) * p_{r,t}^s
    Mred_entries = [[0 for _ in range(M)] for __ in range(M)]

    first_time = datetime.datetime.now()

    for r_idx, zr in enumerate(z):
        trans_r = transitions[r_idx]
        if not trans_r:
            continue
        for t_idx, items in trans_r.items():
            for s_idx, coeff in items:
                Mred_entries[s_idx][t_idx] = Mred_entries[s_idx][t_idx] + zr * coeff

    Mred = cp.bmat(Mred_entries)
    constraints.append(Mred >> 0)

    later_time = datetime.datetime.now()

    print("time spent reducing the size of the matrix:",later_time - first_time)  ### CAN BE OPTIMIZED ###

    # ----------------------------------------------------------
    # 6. Objective assembled combinatorially
    # ----------------------------------------------------------

    first_time = datetime.datetime.now()

    objective_expr = 0
    for (a, b, t), var in gamma.items():
        if a == b:
            mult = num_pairs_in_orbit(n, a, b, t)
        else:
            mult = num_pairs_in_orbit(n, a, b, t) + num_pairs_in_orbit(n, b, a, t)
        objective_expr += mult * var

    later_time = datetime.datetime.now()

    print("time spent assembling the objective function:",later_time - first_time)

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
        "regular_rep_size": M,
        "orbit_keys_H": orbit_keys,
        "precomp": precomp,
    }



##### ONLY FOR TOTAL FUNCTIONS #####

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

def adversary_primal_step_one_half_sparse_blocks(
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
