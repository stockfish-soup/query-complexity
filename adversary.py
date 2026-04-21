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

import cvxpy as cp
from math import comb
from collections import defaultdict


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
    t_rest = t - xy1
    a_rest = a - x1
    b_rest = b - y1
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
    t_rest = t - xy1
    a_rest = a - x1
    b_rest = b - y1

    return comb(n - 1, a_rest) * comb(a_rest, t_rest) * comb((n - 1) - a_rest, b_rest - t_rest)


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
# Combinatorial structural constants for the H-orbit basis
#
# If r=(x1,z1,a,c,u), s=(z1,y1,c,b,v), t=(x1,y1,a,b,w),
# then p_{rs}^t counts the number of intermediate z consistent with both orbit
# conditions relative to a fixed pair (x,y) in orbit t.
# ============================================================

def structural_constant_H(n, r, s, tkey):
    x1r, z1r, a_r, c_r, u_r = r
    z1s, y1s, c_s, b_s, v_s = s
    x1t, y1t, a_t, b_t, w_t = tkey

    # compatibility of endpoints and middle layer
    if x1r != x1t or y1s != y1t:
        return 0
    if z1r != z1s:
        return 0
    if a_r != a_t or b_s != b_t or c_r != c_s:
        return 0

    x1, y1, z1 = x1t, y1t, z1r
    a, b, c = a_t, b_t, c_r
    u, v, w = u_r, v_s, w_t

    # counts on coordinates 2..n for fixed (x,y) in orbit tkey
    n11 = w - x1 * y1
    n10 = (a - x1) - n11
    n01 = (b - y1) - n11
    n00 = (n - 1) - n11 - n10 - n01

    if min(n11, n10, n01, n00) < 0:
        return 0

    # remaining overlaps after removing coordinate 1
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
# Build regular representation combinatorially
# ============================================================

def build_regular_representation_H(n, allowed_layers):
    """
    Returns:
      orbit_keys : ordered H-orbit keys
      orbit_sizes : list of ||C_r||^2 = size of orbit
      L_mats : list of regular-representation matrices L(C_r)
    """
    orbit_keys = enumerate_H_orbit_keys(n, allowed_layers)
    M = len(orbit_keys)

    orbit_sizes = [num_H_pairs_in_orbit(n, key) for key in orbit_keys]
    norms = [size ** 0.5 for size in orbit_sizes]

    first_time = datetime.datetime.now()

    # structural constants p[r][s][t]
    p = [[[0] * M for _ in range(M)] for __ in range(M)]
    for r in range(M):
        for s in range(M):
            for t in range(M):
                p[r][s][t] = structural_constant_H(n, orbit_keys[r], orbit_keys[s], orbit_keys[t])
                print(r,s,t,p[r][s][t])

    later_time = datetime.datetime.now()

    print("[REGULAR REPRESENTATION] time spent computing the p's:",later_time - first_time)

    first_time = datetime.datetime.now()

    # regular representation matrices
    L_mats = []
    for r in range(M):
        Lr = [[0.0] * M for _ in range(M)]
        for s in range(M):
            for t in range(M):
                if norms[t] == 0:
                    raise ValueError("Zero orbit norm encountered.")
                Lr[s][t] = (norms[s] / norms[t]) * p[r][t][s]
        L_mats.append(Lr)

    later_time = datetime.datetime.now()

    print("[REGULAR REPRESENTATION] time spent computing the representation matrices:",later_time - first_time)

    return orbit_keys, orbit_sizes, L_mats


# ============================================================
# Step 1 1/2 SDP for symmetric promise functions
# ============================================================

def adversary_primal_step_one_half_combinatorial(
    n,
    f_values,
    solver=None,
    verbose=False,
):
    """
    Combinatorial Step 1 1/2 implementation for the primal adversary SDP.

    Assumptions:
      - f is symmetric on its promise domain
      - the promise domain is a union of full Hamming layers
      - inputs may be in {0,1}^n or {-1,1}^n

    Returns a dict with the SDP solution and precomputed algebra data.
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
            raise ValueError(f"Function is not symmetric on the given domain: layer {k} has both outputs.")
        layer_value[k] = fx

    allowed_layers = sorted(layer_value.keys())
    full_layer_sizes = layer_sizes_full(n, allowed_layers)

    # verify promise domain is union of full layers
    for k in allowed_layers:
        if layer_count_seen[k] != full_layer_sizes[k]:
            raise ValueError(
                f"Layer {k} is incomplete: saw {layer_count_seen[k]} points but full layer has {full_layer_sizes[k]}. "
                "This combinatorial implementation assumes the promise domain is a union of full Hamming layers."
            )

    zero_layers = sorted(k for k, v in layer_value.items() if v == 0)
    one_layers = sorted(k for k, v in layer_value.items() if v == 1)

    if not zero_layers or not one_layers:
        raise ValueError("The function must take both output values on the domain.")

    # ----------------------------------------------------------
    # 2. Reduced variables
    # ----------------------------------------------------------
    beta = {k: cp.Variable(nonneg=True, name=f"beta_{k}") for k in allowed_layers}

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
    gamma = {key: cp.Variable(name=f"gamma_{key[0]}_{key[1]}_{key[2]}") for key in gamma_keys}

    def gamma_var(a, b, t):
        aa, bb = min(a, b), max(a, b)
        key = (aa, bb, t)
        return gamma[key] if key in gamma else 0.0

    # ----------------------------------------------------------
    # 3. Precompute H-orbit algebra combinatorially
    # ----------------------------------------------------------

    first_time = datetime.datetime.now()

    orbit_keys, orbit_sizes, L_mats = build_regular_representation_H(n, allowed_layers)
    M = len(orbit_keys)

    later_time = datetime.datetime.now()

    print("time spent building the regular representation:",later_time - first_time)

    # ----------------------------------------------------------
    # 4. Express M_1 = diag(beta) - Gamma o Delta_1 in ordered H-orbit basis
    #    M_1 = sum_r z_r C_r
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

    constraints.append(cp.sum([full_layer_sizes[k] * beta[k] for k in zero_layers]) == 0.5)
    constraints.append(cp.sum([full_layer_sizes[k] * beta[k] for k in one_layers]) == 0.5)

    # Step 1 1/2 PSD constraint
    Mred = 0
    for zr, Lr in zip(z, L_mats):
        Mred += zr * cp.Constant(Lr)
    constraints.append(Mred >> 0)

    # ----------------------------------------------------------
    # 6. Objective, combinatorially
    #    sum_{x,y} Gamma[x,y]
    # ----------------------------------------------------------
    objective_expr = 0
    for (a, b, t), var in gamma.items():
        if a == b:
            mult = num_pairs_in_orbit(n, a, b, t)
            objective_expr += mult * var
        else:
            # because we stored only unordered (a,b), but objective is ordered over all pairs
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
        "orbit_keys_H": orbit_keys,
        "orbit_sizes_H": orbit_sizes,
        "regular_rep_size": M,
        "allowed_layers": allowed_layers,
    }