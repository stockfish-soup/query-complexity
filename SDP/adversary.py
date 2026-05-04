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
    