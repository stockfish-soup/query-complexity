import itertools
from tensor1 import tensor1
from tensor2 import tensor2t, tensor2eps
from adversary import adversary_primal, adversary_dual, adversary_primal_orbit_reduced, adversary_primal_step2_symmetric, adversary_primal_step_one_half_symmetric
import numpy as np
import cvxpy as cp


def f_or(n):
    return {
        x: (1.0 if any(xi == 1 for xi in x) else -1.0)
        for x in itertools.product([-1, 1], repeat=n)
    }

def f_or01(n: int) -> dict:
    return {
        x: int(any(x))
        for x in itertools.product([0, 1], repeat=n)
    }


def f_deutsch_jozsa(m: int):
    """
    x in {-1,1}^N, N = 2^m
    f(x) = +1 for constant
    f(x) = -1 for balanced

    Returns only promised inputs.
    """
    N = 2 ** m
    f_values = {}

    # constant inputs
    f_values[tuple([1] * N)] = 1.0
    f_values[tuple([-1] * N)] = 1.0

    # balanced inputs
    for plus_positions in itertools.combinations(range(N), N // 2):
        x = [-1] * N
        for i in plus_positions:
            x[i] = 1
        f_values[tuple(x)] = -1.0

    return f_values

## First method examples ##

m = 3
n = 2 ** m

n = 10

t = 1

f_values = f_or(n)

#f_values = f_deutsch_jozsa(m)



#value = adversary_primal(n,f_values,verbose = True)


#print(value[0])


value = adversary_primal_orbit_reduced(n,f_values,verbose = True)

#value = adversary_primal_step2_symmetric(n,f_values,verbose = True)

#value = adversary_primal_step_one_half_symmetric(n,f_values,verbose = True)


"""

f_values = f_deutsch_jozsa(m)
f_values = f_or(n)


prob, vars_ = tensor1(
    n=n,
    t=t,
    f_values=f_values,
)

sm = prob.size_metrics
print("num_scalar_variables =", sm.num_scalar_variables)
print("num_scalar_eq_constr =", sm.num_scalar_eq_constr)
print("num_scalar_leq_constr =", sm.num_scalar_leq_constr)
print("max_data_dimension =", sm.max_data_dimension)
print("total constraint objects:", len(prob.constraints))

prob.solve(
    solver=cp.SCS,
    verbose=True,
    max_iters=20000,
    eps=1e-2,
    canon_backend=cp.SCIPY_CANON_BACKEND
)

print(n, vars_["eps"].value)




n = 10
t = 2
f_values = f_or(n)

out = tensor2eps(
    n=n,
    t=t,
    f_values=f_values,
    solver=cp.SCS,
    verbose=True,
    solver_opts={
        "eps": 1e-4,
        "max_iters": 10000,
    },
)

print("\n=== OR example ===")
print("status:", out["status"])
print("optimal eps:", out["eps"])
print("largest PSD block:", out["largest_psd_block"])
print("A constraints:", out["A_constraints"])
print(out["size_metrics"])

# Example 2: Deutsch-Jozsa, same style as first code
m = 4
n = 2 ** m
t = 1
f_values = f_deutsch_jozsa(m)

out = tensor2eps(
    n=n,
    t=t,
    f_values=f_values,
    solver=cp.SCS,
    verbose=True,
    solver_opts={
        "eps": 1e-5,
        "max_iters": 15000,
    },
)

print("\n=== Deutsch-Jozsa example ===")
print("status:", out["status"])
print("optimal eps:", out["eps"])
print("largest PSD block:", out["largest_psd_block"])
print("A constraints:", out["A_constraints"])
print(out["size_metrics"])





## Second method examples ##

def print_scan_results(title: str, out: dict) -> None:
    print(f"\n=== {title} ===")
    for row in out["results"]:
        print(
            f"degree={row['degree']:>2d}  "
            f"split={row['split']:>2d}  "
            f"Q<= {row['query_upper_bound']:>2d}  "
            f"eps={row['eps_alg']}  "
            f"status={row['status']}  "
            f"vars={row['num_scalar_variables']}  "
            f"ineq={row['num_scalar_ineq']}  "
            f"PSDdim={row['largest_psd_block']}  "
            f"Aeq={row['A_constraints']}"
        )

    print("cb-degree found:", out["cb_degree"])
    print("query upper bound:", out["query_complexity_upper_bound"])

# Example 1: OR 

n_or = 2
f_values_or = f_or(n_or)

out_or = tensor2(
    n=n_or,
    f_values=f_values_or,
    eps_target=1/3,
    max_degree=4,
    solver=cp.SCS,
    verbose=False,
    solver_opts={
        "eps": 1e-4,
        "max_iters": 10000,
    },
)

print_scan_results(f"OR on n={n_or}", out_or)


# Example 2: Deutsch-Jozsa 

m_dj = 2
n_dj = 2 ** m_dj
f_values_dj = f_deutsch_jozsa(m_dj)

out_dj = tensor2(
    n=n_dj,
    f_values=f_values_dj,
    eps_target=1e-4,
    max_degree=4,
    solver=cp.SCS,
    verbose=False,
    solver_opts={
        "eps": 1e-5,
        "max_iters": 15000,
    },
)

print_scan_results(f"Deutsch-Jozsa with oracle size m={m_dj} (ambient n={n_dj})", out_dj)


for i in range(1,10):

    n_or = i
    f_values_or = f_or(n_or)

    out = tensor2(
        n=n_or,
        f_values=f_values_or,
        eps_target=1/3,
        max_degree=4,
        solver=cp.SCS,
        verbose=False,
        solver_opts={
            "eps": 1e-4,
            "max_iters": 10000,
        },
    )

    print("n = " + str(i) + " t_opt = " + str(out["query_complexity_upper_bound"]))

    """