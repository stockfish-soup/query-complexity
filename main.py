import itertools
from tensor1 import tensor1
from tensor2 import tensor2t, tensor2eps
from adversary import adversary_primal, adversary_dual, adversary_primal_orbit_reduced, adversary_primal_step2_symmetric, adversary_primal_step_one_half_symmetric,  adversary_primal_step_one_half_sparse, adversary_primal_step_one_half_sparse_blocks
from adversary_symmetry_reduced import adversary_symmetry_reduced
from functions import f_or, f_deutsch_jozsa
import numpy as np
import cvxpy as cp
from matplotlib import pyplot as plt
import datetime

## First method examples ##

m = 2
n = 2 ** m

n = 19

t = 1

f_values = f_or(n)

#f_values = f_deutsch_jozsa(m)



#value = adversary_primal(n,f_values,verbose = True)


#print(value[0])


#value = adversary_primal_orbit_reduced(n,f_values,verbose = True)

#value = adversary_primal_step2_symmetric(n,f_values,verbose = True)

#value = adversary_primal_step_one_half_symmetric(n,f_values,verbose = True)

#value = adversary_primal_step_one_half_sparse(n,f_values,verbose = True)

#value = adversary_primal_step_one_half_sparse_blocks(n,f_values,verbose = True)

#print(value)


"""

### TEST FOR 1....N ###

N = 20

L = []



for n in range(1,N+1):

    first_time = datetime.datetime.now()

    f_values = f_or(n)

    value = adversary_primal_step_one_half_sparse_blocks(n,f_values,verbose = False)

    L.append(value["value"])

    later_time = datetime.datetime.now()

    print("Time spent solving for N =",n,":",later_time - first_time)

    print("Optimal value:",value["value"])



plt.plot(L)

plt.show()
"""

n = 10

f_values = f_or(n)

# Solve OR_3
res = adversary_symmetry_reduced(
    n=n,
    f_values=f_values,
    verbose=False,
)

print("status:", res["status"])
print("value:", res["value"])
print("regular representation size:", res["regular_rep_size"])
print("num blocks:", res["num_blocks"])
print("block sizes:", res["block_sizes"])
print("beta:", res["beta"])
print("gamma:", res["gamma"])



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