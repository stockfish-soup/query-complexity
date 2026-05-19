from adversary_transducer import *
import qiskit
from matplotlib import pyplot as plt
import numpy as np
from terwilliger_state_conversion import *

from qiskit.quantum_info import Statevector

n = 2
k = 2

def OR_layers(n: int) -> LayerValue:
    return {k: int(k > 0) for k in range(n + 1)}


def AND_layers(n: int) -> LayerValue:
    return {k: int(k == n) for k in range(n + 1)}


def EXACT_layers(n: int, r: int) -> LayerValue:
    return {k: int(k == int(r)) for k in range(n + 1)}


def THRESHOLD_layers(n: int, r: int) -> LayerValue:
    return {k: int(k >= int(r)) for k in range(n + 1)}


def PARITY_layers(n: int) -> LayerValue:
    return {k: k % 2 for k in range(n + 1)}

sol = solve_symmetric_layers(
    n,
    OR_layers(n),
    solver="MOSEK",
    psd_factor_tol=0.1**5,
    epsilon=0.1**5,
)


print("done")

td = compute_one_vector_direct_sum_transducer_unitary(sol)

print("done")

print(td.public_dim)

show_W_size_one_vector_direct_sum(sol)

"""

print(td.catalyst_norms)

S = td.source_states
T = td.target_states

print(np.linalg.norm((td.U@S-T))) #error

print(td.U)

plt.plot(S,T)
plt.show()

"""
print("done")

epsilon = 0.9
W = catalyst_norm_bound_one_vector_direct_sum(td)
K = choose_repetition_count(W, epsilon)

print("sum objective:", sol.objective_value)
print("adversary cap:", sol.adversary_cap)
print("catalyst W:", W)
print("K:", K)
print("theoretical vector-error bound <=", 2 * np.sqrt(W / K))

for x_idx, x in enumerate(sol.inputs):
    result = run_repeated_one_vector_direct_sum_algorithm(
        td,
        x_idx,
        K,
    )

    print()
    print("x =", x)
    print("f(x) =", sol.outputs[x_idx])
    print("output probabilities:", result["output_probabilities"])
    print("target error:", result["target_error"])
    print("garbage norm:", result["garbage_norm"])



W = catalyst_norm_bound_one_vector_direct_sum(td)
K = choose_repetition_count(W, epsilon)

x_idx = 1

qc, meta = qiskit_circuit_for_one_vector_direct_sum_input(
    td,
    x_idx=x_idx,
    K=K,
    measure_all=False,
)

print(qc)
print(meta)

sv = Statevector.from_instruction(qc)

decoded = logical_output_probabilities_from_statevector(
    sv,
    public_dim=meta["public_dim"],
    logical_dim=meta["logical_dim"],
)

print("input x:", sol.inputs[x_idx])
print("true f(x):", sol.outputs[x_idx])
print(decoded)