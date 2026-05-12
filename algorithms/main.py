from adversary_transducer import *
import qiskit

n = 4
k = 2

# OR on n bits
#outputs = [int(sum(x) == n / 2) for x in inputs]
#outputs = [sum(x) % 2 for x in inputs]

edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
edge_index = {e: idx for idx, e in enumerate(edges)}

inputs = all_binary_inputs(len(edges))
outputs = [
    int(any(
        x[edge_index[(i, j)]] and
        x[edge_index[(i, k)]] and
        x[edge_index[(j, k)]]
        for i in range(n)
        for j in range(i + 1, n)
        for k in range(j + 1, n)
    ))
    for x in inputs
]

inputs = all_binary_inputs(n)
outputs = [int(any(x)) for x in inputs]
#outputs = [sum(x) >= k for x in inputs]
#outputs = [sum(x) == k for x in inputs]
#outputs = [sum(x) % 2 for x in inputs]


"""sol=solve_general_adversary_sdp(
    inputs,
    outputs,
    solver="MOSEK",
    psd_factor_tol=0.1**10
)
"""

"""
sol = solve_adversary_min_rank(
    inputs,
    outputs,
    solver="MOSEK",
    psd_factor_tol=0.1**5
)"""

"""
sol = solve_adversary_min_rank_one_vector(
    inputs,
    outputs,
    solver="MOSEK",
    psd_factor_tol=0.1**5,
)"""

sol = solve_adversary_min_rank_one_vector_direct_sum(
    inputs,
    outputs,
    solver="MOSEK",
    psd_factor_tol=0.1**5,
)

td = compute_one_vector_direct_sum_transducer_unitary(sol)

print("sum objective:", sol.objective_value)
print("adversary cap:", sol.adversary_cap)
print("local dim W_i:", sol.witness_dims)
print("dim L:", sol.private_dim)
print("Gram error:", td.gram_error)
print("map error:", td.map_error)
print("unitarity error:", td.unitarity_error)

for x_idx in range(len(sol.inputs)):
    print(x_idx, verify_one_vector_direct_sum_transduction(td, x_idx))
#sol_sdp = solve_general_adversary_sdp(inputs, outputs)


epsilon = 0.5

transducer = compute_one_vector_direct_sum_transducer_unitary(sol)

W = catalyst_norm_bound_one_vector_direct_sum(transducer)
K = choose_repetition_count(W, epsilon)

print("sum objective:", sol.objective_value)
print("adversary cap:", sol.adversary_cap)
print("catalyst W:", W)
print("K:", K)
print("theoretical vector-error bound <=", 2 * np.sqrt(W / K))

show_W_size_one_vector_direct_sum(sol)

summary = summarize_one_vector_direct_sum_algorithm(sol, epsilon=0.5)

transducer = summary["transducer"]
K = summary["K"]

for x_idx, x in enumerate(sol.inputs):
    result = run_repeated_one_vector_direct_sum_algorithm(
        transducer,
        x_idx,
        K,
    )

    print()
    print("x =", x)
    print("f(x) =", sol.outputs[x_idx])
    print("output probabilities:", result["output_probabilities"])
    print("target error:", result["target_error"])
    print("garbage norm:", result["garbage_norm"])




"""
epsilon = 0.5

transducer = compute_transducer_unitary(sol)
W = catalyst_norm_bound(transducer)
K = choose_repetition_count(W, epsilon)
print("Adv objective:", sol.objective_value)
print("catalyst W:", W)
print("K:", K)
print("theoretical vector-error bound <=", 2 * np.sqrt(W / K))
show_W_size(sol, assume_boolean_query_register=True)


for x, y in zip(sol.inputs, sol.outputs):
    result = run_repeated_transducer_algorithm(transducer, x, K)
    output = np.asarray(result["output_slot"])
    probs = output_probabilities_from_slot(output)
    print("x=", x, "f(x)=", y, "output probabilities=", probs, "garbage norm=", result["garbage_norm"])
"""




