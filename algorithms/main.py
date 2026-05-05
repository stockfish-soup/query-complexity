from adversary_transducer import *
import qiskit

n = 4

# OR on n bits
inputs = all_binary_inputs(n)
outputs = [int(any(x)) for x in inputs]

sol = solve_general_adversary_sdp(
    inputs,
    outputs,
    solver="MOSEK",
)

epsilon = 0.5

transducer = compute_transducer_unitary(sol)
W = catalyst_norm_bound(transducer)
K = choose_repetition_count(W, epsilon)
print("Adv objective:", sol.objective_value)
print("catalyst W:", W)
print("K:", K)
print("theoretical vector-error bound <=", 2 * np.sqrt(W / K))
res = show_W_size(sol, assume_boolean_query_register=True)

for key in res.keys():
    print(key, res[key])

"""

for x, y in zip(sol.inputs, sol.outputs):
    result = run_repeated_transducer_algorithm(transducer, x, K)
    output = np.asarray(result["output_slot"])
    probs = output_probabilities_from_slot(output)
    print("x=", x, "f(x)=", y, "output probabilities=", probs, "garbage norm=", result["garbage_norm"])


"""