from adversary_transducer import (
    all_binary_inputs,
    solve_general_adversary_sdp,
    compute_transducer_unitary,
    verify_adversary_constraints,
)

n = 5

# OR on n bits
inputs = all_binary_inputs(n)
outputs = [int(any(x)) for x in inputs]

sol = solve_general_adversary_sdp(
    inputs,
    outputs,
    solver="MOSEK",
)

transducer = compute_transducer_unitary(sol)

U = transducer.U

print(sol.objective_value)
print(verify_adversary_constraints(sol))
print(transducer.gram_error)
print(transducer.map_error)
print(transducer.unitarity_error)