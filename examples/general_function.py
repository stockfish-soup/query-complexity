"""Solve a small nonsymmetric Boolean evaluation problem with Clarabel."""
import query_complexity as qc


def main():
    inputs = qc.all_binary_inputs(3)
    # f(x) = x0 AND (x1 OR x2).
    outputs = [int(x[0] and (x[1] or x[2])) for x in inputs]
    solution = qc.solve_adversary_sdp(inputs, outputs, solver="CLARABEL")
    print(f"Phase-normalized objective: {solution.objective_value:.8f}")
    transducer = qc.build_adversary_transducer(solution)
    print(f"Transducer mapping error: {transducer.map_error:.3g}")


if __name__ == "__main__":
    main()
