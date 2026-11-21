import numpy as np

import query_complexity as qc


def test_trace_minimized_dual_respects_relaxed_cap():
    n = 2
    inputs = qc.all_binary_inputs(n)
    outputs = [int(any(x)) for x in inputs]
    phase_adversary = qc.threshold_adversary_value(n, 1) / 2.0
    alpha = n / (2.0 * phase_adversary)
    solution = qc.solve_trace_minimized_dual(
        inputs,
        outputs,
        alpha=alpha,
        adversary_value=phase_adversary,
        epsilon=1e-6,
        solver="CLARABEL",
        psd_tol=1e-6,
    )

    assert max(solution.input_costs) <= alpha * phase_adversary + 2e-6
    assert all(np.linalg.eigvalsh(gram).min() >= -1e-7 for gram in solution.grams)


def test_symmetric_block_rank_matches_expanded_rank():
    n, k = 4, 2
    phase_adversary = qc.threshold_adversary_value(n, k) / 2.0
    orbit = qc.solve_symmetric_trace_minimized_dual(
        n,
        qc.THRESHOLD_layers(n, k),
        alpha=1.2,
        adversary_value=phase_adversary,
        epsilon=1e-6,
        solver="CLARABEL",
    )
    compact_rank = qc.symmetric_witness_rank(
        orbit,
        relative_tol=1e-6,
        absolute_tol=1e-7,
    )["rank_X_i"]
    expanded = qc.expand_orbit_solution(orbit, psd_tol=1e-6)

    assert len(set(expanded.witness_dims)) == 1
    assert compact_rank == expanded.witness_dims[0]


def test_trace_is_nonincreasing_with_alpha():
    n, k = 4, 2
    phase_adversary = qc.threshold_adversary_value(n, k) / 2.0
    traces = []
    for alpha in (1.0, 1.2, n / (2.0 * phase_adversary)):
        orbit = qc.solve_symmetric_trace_minimized_dual(
            n,
            qc.THRESHOLD_layers(n, k),
            alpha=alpha,
            adversary_value=phase_adversary,
            epsilon=1e-6,
            solver="CLARABEL",
        )
        traces.append(orbit.trace_value)

    assert traces[1] <= traces[0] + 1e-5
    assert traces[2] <= traces[1] + 1e-5


def test_nonsymmetric_rank_two_threshold_candidate():
    n, k = 4, 2
    inputs = qc.all_binary_inputs(n)
    outputs = [int(sum(x) >= k) for x in inputs]
    phase_adversary = qc.threshold_adversary_value(n, k) / 2.0
    result = qc.solve_fixed_rank_dual_feasibility(
        inputs,
        outputs,
        rank=2,
        alpha=1.1,
        adversary_value=phase_adversary,
        restarts=2,
        random_seed=4,
        max_nfev=1000,
        feasibility_tol=1e-7,
    )

    assert result.feasible
    assert result.max_equality_residual <= 1e-7
    assert result.max_cap_violation <= 1e-7
    assert all(np.linalg.matrix_rank(gram, tol=1e-7) <= 2 for gram in result.grams)

    solution = qc.fixed_rank_result_to_adversary_solution(result, feasibility_tol=1e-7)
    transducer = qc.build_adversary_transducer(solution)
    assert transducer.map_error <= 1e-6
    assert transducer.unitarity_error <= 1e-6
