# Python API reference

All three packages are installed by the `query-complexity` distribution.
The functions' docstrings provide full signatures and keyword options.

## `query_complexity`

| Entry point | Purpose |
| --- | --- |
| `analytic_or_solution(n)` | Analytic phase-normalized OR witness |
| `all_binary_inputs(n)` | Enumerate Boolean inputs |
| `OR_layers`, `AND_layers`, `EXACT_layers`, `THRESHOLD_layers`, `PARITY_layers` | Symmetric output profiles |
| `solve_adversary_sdp(inputs, outputs, ...)` | General Boolean adversary witness |
| `solve_symmetric_value(n, layer_values, ...)` | Symmetric adversary value SDP |
| `solve_symmetric_orbit_sdp(n, layer_values, ...)` | Compact symmetric witness blocks |
| `solve_symmetric_adversary_sdp(n, layer_values, ...)` | Expanded symmetric witness |
| `solve_trace_minimized_dual(...)` | Trace minimization under a relaxed cost cap |
| `solve_symmetric_trace_minimized_dual(...)` | Symmetric trace minimization |
| `solve_fixed_rank_dual_feasibility(...)` | Nonconvex fixed-rank search |
| `solve_symmetric_fixed_rank_dual_feasibility(...)` | Symmetric fixed-rank search |
| `symmetric_witness_rank(...)` | Recover witness rank from compact blocks |
| `build_adversary_transducer(solution, ...)` | Extract the input-independent unitary |
| `run_adversary_algorithm(transducer, x, K, ...)` | Simulate repeated transduction |
| `threshold_eta_blocks(n, k)` | Analytic threshold block factors |
| `qiskit_adversary_circuit(...)` | Optional dense Qiskit circuit export |

Results include dataclasses `AdversarySolution`, `OrbitSolution`,
`SdpValueResult`, `AdversaryTransducer` and feasibility-search results.
Core implementation is in `query_complexity.core`; public imports from
`query_complexity` are preserved.

## `junta_learning`

Import submodules explicitly, for example `from junta_learning import rank_one`.

| Submodule / entry point | Purpose |
| --- | --- |
| `majority_sdp.solve(k)` | Return `(cost, count_psd_matrix, status)` for odd k |
| `majority_sdp.solve_cross_block(k)` | Equivalent nuclear-norm optimization |
| `rank_one.count_witness(k)` | Explicit period-four compact count factor |
| `rank_one.exact_cost(k)`, `exact_cost_squared(k)` | Explicit cost; squared value uses rational arithmetic |
| `rank_one.exhaustive_best(k)` | Finite half-set search, combinatorial in k |
| `bounded_size.solve(k, ...)` | Global PSD optimization across support-size layers |
| `multibias.solve_general_level0_cvxpy(...)` | Arbitrary symmetric profile and finite bias set |
| `multibias.solve_exact_primal_cvxpy(...)` | Specialized finite-bias EXACT SOCP |
| `multibias.exchange_exact_continuum(...)` | Numerical bias exchange, using a scanned continuum envelope |
| `witness_audit.construct_witness(k)` | Earlier dyadic-current construction |
| `witness_audit.explicit_dual_residual(k, n)` | Small original-SDP enumeration |

## `sdp_hierarchy`

- `OracleProblem(answers, generators, targets=None)` defines an explicit
  binary oracle table, group action and desired target equivalence.
- `Hierarchy(problem, max_order=10000, decomposer=..., sector_order=None)`
  prepares query orbits and multiplicity PSD blocks.
- `Hierarchy.solve(level=0, solver="CLARABEL", dense=False,
  intersection_free=False, validation_tol=1e-6)` solves a prefix or independent
  reference formulation. An invalid level is rejected. Prefixes may be infeasible.
- `Hierarchy.block_summary()` reports sector labels, degrees and multiplicities.
- `symmetric_evaluation(n, profile)`, `symmetric_junta(n, k, profile)` and
  `triangle_finding(n)` build example problems.

Solve results include `cost`, `status`, `solver`, `block_sizes`, and
`numerically_feasible`. When a witness exists, they also include reconstructed
`matrices`, equality residual, minimum eigenvalue, maximum load, support
leakage and covariance residual. Dense references report no support leakage.
The representation backend lives in `sdp_hierarchy.representation`.

See the guides for the mathematical definitions; a numerical status or
validation flag is not an exact certificate.
