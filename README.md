# query_complexity_clean

A compact rewrite of the uploaded query-complexity codebase. It keeps only the
space-efficient **phase-oracle one-vector transducer** path.

## What is kept

- `solve_phase_sdp(...)`: one-vector phase SDP for arbitrary Boolean domains.
- `solve_symmetric_orbit_sdp(...)`: Terwilliger/orbit-reduced phase SDP for symmetric functions.
- `solve_symmetric_phase_sdp(...)`: convenience wrapper that solves the orbit SDP and expands to the Boolean input basis.
- `build_phase_transducer(...)`: extracts the input-independent transducer unitary `U`.
- `run_phase_algorithm(...)`: simulates the repeated transducer algorithm.
- `threshold_eta_blocks(...)`: analytic rank-one Schrijver blocks for `THRESHOLD^k_n`.
- Optional `qiskit_phase_circuit(...)` for dense Qiskit simulation.

## What was removed

- direct-sum / bit-flip transducer code;
- two-vector adversary SDP;
- primal-only experiments;
- polynomial-method experiments;
- duplicate `main.py` scripts;
- rescaled and triangle-specific variants.

## Quick OR example

```python
import query_complexity as qc

sol = qc.analytic_or_solution(3)
td = qc.build_phase_transducer(sol)

print("objective:", sol.objective_value)
print("private dim:", td.private_dim)
print("map error:", td.map_error)
qc.show_catalysts(sol)
```

## Symmetric SDP example

```python
import query_complexity as qc

n, k = 4, 2
sol = qc.solve_symmetric_phase_sdp(
    n,
    qc.THRESHOLD_layers(n, k),
    solver="MOSEK",          # or omit to use an installed default
    epsilon=1e-5,
    psd_tol=1e-5,
)

td = qc.build_phase_transducer(sol)
print("objective:", sol.objective_value)
print("rank dims:", sol.witness_dims)
print("private dim:", td.private_dim)
```

## Analytic threshold block example

```python
import query_complexity as qc

blocks = qc.threshold_eta_blocks(4, 2)
for r, data in blocks.items():
    print("r =", r)
    print("labels =", data["labels"])
    print("eta =", data["eta"])
    print("B rank =", 1 if data["eta"].any() else 0)
```
