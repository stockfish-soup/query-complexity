# Python API guide

The Python library implements the space-efficient **phase-oracle one-vector
transducer** path and query/space tradeoff searches.

## What is kept

- `solve_adversary_sdp(...)`: one-vector phase SDP for arbitrary Boolean domains.
- `solve_symmetric_orbit_sdp(...)`: Terwilliger/orbit-reduced phase SDP for symmetric functions.
- `solve_symmetric_adversary_sdp(...)`: convenience wrapper that solves the orbit SDP and expands to the Boolean input basis.
- `build_adversary_transducer(...)`: extracts the input-independent transducer unitary `U`.
- `run_adversary_algorithm(...)`: simulates the repeated transducer algorithm.
- `threshold_eta_blocks(...)`: analytic rank-one Schrijver blocks for `THRESHOLD^k_n`.
- Optional `qiskit_adversary_circuit(...)` for dense Qiskit simulation.

## Quick OR example

```python
import query_complexity as qc

sol = qc.analytic_or_solution(3)
td = qc.build_adversary_transducer(sol)

print("objective:", sol.objective_value)
print("private dim:", td.private_dim)
print("map error:", td.map_error)
qc.show_catalysts(sol)
```

## Symmetric SDP example

```python
import query_complexity as qc

n, k = 4, 2
sol = qc.solve_symmetric_adversary_sdp(
    n,
    qc.THRESHOLD_layers(n, k),
    solver="MOSEK",          # or omit to use an installed default
    epsilon=1e-5,
    psd_tol=1e-5,
)

td = qc.build_adversary_transducer(sol)
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

## Trace/query-space tradeoff

`solve_trace_minimized_dual(...)` minimizes
`1/2 * sum_i trace(X_i)` subject to the adversary constraints and

```text
max_x 1/2 * sum_i X_i[x,x] <= alpha * adversary_value + epsilon.
```

The module stores the phase-normalized adversary value
`adversary_value = Adv±(f) / 2`, so this is equivalently
`OBJ <= alpha * Adv±(f)` in the usual normalization.

For symmetric functions, use `solve_symmetric_trace_minimized_dual(...)`.
The helper `symmetric_witness_rank(...)` recovers the full-basis rank of each
`X_i` directly from the compact Terwilliger blocks.

After an editable installation, run from `results/query_space`:

```bash
python ../../experiments/experiment_threshold_tradeoff.py --n 8 --ks 1,2,4 --points 9
```

It scans `alpha` from `1` to `n / Adv±(THRESHOLD(n,k))` and writes
`threshold_tradeoff.png` plus the underlying `threshold_tradeoff.csv`.

For a nonsymmetric fixed-rank feasibility search, use
`solve_fixed_rank_dual_feasibility(...)`. It factors each witness matrix as
`X_i = Z_i Z_i.T`, fixes the number of columns of `Z_i`, and solves the
resulting nonconvex least-squares problem with multiple starts. A returned
feasible candidate can be checked directly; a failed search is not a proof of
infeasibility.

The symmetric/full-convex/fixed-rank comparison is reproducible with:

```bash
python ../../experiments/experiment_fixed_rank_tradeoff.py --n 4 --ks 1,2
```
