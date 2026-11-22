# Example walkthroughs

Install the package first. The snippets in the main README work from any
working directory. The following scripts are in the repository and source
archive; execute them from the repository root.

| Script | Use case | Expected result |
| --- | --- | --- |
| `python examples/example_or.py` | Analytic three-bit OR transducer | Objective sqrt(3)/2, private dimension 3, tiny mapping error |
| `python examples/example_threshold_blocks.py` | Analytic threshold factors | Schrijver block labels, factors and matrices |
| `python examples/general_function.py` | Nonsymmetric Boolean SDP | Finite phase objective and a transducer mapping residual |
| `python examples/majority_junta.py` | Explicit rank-one majority witness versus convex optimum | Costs approximately 5.155556 and 5.069052 |
| `python examples/symmetric_hierarchy.py` | Irrep prefixes on three-bit majority | Level 0 cost 3, next and dense-reference costs 2 |
| `python examples/triangle_hierarchy.py --vertices 4` | Triangle-finding level 2 | Approximately 3; status/residual validation reported |

All use analytic formulas or Clarabel/SCS; no MOSEK license is needed.
Roundoff and solver versions can slightly change numbers and statuses.

## Select the right normalization

`query_complexity` reports phase-normalized `Adv/2`. The learning modules
and generic hierarchy report the unhalved filtered-adversary objective.
For example, analytic OR on three bits reports sqrt(3)/2 through the former
API, while evaluating OR through the generic hierarchy gives sqrt(3).

## Export a majority witness

```python
import numpy as np
from junta_learning.rank_one import count_witness

z = count_witness(7)
np.save("majority_count_factor.npy", z)
```

This compact count-space factor defines rank-one PSD matrices for all
subset queries without enumerating the ambient universe. The
[junta guide](guides/junta-learning.md) gives the normalization and lift.
The theorem proves O(log k); the observed 4/pi asymptotic coefficient
remains numerical.

## Define a different finite oracle problem

```python
from sdp_hierarchy import OracleProblem, Hierarchy

problem = OracleProblem(
    answers=[[0, 1], [1, 0]],
    generators=[([1, 0], [1, 0])],
    targets=[0, 1],
)
result = Hierarchy(problem).solve(0)
assert result["numerically_feasible"]
```

Generators specify zero-based concept and query permutations. The pipeline
validates oracle covariance and target equivalence. Level 0 is the trivial
query-stabilizer sector; level r keeps r+1 present types. See the
[hierarchy guide](guides/irrep-hierarchy.md) for ordering and scale limits.

## Larger experiments

The `experiments/` scripts generate measurements; the short `examples/`
scripts introduce APIs. Experiments document output locations and may
replace archived files. The earlier unrestricted OR value experiment
`experiment_or_symmetric_value.py` uses n=60 and a solver, so it is kept
with experiments rather than introductory examples.
