# Complete installed research workspace

Version 0.2.1 includes the research assets in the wheel as well as the source
archive. No Git checkout is needed to read the documentation, run examples,
inspect measurements, or obtain the Julia project.

## Install and export

```bash
python -m pip install --upgrade 'query-complexity[experiments,symbolic]'
python -m query_complexity_resources list
python -m query_complexity_resources export query-complexity-workspace
cd query-complexity-workspace
```

The export directory must be new or empty. The exporter verifies each file
against its bundled SHA256 digest and refuses to overwrite an existing
workspace. Export to another directory to preserve previous experiment
outputs. Python library sources are installed as ordinary import packages;
the export includes their source under `src/` as well, producing a complete
buildable workspace. The PyPI source archive provides the same source
layout; use GitHub for version history.

## What is included

| Path in exported workspace | Contents |
| --- | --- |
| `src/` | Python numerical packages, resource exporter and build helper |
| `docs/` | Installation, API, tutorials, mathematical guides and proof sources |
| `examples/` | Six short executable examples |
| `experiments/` | Longer parameter sweeps and reproducible comparisons |
| `plots/` | Plot generation scripts |
| `results/` | Recorded data, figures, solver statuses and interpretation |
| `notebooks/` | Interactive research notebook |
| `tests/` | Python numerical regression tests |
| `julia/` | Project and manifest, triangle code, results, vendored SDPLRPlus |
| Root files | README, changelog, contribution/release metadata, license and majority proof PDF |

The generated bundle excludes caches, bytecode, credentials and Git history.
Runtime imports remain `query_complexity`, `junta_learning`, and
`sdp_hierarchy`. Resource access is through `query_complexity_resources`.
The resource manifest is generated during wheel construction, so editable
source installs should use the checkout's research directories directly.

## Walkthrough 1: query transducer and witness size

```bash
python examples/example_or.py
python examples/general_function.py
python examples/example_threshold_blocks.py
```

The first example constructs an analytic OR witness and transducer. The
second solves a nonsymmetric function with Clarabel. The third prints
analytic threshold factors. The Boolean API reports the phase-normalized
objective `Adv/2`; inspect mapping and unitarity errors when extracting a
transducer from numerical SDP output.

For a larger query/space sweep, read
[the query-complexity guide](guides/query-complexity.md). Fixed-rank search
failure does not certify infeasibility.

## Walkthrough 2: majority junta learning

```bash
python examples/majority_junta.py
python -m junta_learning.rank_one --ks 3 7 15 31 --exact
python experiments/experiment_majority_junta.py --ks 3 7 15
```

The explicit period-four rank-one witness has proved logarithmic cost.
The convex count SDP can improve the constant because it allows larger
rank. The observed `4/pi` asymptotic is numerical. The experiment writes
`results/junta/majority_comparison.csv` and can overwrite that archived file.
See [the learning guide](guides/junta-learning.md) for all normalizations,
multi-bias optimization, and bounded-size learning.

## Walkthrough 3: irrep hierarchy

```bash
python examples/symmetric_hierarchy.py
python experiments/experiment_irrep_hierarchy.py
```

For three-bit majority, the trivial query-stabilizer sector costs 3, while
the next level costs 2 and matches a dense SDP. Level r retains r+1 present
irrep types; multiplicities are retained in full. The first sector is
intersection-free with independent query-orbit channels. The earlier
common-bias learning SDP is a further restriction. See
[the hierarchy guide](guides/irrep-hierarchy.md) for definitions and proofs
of nestedness, normalization and the full-level endpoint.

## Walkthrough 4: triangle finding and limits

```bash
python examples/triangle_hierarchy.py --vertices 4
python experiments/experiment_triangle_level2_scaling.py
```

The second script checks the full-domain resource requirements through ten
vertices but only solves feasible small instances. It records larger sizes
as unrun. Five vertices already require at least 62 GB for one lift in this
reference implementation. Restricted-domain adversary measurements are
lower bounds; fitted growth curves are not measured upper bounds. Read
[the scaling report](../results/hierarchy/TRIANGLE_LEVEL2_SCALING.md).

For the separate Julia implementation:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/test/runtests.jl
```

Pip supplies the Julia files but does not install Julia, download Julia
packages or configure a MOSEK license. Clarabel supports the small examples
in [the Julia guide](../julia/triangle/README.md).

## Tests and solver diagnostics

```bash
python -m pip install 'query-complexity[dev]'
python -m pytest tests -q
```

To test the installed wheel rather than the exported source tree, pass
`-o pythonpath= --import-mode=importlib` to pytest. To develop from the
exported sources, run `python -m pip install -e .` first.
Check `status`, equality residuals, minimum eigenvalues, and load/support
checks. `optimal_inaccurate` and the hierarchy's `numerically_feasible` flag
are numerical results, not exact certificates.
