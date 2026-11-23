# Documentation

Start with [installation](installation.md), then [runnable examples](examples.md).

## Choose an API

| Task | Import | Guide |
| --- | --- | --- |
| Boolean adversary SDPs, trace/rank tradeoffs, transducers | `import query_complexity as qc` | [Query-complexity guide](guides/query-complexity.md) |
| Hidden-support learning and logarithmic-cost majority witnesses | `from junta_learning import majority_sdp, rank_one` | [Junta guide](guides/junta-learning.md) |
| General finite permutation-action SDP hierarchies | `from sdp_hierarchy import Hierarchy` | [Hierarchy guide](guides/irrep-hierarchy.md) |
| Large selected triangle graph families using Julia | Julia environment | [Triangle guide](../julia/triangle/README.md) |

The [API reference](api.md) lists the supported entry points. The guides
state mathematical conventions, assumptions, and numerical limitations.

## Repository organization

| Directory | Contents |
| --- | --- |
| `src/query_complexity/` | Boolean-query numerical implementation and stable public exports |
| `src/junta_learning/` | Count SDPs, rank-one formulas, mixed-size and multi-bias learning |
| `src/sdp_hierarchy/` | Group decomposition, generic SDP pipeline, example model builders |
| `examples/` | Small runnable introductions, using public APIs |
| `tests/` | Numerical and structural regression tests |
| `experiments/` | Longer research sweeps and artifact generation |
| `plots/` | Rendering of query/space measurements |
| `results/` | Archived measurements, statuses, provenance and scaling reports |
| `docs/guides/` | User and mathematical guides |
| `docs/junta/` | Majority proof sources and derivations |
| `julia/` | Separate pinned Julia project, triangle tools, vendored solver |
| `notebooks/` | Interactive research examples |

Python runtime packages contain no archived plots or solver transcripts.
The wheel installs Python runtime packages plus the exportable research bundle. The source archive also
contains guides, examples, tests, experiments, recorded data and Julia
sources. Use a Git checkout for development; source archives have no Git
history. The Julia environment requires Julia and its own dependencies.

## Reproduce results

After an editable installation, from the repository root:

```bash
python experiments/experiment_majority_junta.py
python experiments/experiment_irrep_hierarchy.py
python experiments/experiment_triangle_hierarchy.py
python experiments/experiment_triangle_level2_scaling.py
```

Experiments can overwrite their documented output files. The level-2
triangle scaling script records explicit resource-limit skips rather than
allocating infeasible graph domains. For plotting older query/space runs,
run from `results/query_space` as described in the query-complexity guide.

See [contributing](../CONTRIBUTING.md), [release procedure](releasing.md),
and [changes in 0.2.0](../CHANGELOG.md).

## Export all documentation and research assets

```bash
python -m query_complexity_resources export query-complexity-workspace
```

The installed wheel includes the documentation, examples, tests, experiments,
results, notebooks and Julia sources. See [the complete workspace guide](research-workspace.md) for workflows, solver setup and limitations.
