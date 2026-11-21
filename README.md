# Query complexity

Research code for quantum query complexity: adversary-bound SDPs,
space-efficient phase-oracle transducers, symmetric-function query/space
tradeoffs, and SymbolicWedderburn symmetry reduction for triangle finding
and graph collision.

## Layout

| Path | Purpose |
| --- | --- |
| `sdp_hierarchy/` | General permutation-action irrep-prefix hierarchy and symmetric-function examples |
| `junta_learning/` | Intersection-free junta learning SDPs and explicit logarithmic-cost majority witnesses |
| `docs/junta/` | Optimization derivations and majority proof notes |
| `query_complexity.py` | Python library and existing public import |
| `examples/` | Small examples and Python API guide |
| `experiments/` | Convex and fixed-rank query/space searches |
| `plots/` | Plotting scripts for those searches |
| `tests/` | Numerical regression checks |
| `notebooks/` | Interactive visual examples |
| `results/query_space/` | Existing CSV measurements and figures |
| `julia/Project.toml`, `julia/Manifest.toml` | Reproducible triangle and benchmark environment |
| `julia/triangle/` | SymbolicWedderburn solvers, runners, and detailed research notes |
| `julia/triangle/results/` | Orbit stores, measurements, and figures |
| `julia/benchmarks/` | SDPLRPlus threshold benchmark |
| `julia/vendor/` | Local SDPLRPlus source and upstream license |

## Python setup and examples

From the repository root:

```bash
python -m pip install -e '.[experiments,dev]'
python examples/example_or.py
python examples/example_threshold_blocks.py
python -m pytest -q
```

Python 3.10 or later is required. NumPy, SciPy and CVXPY provide the core
numerics; Matplotlib is used by experiment and plotting scripts. MOSEK and
Qiskit are optional extras (`.[mosek]`, `.[qiskit]`). MOSEK requires a license.
See [the Python API guide](examples/README.md) for mathematical conventions
and transducer examples. The phase-normalized objective is `Adv±(f)/2`.

Experiments and plots interpret relative input/output paths from the current
working directory. To reproduce the existing figures and keep new output
beside the archived measurements, run them from `results/query_space` after
the editable installation:

```bash
cd results/query_space
python ../../experiments/experiment_threshold_tradeoff.py --n 4 --ks 1,2 --points 3
python ../../plots/plot_alpha_min_rank_curves.py
```

Fixed-rank feasibility searches are nonconvex: a verified candidate certifies
feasibility, while failure to find one does not certify infeasibility.

## Junta learning and the logarithmic majority witness

The [junta-learning guide](junta_learning/README.md) contains the
intersection-free count SDP, equivalent nuclear-norm optimization, rank-one
half-set search, explicit period-four `O(log k)` majority witness, mixed-size
SDP, and general symmetric multi-bias optimizer.

```bash
python -m junta_learning.majority_sdp --ks 1 3 5 7
python -m junta_learning.rank_one --ks 3 7 15 31 --exact
python -m junta_learning.bounded_size --ks 1 2 3 --details
```

The explicit witness has a proved logarithmic cost; the observed `4/pi`
leading coefficient is numerical. The guide explains the discovery sequence,
normalization, and how to recover the full rank-one matrices.

## General irrep-prefix SDP hierarchy

The [hierarchy guide](sdp_hierarchy/README.md) defines a finite-action pipeline
with intersection-free level 0, whole-irrep prefix restrictions, multiplicity
PSD blocks, independent reference formulations, and numerical validation.
It includes symmetric Boolean evaluation and symmetric-junta identification.

```bash
python experiments/experiment_irrep_hierarchy.py
```

For three-bit majority, level 0 costs 3 and the next level costs 2, matching
the full SDP. The enumerative backend targets small instances; the guide
specifies the stabilizer ordering, normalization, and relationship to the
earlier common-bias junta SDP.

## Julia: triangle finding with SymbolicWedderburn

The triangle code forms invariant SDPs under vertex relabelings and uses
SymbolicWedderburn to decompose PSD constraints into smaller blocks. It
includes the restricted adversary dual, primal/orbit searches, completely
bounded degree SDP, and the related graph-collision experiments.

From the repository root:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
julia --project=julia julia/test/runtests.jl
julia --project=julia julia/triangle/run_triangle_symmetric.jl 3 1
julia --project=julia julia/triangle/triangle_cb_degree_sdp.jl 3 2 0.3333333333333333 clarabel
julia --project=julia julia/triangle/graph_collision_adversary_dual.jl star 4 clarabel
julia --project=julia julia/benchmarks/benchmark_sdplrplus.jl
```

The manifest was generated with Julia 1.12.5; use that version to reproduce
it. SymbolicWedderburn 0.4.4 is pinned as a registered dependency. The local
SDPLRPlus version is included because the original environment depended on
a modified checkout via an absolute path. The environment now references
`vendor/SDPLRPlus` relative to `julia/`. MOSEK is the default for many runners;
use Clarabel for the small license-free examples above where supported.

See [triangle commands and research notes](julia/triangle/README.md) for
orbit selection, pruning, greedy search, and stored results. Most Julia
runners locate default stores and outputs in `julia/triangle/results`
independently of the working directory. For triangle Python plots, run from
that results directory using `python ../plot_NAME.py`; scripts that resolve
their own directory also target `results/`.

## Provenance

This organization preserves the existing local Python implementation,
experiments, notebook, and results. Triangle sources and results were copied
from the sibling `triangle_adversary_sdp` project; that checkout remains
untouched. See [Julia provenance](julia/PROVENANCE.md) for dependency details.
Generated caches, build output, and package metadata are ignored.
