# Contributing

Install from the checkout using `python -m pip install -e '.[dev,experiments,symbolic]'`.
Run `python -m pytest -q`. Python code belongs under `src/`; public examples
belong under `examples/`; expensive sweeps belong under `experiments/`.
Keep archived measurements separate from runtime code.

Preserve the existing three import namespaces and the documented objective
normalizations. Numerical changes should include checks against independent
formulations, known small values, or reconstructed witness constraints.
Record solver status and residuals along with objective values. Do not
replace missing large-instance solves with extrapolated measurements.

Build with `python -m build` and check metadata with `python -m twine check dist/*`.
Verify both wheel and source archive in a temporary environment before a
release. See [release instructions](docs/releasing.md).
