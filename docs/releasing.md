# Releasing to PyPI

The distribution name is `query-complexity`. It installs `query_complexity`,
`junta_learning` and `sdp_hierarchy`. Version 0.2.0 restored the earlier
package's `query_complexity.core` layout and expands the distribution.

1. Update `project.version` in `pyproject.toml` and `__version__` in
   `src/query_complexity/__init__.py`; update `CHANGELOG.md`.
2. Run the tests and examples against a wheel installed outside the checkout.
3. Build fresh artifacts into a new output directory:

   ```bash
   python -m build --outdir /tmp/query-complexity-release
   python -m twine check /tmp/query-complexity-release/*
   ```

4. Inspect the wheel and source archive. Runtime packages and the generated research bundle belong in the wheel;
   the buildable sources also belong in the source archive.
5. Upload the exact validated files:

   ```bash
   python -m twine upload --non-interactive /tmp/query-complexity-release/*
   ```

Use PyPI credentials from the publishing environment, a credential store or
an external `.pypirc`. Never commit or print tokens. Verify the published
metadata and file checksums after upload. PyPI versions are immutable; a
changed artifact needs a new release version.

The Julia project is not installed or instantiated by pip.
