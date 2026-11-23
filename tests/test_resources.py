"""Source installs explain generation; built wheels export verified assets."""
from importlib.resources import files
import hashlib
import pytest
import query_complexity_resources as resources


def test_resource_bundle_and_safe_export(tmp_path):
    if not files('query_complexity_resources').joinpath('manifest.json').is_file():
        with pytest.raises(RuntimeError, match='building a wheel'):
            resources.manifest()
        return
    manifest = resources.manifest()
    for name in ['docs/research-workspace.md', 'examples/majority_junta.py',
                 'julia/Project.toml', 'julia/vendor/SDPLRPlus/LICENSE',
                 'results/hierarchy/triangle_finding.json', 'notebooks/query_complexity_visual_examples.ipynb']:
        assert name in manifest
    target = resources.export(tmp_path / 'workspace')
    for name, digest in manifest.items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        resources.export(target)
