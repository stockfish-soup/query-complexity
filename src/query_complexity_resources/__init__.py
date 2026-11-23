"""Installed documentation, examples, research data and Julia project.

Use ``python -m query_complexity_resources export DIRECTORY`` to obtain a
standalone research workspace without cloning the repository.
"""
import hashlib
from importlib.resources import files
import json
from pathlib import Path


def manifest():
    """Return resource paths and SHA256 digests from the built distribution."""
    resource = files(__package__).joinpath('manifest.json')
    if not resource.is_file():
        raise RuntimeError('Resource bundle is generated when building a wheel. Install the PyPI package or a locally built wheel.')
    return json.loads(resource.read_text(encoding='utf-8'))


def export(destination):
    """Export and checksum-verify all assets to a new or empty directory.

    Existing nonempty directories are rejected; files are never overwritten.
    Return the resolved output Path.
    """
    destination = Path(destination).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f'Destination must be a new or empty directory: {destination}')
    records = manifest()
    destination.mkdir(parents=True, exist_ok=True)
    assets = files(__package__).joinpath('assets')
    for name, digest in records.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid resource path')
        data = assets.joinpath(*relative.parts).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f'Resource checksum mismatch: {name}')
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as handle:
            handle.write(data)
    return destination
