"""Include the research workspace as installed, exportable package resources."""
import hashlib
import json
from pathlib import Path
import shutil
from setuptools.command.build_py import build_py

DIRECTORIES = ('src', 'docs', 'examples', 'experiments', 'plots', 'results', 'julia', 'notebooks', 'tests')
FILES = ('README.md', 'LICENSE', 'CHANGELOG.md', 'CONTRIBUTING.md', 'pyproject.toml', 'MANIFEST.in', 'src/build_support.py', 'majority_rank_one_closed_X.pdf')


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parents[1]
        target = Path(self.build_lib) / 'query_complexity_resources' / 'assets'
        if target.exists():
            shutil.rmtree(target)
        paths = [root / name for name in FILES]
        for name in DIRECTORIES:
            paths.extend(p for p in (root / name).rglob('*') if p.is_file())
        records = {}
        for source in sorted(paths):
            relative = source.relative_to(root)
            if any(part.startswith('.') or part == '__pycache__' or part.endswith('.egg-info') for part in relative.parts):
                continue
            if source.suffix in ('.pyc', '.pyo'):
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            records[relative.as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
        (target.parent / 'manifest.json').write_text(json.dumps(records, sort_keys=True, indent=2)+'\n')
