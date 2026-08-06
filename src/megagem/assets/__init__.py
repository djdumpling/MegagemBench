"""Packaged model artifacts (selector price laws, value head, certified sweep).

See this directory's README.md for provenance and regeneration instructions.
"""

from importlib.resources import as_file, files
from pathlib import Path


def asset_path(name: str) -> Path:
    """Resolve a packaged asset to a filesystem path.

    The package installs unzipped (wheel/editable), so the path stays valid
    after the context manager exits.
    """
    with as_file(files("megagem.assets").joinpath(name)) as p:
        return Path(p)
