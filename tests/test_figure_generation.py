from __future__ import annotations

import contextlib
import hashlib
from importlib import import_module
from pathlib import Path

import pytest
from PIL import Image

from iaqf.config import (
    EXPECTED_FIGURE_DIMENSIONS,
    EXPECTED_FIGURE_SHA256,
    EXPECTED_FIGURES,
    RepoPaths,
)
from iaqf.data import load_processed

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.golden
def test_write_figures_writes_valid_pngs_only_under_explicit_paths(
    tmp_path: Path,
) -> None:
    figures = import_module("iaqf.figures")
    if not hasattr(figures, "write_figures"):
        raise AssertionError("missing planned iaqf.figures.write_figures API")

    data = load_processed(RepoPaths.from_root(ROOT))
    paths = RepoPaths.from_root(tmp_path / "output")
    paths.figures.mkdir(parents=True)
    (paths.figures / "stale.png").write_bytes(b"stale")
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    committed_state = {
        path.name: (
            path.stat().st_mtime_ns,
            path.read_bytes(),
        )
        for path in (ROOT / "figures_col").glob("*.png")
    }

    with contextlib.chdir(unrelated_cwd):
        result = figures.write_figures(data, paths)

    assert result is None
    observed = {path.name for path in paths.figures.iterdir() if path.is_file()}
    assert observed == EXPECTED_FIGURES
    for filename in EXPECTED_FIGURES:
        path = paths.figures / filename
        with Image.open(path) as image:
            image.verify()
            assert image.size == EXPECTED_FIGURE_DIMENSIONS[filename]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == EXPECTED_FIGURE_SHA256[filename]
        )
    assert not paths.tables.exists()
    assert not any(unrelated_cwd.iterdir())
    assert committed_state == {
        path.name: (
            path.stat().st_mtime_ns,
            path.read_bytes(),
        )
        for path in (ROOT / "figures_col").glob("*.png")
    }
