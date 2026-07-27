"""Validation of the repository's frozen paper and artifact surface."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from iaqf.config import (
    EXPECTED_FIGURE_DIMENSIONS,
    EXPECTED_FIGURE_SHA256,
    EXPECTED_FIGURES,
    EXPECTED_TABLES,
    PAPER_SHA256,
    RepoPaths,
)
from iaqf.data import PROCESSED_FILES, RAW_MARKETS, load_processed, validate_raw

LEGACY_BOOTSTRAP_CLAIM = {
    "submitted_method": "moving-block bootstrap",
    "submitted_replications": 5_000,
    "recovered_method": "parametric AR(1) residual sieve bootstrap",
    "recovered_replications": 10_000,
    "seed": 42,
    "median_ratio": 567,
    "ci_95_ratio": (216, 1_898),
    "p_value": "<1e-4",
    "provenance_exception": True,
    "computed": True,
    "method_matches_submitted_text": False,
}


def validate_frozen_paper(paths: RepoPaths) -> None:
    """Raise when the committed paper TeX or PDF differs from its frozen hash."""
    paper_paths = {
        "IAQF_Inefficient_Markets_2026.tex": paths.paper_tex,
        "IAQF_Inefficient_Markets_2026.pdf": paths.paper_pdf,
    }
    observed = {}
    try:
        for name, path in paper_paths.items():
            observed[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"frozen paper is missing or unreadable: {error}") from error
    if observed != PAPER_SHA256:
        raise ValueError(
            f"frozen paper SHA-256 mismatch: expected {PAPER_SHA256}, observed {observed}"
        )


def validate_artifacts(paths: RepoPaths) -> None:
    """Require the exact owned sets of figures and tables."""
    figures = _owned_files(paths.figures, "figure")
    tables = _owned_files(paths.tables, "table")
    errors = []
    if figures != EXPECTED_FIGURES:
        errors.append(_set_difference("figure", EXPECTED_FIGURES, figures))
    if tables != EXPECTED_TABLES:
        errors.append(_set_difference("table", EXPECTED_TABLES, tables))
    if errors:
        raise ValueError("artifact set mismatch: " + "; ".join(errors))
    _validate_figure_dimensions(paths.figures)
    _validate_figure_hashes(paths.figures)


def validate_repository(paths: RepoPaths) -> None:
    """Validate the complete committed reproduction surface."""
    validate_frozen_paper(paths)
    _require_exact_files(
        paths.raw,
        {f"{market}.parquet" for market in RAW_MARKETS},
        "raw data",
    )
    validate_raw(paths)
    _require_exact_files(
        paths.processed,
        set(PROCESSED_FILES.values()),
        "processed data",
    )
    load_processed(paths)
    validate_artifacts(paths)


def _owned_files(directory: Path, kind: str) -> set[str]:
    try:
        return {path.name for path in directory.iterdir() if path.is_file()}
    except OSError as error:
        raise ValueError(
            f"{kind} artifact directory is missing: {directory}"
        ) from error


def _require_exact_files(directory: Path, expected: set[str], kind: str) -> None:
    observed = _owned_files(directory, kind)
    if observed != expected:
        raise ValueError(
            f"{kind} set mismatch: {_set_difference(kind, expected, observed)}"
        )


def _validate_figure_dimensions(directory: Path) -> None:
    errors = []
    for name, expected in EXPECTED_FIGURE_DIMENSIONS.items():
        path = directory / name
        try:
            with Image.open(path) as figure:
                observed = figure.size
                figure.verify()
        except OSError as error:
            raise ValueError(f"figure is corrupt or unreadable: {path}") from error
        if observed != expected:
            errors.append(f"{name}: expected {expected}, observed {observed}")
    if errors:
        raise ValueError("figure dimension mismatch: " + "; ".join(errors))


def _validate_figure_hashes(directory: Path) -> None:
    observed = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in EXPECTED_FIGURE_SHA256
    }
    if observed != EXPECTED_FIGURE_SHA256:
        raise ValueError("figure SHA-256 mismatch")


def _set_difference(kind: str, expected: set[str], observed: set[str]) -> str:
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return f"{kind}s missing={missing}, unexpected={unexpected}"
