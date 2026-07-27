"""Validation of the repository's frozen paper and artifact surface."""

from __future__ import annotations

import hashlib
from pathlib import Path

from iaqf.config import EXPECTED_FIGURES, EXPECTED_TABLES, PAPER_SHA256, RepoPaths

LEGACY_BOOTSTRAP_CLAIM = {
    "method": "moving-block bootstrap",
    "replications": 5_000,
    "block_length_minutes": 60,
    "median_ratio": 567,
    "ci_95_ratio": (216, 1_898),
    "p_value": "<1e-4",
    "provenance_exception": True,
    "computed": False,
}


def validate_frozen_paper(paths: RepoPaths) -> None:
    """Raise when the committed paper TeX or PDF differs from its frozen hash."""
    paper_paths = {
        "IAQF_column_Final.tex": paths.paper_tex,
        "IAQF_column_Final.pdf": paths.paper_pdf,
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
    """Require the exact owned sets of fourteen figures and thirty-six tables."""
    figures = _owned_files(paths.figures, "figure")
    tables = _owned_files(paths.tables, "table")
    errors = []
    if figures != EXPECTED_FIGURES:
        errors.append(_set_difference("figure", EXPECTED_FIGURES, figures))
    if tables != EXPECTED_TABLES:
        errors.append(_set_difference("table", EXPECTED_TABLES, tables))
    if errors:
        raise ValueError("artifact set mismatch: " + "; ".join(errors))


def _owned_files(directory: Path, kind: str) -> set[str]:
    try:
        return {path.name for path in directory.iterdir() if path.is_file()}
    except OSError as error:
        raise ValueError(
            f"{kind} artifact directory is missing: {directory}"
        ) from error


def _set_difference(kind: str, expected: set[str], observed: set[str]) -> str:
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    return f"{kind}s missing={missing}, unexpected={unexpected}"
