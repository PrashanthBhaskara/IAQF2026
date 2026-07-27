from __future__ import annotations

import hashlib
import importlib
import re
import shutil
from pathlib import Path

import pytest
from PIL import Image

from iaqf.config import EXPECTED_FIGURES, EXPECTED_TABLES

ROOT = Path(__file__).resolve().parents[1]

PAPER_SHA256 = {
    "IAQF_Inefficient_Markets_2026.tex": "8f645f8d7d97e0ef0042f4a1d10e206fb9c76be967224a9194e340b778973335",
    "IAQF_Inefficient_Markets_2026.pdf": "b583b7051b00604af3da42559c287865335ea51b7afb2b590d60ff37f454025b",
}


def validation_module():
    return importlib.import_module("iaqf.validation")


def repo_paths(root: Path):
    return importlib.import_module("iaqf.config").RepoPaths.from_root(root)


def copy_artifact_surface(destination: Path) -> None:
    shutil.copytree(ROOT / "figures_col", destination / "figures_col")
    shutil.copytree(ROOT / "tables", destination / "tables")
    for paper_name in PAPER_SHA256:
        shutil.copy2(ROOT / paper_name, destination / paper_name)


def test_frozen_tex_and_pdf_sha256_are_immutable() -> None:
    observed = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in PAPER_SHA256
    }

    assert observed == PAPER_SHA256


def test_final_tex_references_exactly_the_fourteen_owned_pngs() -> None:
    tex = (ROOT / "IAQF_Inefficient_Markets_2026.tex").read_text(encoding="utf-8")
    referenced = {
        Path(token).name
        for token in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    }
    owned = {path.name for path in (ROOT / "figures_col").iterdir() if path.is_file()}

    assert referenced == EXPECTED_FIGURES
    assert owned == EXPECTED_FIGURES


def test_owned_table_artifact_set_is_exact() -> None:
    owned = {path.name for path in (ROOT / "tables").iterdir() if path.is_file()}

    assert owned == EXPECTED_TABLES


def test_validators_accept_the_committed_frozen_artifacts() -> None:
    paths = repo_paths(ROOT)

    assert validation_module().validate_frozen_paper(paths) is None
    assert validation_module().validate_artifacts(paths) is None
    assert validation_module().validate_repository(paths) is None


@pytest.mark.parametrize("mutation", ["missing", "stale"])
def test_artifact_validation_rejects_an_invalid_owned_set(
    tmp_path: Path, mutation: str
) -> None:
    copy_artifact_surface(tmp_path)
    if mutation == "missing":
        (tmp_path / "figures_col" / "fig_var_irf.png").unlink()
    else:
        (tmp_path / "figures_col" / "stale_preview.png").write_bytes(b"stale")

    with pytest.raises(ValueError, match=r"(?i)(artifact|figure|missing|unexpected)"):
        validation_module().validate_artifacts(repo_paths(tmp_path))


def test_frozen_paper_validation_rejects_tampering(tmp_path: Path) -> None:
    copy_artifact_surface(tmp_path)
    with (tmp_path / "IAQF_Inefficient_Markets_2026.tex").open("ab") as paper:
        paper.write(b"\n% tampered\n")

    with pytest.raises(ValueError, match=r"(?i)(sha|hash|frozen|paper)"):
        validation_module().validate_frozen_paper(repo_paths(tmp_path))


def test_artifact_validation_rejects_wrong_figure_dimensions(tmp_path: Path) -> None:
    copy_artifact_surface(tmp_path)
    Image.new("RGB", (1, 1)).save(
        tmp_path / "figures_col" / "fig_var_irf.png",
        format="PNG",
    )

    with pytest.raises(ValueError, match=r"(?i)(dimension|size|figure)"):
        validation_module().validate_artifacts(repo_paths(tmp_path))


def test_artifact_validation_rejects_same_size_figure_tampering(tmp_path: Path) -> None:
    copy_artifact_surface(tmp_path)
    path = tmp_path / "figures_col" / "fig_var_irf.png"
    with Image.open(path) as source:
        changed = source.convert("RGB")
    changed.putpixel((0, 0), (0, 0, 0))
    changed.save(path)

    with pytest.raises(ValueError, match=r"(?i)(sha|hash|figure)"):
        validation_module().validate_artifacts(repo_paths(tmp_path))


def test_legacy_bootstrap_claim_records_recovered_numbers_and_method_mismatch() -> None:
    assert validation_module().LEGACY_BOOTSTRAP_CLAIM == {
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
