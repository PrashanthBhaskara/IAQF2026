from __future__ import annotations

import hashlib
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from iaqf.config import EXPECTED_FIGURES, EXPECTED_TABLES
from iaqf.data import PROCESSED_FILES

ROOT = Path(__file__).resolve().parents[1]
PAPER_SHA256 = {
    "IAQF_Inefficient_Markets_2026.tex": (
        "8f645f8d7d97e0ef0042f4a1d10e206fb9c76be967224a9194e340b778973335"
    ),
    "IAQF_Inefficient_Markets_2026.pdf": (
        "b583b7051b00604af3da42559c287865335ea51b7afb2b590d60ff37f454025b"
    ),
}


def _record_calls(monkeypatch: pytest.MonkeyPatch, runner):
    calls: list[str] = []
    raw = object()
    analysis = object()

    def replace(name: str, result=None) -> None:
        def fake(*_args, **_kwargs):
            calls.append(name)
            return result

        monkeypatch.setattr(runner, name, fake)

    replace("validate_frozen_paper")
    replace("refresh_raw")
    replace("validate_raw")
    replace("load_raw", raw)
    replace("build_master", analysis)
    replace("write_processed")
    replace("write_tables")
    replace("write_figures")
    replace("validate_repository")
    return calls


def test_main_is_offline_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = importlib.import_module("run_all")
    calls = _record_calls(monkeypatch, runner)

    assert runner.main([]) == 0
    assert calls == [
        "validate_frozen_paper",
        "validate_raw",
        "load_raw",
        "build_master",
        "write_processed",
        "write_tables",
        "write_figures",
        "validate_repository",
    ]


def test_refresh_flag_is_the_only_path_to_network_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = importlib.import_module("run_all")
    calls = _record_calls(monkeypatch, runner)

    assert runner.main(["--refresh-data"]) == 0
    assert calls == [
        "validate_frozen_paper",
        "refresh_raw",
        "validate_raw",
        "load_raw",
        "build_master",
        "write_processed",
        "write_tables",
        "write_figures",
        "validate_repository",
    ]


def _copy_reproduction_checkout(destination: Path) -> None:
    shutil.copytree(ROOT / "iaqf", destination / "iaqf")
    shutil.copytree(ROOT / "data_raw", destination / "data_raw")
    shutil.copytree(ROOT / "figures_col", destination / "figures_col")
    shutil.copy2(ROOT / "run_all.py", destination / "run_all.py")
    for paper_name in PAPER_SHA256:
        shutil.copy2(ROOT / paper_name, destination / paper_name)


def _hash_outputs(checkout: Path) -> dict[str, str]:
    directories = ("data_processed", "tables", "figures_col")
    return {
        str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in directories
        for path in sorted((checkout / directory).iterdir())
        if path.is_file()
    }


@pytest.mark.integration
def test_pipeline_is_idempotent_from_a_non_root_working_directory(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    unrelated = tmp_path / "unrelated"
    checkout.mkdir()
    unrelated.mkdir()
    _copy_reproduction_checkout(checkout)

    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    command = [sys.executable, str(checkout / "run_all.py")]

    first = subprocess.run(
        command,
        cwd=unrelated,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    first_hashes = _hash_outputs(checkout)

    second = subprocess.run(
        command,
        cwd=unrelated,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert _hash_outputs(checkout) == first_hashes

    assert set(path.name for path in (checkout / "data_processed").iterdir()) == set(
        PROCESSED_FILES.values()
    )
    assert set(path.name for path in (checkout / "tables").iterdir()) == set(
        EXPECTED_TABLES
    )
    assert set(path.name for path in (checkout / "figures_col").iterdir()) == set(
        EXPECTED_FIGURES
    )
    assert {
        name: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
        for name in PAPER_SHA256
    } == PAPER_SHA256
    assert not any(unrelated.iterdir())
