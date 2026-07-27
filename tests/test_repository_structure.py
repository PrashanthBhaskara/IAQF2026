from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODULES = {
    "__init__.py",
    "config.py",
    "data.py",
    "figures.py",
    "metrics.py",
    "tables.py",
    "validation.py",
}


def test_repository_has_only_the_documented_pipeline_surface() -> None:
    assert {path.name for path in (ROOT / "iaqf").glob("*.py")} == PACKAGE_MODULES
    assert not any((ROOT / "src").glob("*.py"))
    assert not (ROOT / ".env.example").exists()
    assert not (ROOT / "requirements.txt").exists()
    assert (ROOT / "run_all.py").is_file()
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "uv.lock").is_file()
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()


def test_production_code_uses_only_the_two_approved_state_classes() -> None:
    production = sorted((ROOT / "iaqf").glob("*.py")) + [ROOT / "run_all.py"]
    classes = []
    for path in production:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes.extend(
            (str(path.relative_to(ROOT)), node.name)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        )

    assert sorted(classes) == [
        ("iaqf/config.py", "RepoPaths"),
        ("iaqf/data.py", "AnalysisData"),
    ]


def test_formatted_production_code_remains_compact_after_full_recovery() -> None:
    production = sorted((ROOT / "iaqf").glob("*.py")) + [ROOT / "run_all.py"]

    assert (
        sum(len(path.read_text(encoding="utf-8").splitlines()) for path in production)
        < 4_900
    )


def test_reference_environment_is_exactly_pinned() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["requires-python"] == "==3.12.4"
    assert project["tool"]["uv"]["package"] is False
    dependencies = [
        *project["project"]["dependencies"],
        *project["dependency-groups"]["dev"],
    ]
    assert dependencies
    assert all("==" in dependency for dependency in dependencies)


def test_readme_documents_the_single_entry_point_and_legacy_exception() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "python run_all.py" in readme
    assert "python run_all.py --refresh-data" in readme
    assert "567 [216, 1,898]" in readme
    assert "64 [46, 88]" in readme
    assert "parametric AR(1) residual sieve" in readme
    assert "method mismatch" in readme.lower()
    assert "horizon 3" in readme.lower()
    assert "src/01_fetch_data.py" not in readme
    assert "requirements.txt" not in readme


def test_ci_runs_pytest_as_a_module_for_the_non_package_project() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert 'uv run python -m pytest -q -m "not golden and not integration"' in workflow
    assert 'uv run python -m pytest -q -m "golden or integration"' in workflow
    assert "runs-on: macos-14" in workflow
    assert "uv run pytest" not in workflow
    assert "astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b" in workflow
    assert "astral-sh/setup-uv@v8" not in workflow
