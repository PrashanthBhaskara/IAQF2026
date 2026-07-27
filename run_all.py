"""Single entry point for reproducing the IAQF analysis."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from iaqf.config import RepoPaths
from iaqf.data import (
    build_master,
    load_raw,
    refresh_raw,
    validate_raw,
    write_processed,
)
from iaqf.figures import write_figures
from iaqf.tables import write_tables
from iaqf.validation import validate_frozen_paper, validate_repository

ROOT = Path(__file__).resolve().parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the IAQF paper artifacts from committed raw data."
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="retrieve raw exchange data before rebuilding (requires network access)",
    )
    arguments = parser.parse_args(argv)
    paths = RepoPaths.from_root(ROOT)

    validate_frozen_paper(paths)
    if arguments.refresh_data:
        print("Refreshing raw exchange data...")
        refresh_raw(paths)

    print("Building processed data...")
    validate_raw(paths)
    data = build_master(load_raw(paths))
    write_processed(data, paths)

    print("Writing tables and figures...")
    write_tables(data, paths)
    write_figures(data, paths)
    validate_repository(paths)
    print("Reproduction complete; the frozen paper was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
