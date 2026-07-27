from __future__ import annotations

import contextlib
import hashlib
from importlib import import_module
from pathlib import Path

import pytest
from PIL import Image

from iaqf.config import EXPECTED_FIGURE_DIMENSIONS, RepoPaths
from iaqf.data import load_processed

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_PNGS = {
    "fig_arbitrage_after_fees.png": "4cde656cf3000c0466b0b6ef33e5ef1ba58db5d46b703ee674e515db3aaf81ee",
    "fig_correlation_regime_heatmap.png": "a9e8cfa57513146683e284e5bc8e43c26527261baa18044a32a03c9efdef5208",
    "fig_cross_exchange_basis.png": "76584526df591d78d9477f7903de97fcb16e6d8c88059e475e26dc7944f333d6",
    "fig_dispersion_vs_adjusted_kraken.png": "52f744b5734d34bf4f64af65e091a3377437263d4b3e66ff070dc7e45da7444a",
    "fig_half_life_robustness.png": "f4d28ff8093e25623832124b26d207d9762dbf9fc7109ac84a3bb4332b9d7b2d",
    "fig_liquidity_roll_amihud.png": "4f560c15a56842decfcc40726fc621b143eebd33b882f2d408e8a6f5889c01f6",
    "fig_realized_volatility.png": "7704507007cd6fbd6df5be745dc15d14a0d50e54a5fd96f598164170dcbeacd0",
    "fig_stablecoin_peg.png": "435d21492693e80461cbb3669b39aca96c106f605e0e1cd96bd418beb1ad1351",
    "fig_stablecoin_substitution_scatter.png": "1068d149717446249c92f7559fa8679d375f40e45d4ed60c2873e400086b7311",
    "fig_svb_crisis_zoom.png": "b06912d322d4edd0558757e11d8af676ea88fcd16d0f9dbd3abe479bb34b9562",
    "fig_tail_blowout_kde.png": "806ce8b59497d8f1348b89d05971c5c9b4bfb81715e941e27bc81a608773a5c1",
    "fig_two_layer_persistence.png": "2d40e5ecb1673103ab860f4de5d9874459cb137f89f13555dbefe44419c5bec8",
    "fig_var_irf.png": "0f5b9e75b5627ee8a6632dfb214d6abbebef5110a74f38c9bfeef306ab9fadd7",
    "fig_volume_share.png": "80da9f714a5719bee4b606acf805526a377b55ba4f64edd4192f3a46a4df6075",
}


@pytest.mark.golden
def test_write_figures_reproduces_exact_pngs_only_under_explicit_paths(
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
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in (ROOT / "figures_col").glob("*.png")
    }

    with contextlib.chdir(unrelated_cwd):
        result = figures.write_figures(data, paths)

    assert result is None
    observed = {path.name for path in paths.figures.iterdir() if path.is_file()}
    assert observed == set(EXPECTED_PNGS)
    for filename, expected_hash in EXPECTED_PNGS.items():
        path = paths.figures / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        with Image.open(path) as image:
            assert image.size == EXPECTED_FIGURE_DIMENSIONS[filename]
    assert not paths.tables.exists()
    assert not any(unrelated_cwd.iterdir())
    assert committed_state == {
        path.name: (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in (ROOT / "figures_col").glob("*.png")
    }
