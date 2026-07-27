import math
import tempfile
import unittest
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAW_MARKETS = [
    "binance_btcusdt",
    "binance_btcusdc",
    "binance_usdcusdt",
    "coinbase_btcusd",
    "coinbase_btcusdt",
    "coinbase_usdtusd",
    "kraken_btcusd",
    "kraken_btcusdt",
    "kraken_btcusdc",
    "kraken_usdcusd",
    "kraken_usdtusd",
]

ROOT = Path(__file__).resolve().parents[1]

PRICE_COLUMNS = RAW_MARKETS + [
    "implied_usdt_usd_coinbase",
    "implied_usdt_usd_kraken",
    "implied_usdc_usd_kraken",
]

BASIS_COLUMNS = [
    "dispersion_usdc_kraken",
    "dispersion_usdt_kraken",
    "basis_usdc_kraken",
    "basis_usdt_kraken",
    "dispersion_usdt_coinbase",
    "basis_usdt_coinbase",
    "basis_usdc_usdt_binance",
    "xbasis_btcusdt_binance_kraken",
    "xbasis_btcusdt_coinbase_kraken",
    "xbasis_btcusd_coinbase_kraken",
    "usdt_peg_dev_kraken",
    "usdt_peg_dev_coinbase",
    "usdc_peg_dev_kraken",
]


def data_api():
    try:
        return import_module("iaqf.config"), import_module("iaqf.data")
    except ModuleNotFoundError as error:
        raise AssertionError("missing planned iaqf data API") from error


def candle_frame(close_by_minute, volume_by_minute=None):
    index = pd.DatetimeIndex(
        [pd.Timestamp(minute, tz="UTC") for minute in close_by_minute],
        name="timestamp",
    )
    closes = list(close_by_minute.values())
    volumes = (
        list(volume_by_minute.values())
        if volume_by_minute is not None
        else [1.0] * len(index)
    )
    return pd.DataFrame(
        {
            "open": closes,
            "high": [value * 1.01 for value in closes],
            "low": [value * 0.99 for value in closes],
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )


def valid_raw(overrides=None):
    default = candle_frame(
        {"2023-03-01 00:00": 100.0, "2023-03-01 00:07": 107.0},
        {"2023-03-01 00:00": 12.0, "2023-03-01 00:07": 19.0},
    )
    raw = {market: default.copy() for market in RAW_MARKETS}
    raw.update(overrides or {})
    return raw


class DataContractTests(unittest.TestCase):
    @pytest.mark.golden
    def test_committed_raw_data_rebuilds_all_six_processed_frames_exactly(self):
        _, data = data_api()
        raw = {
            market: pd.read_parquet(ROOT / "data_raw" / f"{market}.parquet")
            for market in RAW_MARKETS
        }

        rebuilt = data.build_master(raw)
        expected_files = {
            "prices": "prices.parquet",
            "price_ffill_flags": "price_ffill_flags.parquet",
            "ranges": "intraminute_ranges.parquet",
            "volumes": "volumes.parquet",
            "basis": "basis.parquet",
            "basis_ffill_flags": "basis_ffill_flags.parquet",
        }
        for attribute, filename in expected_files.items():
            pd.testing.assert_frame_equal(
                getattr(rebuilt, attribute),
                pd.read_parquet(ROOT / "data_processed" / filename),
                check_exact=True,
            )

    def test_build_master_returns_fixed_utc_grid_and_ordered_schema(self):
        _, data = data_api()
        result = data.build_master(valid_raw())

        self.assertIsInstance(result, data.AnalysisData)
        expected_index = pd.date_range(
            "2023-03-01 00:00",
            "2023-03-21 23:59",
            freq="1min",
            tz="UTC",
        )
        self.assertEqual(len(expected_index), 30_240)
        for frame in (
            result.prices,
            result.price_ffill_flags,
            result.ranges,
            result.volumes,
            result.basis,
            result.basis_ffill_flags,
        ):
            pd.testing.assert_index_equal(frame.index, expected_index)
        self.assertEqual(list(result.prices.columns), PRICE_COLUMNS)
        self.assertEqual(list(result.price_ffill_flags.columns), PRICE_COLUMNS)
        self.assertEqual(list(result.ranges.columns), RAW_MARKETS)
        self.assertEqual(list(result.volumes.columns), RAW_MARKETS)
        self.assertEqual(list(result.basis.columns), BASIS_COLUMNS)
        self.assertEqual(list(result.basis_ffill_flags.columns), BASIS_COLUMNS)

    def test_prices_fill_exactly_five_minutes_and_flag_only_filled_rows(self):
        _, data = data_api()
        result = data.build_master(valid_raw())
        minutes = pd.date_range("2023-03-01 00:00", periods=8, freq="1min", tz="UTC")
        expected_prices = pd.Series(
            [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, np.nan, 107.0],
            index=minutes,
            name="binance_btcusdt",
        )
        expected_flags = pd.Series(
            [False, True, True, True, True, True, False, False],
            index=minutes,
            name="binance_btcusdt",
        )

        pd.testing.assert_series_equal(
            result.prices.loc[minutes, "binance_btcusdt"],
            expected_prices,
            check_freq=False,
        )
        pd.testing.assert_series_equal(
            result.price_ffill_flags.loc[minutes, "binance_btcusdt"],
            expected_flags,
            check_freq=False,
        )

    def test_volumes_are_not_forward_filled(self):
        _, data = data_api()
        result = data.build_master(valid_raw())

        self.assertEqual(
            result.volumes.at[
                pd.Timestamp("2023-03-01 00:00", tz="UTC"), "binance_btcusdt"
            ],
            12.0,
        )
        self.assertTrue(
            pd.isna(
                result.volumes.at[
                    pd.Timestamp("2023-03-01 00:01", tz="UTC"),
                    "binance_btcusdt",
                ]
            )
        )
        self.assertEqual(
            result.volumes.at[
                pd.Timestamp("2023-03-01 00:07", tz="UTC"), "binance_btcusdt"
            ],
            19.0,
        )

    def test_adjusted_basis_obeys_decomposition_and_propagates_fill_flags(self):
        _, data = data_api()
        complete_minutes = {
            "2023-03-01 00:00": 100.0,
            "2023-03-01 00:01": 100.0,
        }
        raw = valid_raw(
            {
                "kraken_btcusd": candle_frame(complete_minutes),
                "kraken_btcusdc": candle_frame(
                    {
                        "2023-03-01 00:00": 110.0,
                        "2023-03-01 00:01": 110.0,
                    }
                ),
                "kraken_usdcusd": candle_frame({"2023-03-01 00:00": 0.8}),
            }
        )

        result = data.build_master(raw)
        minute = pd.Timestamp("2023-03-01 00:01", tz="UTC")
        dispersion = result.basis.at[minute, "dispersion_usdc_kraken"]
        adjusted_basis = result.basis.at[minute, "basis_usdc_kraken"]

        self.assertAlmostEqual(dispersion, math.log(1.1) * 10_000)
        self.assertAlmostEqual(adjusted_basis, math.log(0.88) * 10_000)
        self.assertAlmostEqual(adjusted_basis, dispersion + math.log(0.8) * 10_000)
        self.assertFalse(result.basis_ffill_flags.at[minute, "dispersion_usdc_kraken"])
        self.assertTrue(result.price_ffill_flags.at[minute, "kraken_usdcusd"])
        self.assertTrue(result.basis_ffill_flags.at[minute, "basis_usdc_kraken"])

    def test_validate_raw_rejects_a_missing_market(self):
        config, data = data_api()
        raw = valid_raw()
        del raw["kraken_usdcusd"]

        with tempfile.TemporaryDirectory() as directory:
            paths = config.RepoPaths.from_root(Path(directory))
            paths.raw.mkdir()
            for name, frame in raw.items():
                frame.to_parquet(paths.raw / f"{name}.parquet")
            with self.assertRaisesRegex(ValueError, "kraken_usdcusd"):
                data.validate_raw(paths)

    def test_validate_raw_rejects_a_corrupt_candle_schema(self):
        config, data = data_api()
        raw = valid_raw()
        raw["kraken_usdcusd"] = raw["kraken_usdcusd"].drop(columns="close")

        with tempfile.TemporaryDirectory() as directory:
            paths = config.RepoPaths.from_root(Path(directory))
            paths.raw.mkdir()
            for name, frame in raw.items():
                frame.to_parquet(paths.raw / f"{name}.parquet")
            with self.assertRaisesRegex(
                ValueError, "kraken_usdcusd.*close|close.*kraken_usdcusd"
            ):
                data.validate_raw(paths)


if __name__ == "__main__":
    unittest.main()
