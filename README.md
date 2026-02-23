# IAQF Student Competition 2026: Cross-Currency Dynamics

This repository contains the analysis codebase for the 2026 IAQF Student Competition, focusing on cross-currency dynamics in cryptocurrencies under stablecoin regulation (the GENIUS Act context).

## Project Structure
- `data_raw/`: Raw downloaded minute-level OHLCV data.
- `data_processed/`: Cleaned, aligned, and merged master datasets.
- `src/`: Python source code for data fetching, processing, and modeling.
- `figures/`: Output figures generated for the paper.
- `tables/`: Output statistical tables.
- `IAQF_Final.tex`: The LaTeX source code for the final write-up.

## Data Assets
- **Base Asset**: Bitcoin (BTC)
- **Quote Currencies**: USD, EUR, USDT, USDC
- **Exchanges**: Binance, Coinbase, Kraken
- **Sample Window**: March 1, 2023, 00:00:00 UTC to March 21, 2023, 23:59:00 UTC (Event window capturing the SVB crisis and USDC de-peg).

## Requirements
To install the required dependencies:
```bash
pip install -r requirements.txt
```

## Reproduction
To reproduce all data fetching, processing, models, and figures end-to-end, simply run:
```bash
python run_all.py
```
*(Note: Data fetching respects exchange API rate limits and may take some time if `data_raw/` is empty).*
