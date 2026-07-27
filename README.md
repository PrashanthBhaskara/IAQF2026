# Cross-Currency Dynamics in Cryptocurrencies

**IAQF Student Competition 2026 | Final Research Repository**

Here is UChicago's submission to the 2026 IAQF where we were eventually selected as one of the winning papers.

This project studies how the March 2023 Silicon Valley Bank crisis transmitted into crypto markets through the USDC stablecoin de-peg. Using 1-minute OHLCV data from Binance, Coinbase, and Kraken, we decompose Bitcoin price fragmentation into a stablecoin peg layer and a crypto-market arbitrage layer, then test where dislocation persisted, where price discovery occurred, and whether arbitrage remained executable after realistic frictions.

**Final deliverables**

- [Final paper PDF](IAQF_Inefficient_Markets_2026.pdf)
- [Final LaTeX source](IAQF_Inefficient_Markets_2026.tex)
- [Reproducible analysis pipeline](run_all.py)
- [Final paper figures](figures_col/)
- [Generated tables and numeric provenance](tables/)

## Executive Summary

When Silicon Valley Bank failed in March 2023, approximately **$3.3B of Circle's USDC reserves** became temporarily inaccessible. USDC fell as low as about **$0.87**, while USDT traded at a premium. This created large apparent Bitcoin price gaps across quote currencies.

Our main finding is that the headline dislocation was not primarily a persistent crypto arbitrage failure. Instead, it was a **two-layer failure chain**:

1. **Peg layer:** USDC's dollar peg deviation persisted for roughly **572 minutes**.
2. **Crypto arbitrage layer:** the adjusted BTC cross-currency basis mean-reverted in roughly **0.6 minutes**.

That gap, about **940x** at the point estimate, localizes the persistent shock to the fiat reserve layer. We then show that the peg shock still contaminated crypto-native markets through liquidity withdrawal, volatility, cross-exchange basis widening, and reduced post-friction arbitrage.

## Core Contributions

1. **Two-layer decomposition:** separates stablecoin-driven nominal dispersion, `D_t`, from the USD-adjusted arbitrage residual, `B_t`.
2. **High-frequency persistence test:** estimates sub-minute mean reversion in adjusted residuals while peg deviations persist for hours.
3. **Contagion intensity model:** shows peg stress significantly predicts changes in adjusted basis during crisis, beyond mechanical stablecoin conversion.
4. **Microstructure evidence:** documents liquidity, volatility, tail-risk, and cross-exchange fragmentation during the SVB window.
5. **Price discovery evidence:** shows fiat BTC/USD remains the primary information anchor, with Hasbrouck information share near **0.73** for Kraken BTC/USD versus BTC/USDT.
6. **Policy mapping:** connects empirical failure channels to reserve, redemption, transparency, and bankruptcy-priority provisions in the GENIUS Act.

## Paper in Brief

### Research Question

Stablecoins are used as quote currencies across crypto exchanges, but they are not identical to bank deposits. Their values can deviate from $1 when reserve access, redemption trust, or issuer transparency deteriorates. The central question is:

**When a stablecoin de-pegs, does crypto-market arbitrage fail, or does the persistent dislocation remain trapped in the stablecoin reserve layer?**

### Data

The sample covers **March 1, 2023 00:00 UTC through March 21, 2023 23:59 UTC**, with the SVB crisis window defined as **March 10-13, 2023 UTC**.

Markets used:

- **Binance:** BTC/USDT, BTC/USDC, USDC/USDT
- **Coinbase:** BTC/USD, BTC/USDT, USDT/USD
- **Kraken:** BTC/USD, BTC/USDT, BTC/USDC, USDC/USD, USDT/USD

All series are aligned to a unified 1-minute UTC grid. Short gaps are forward-filled for up to five minutes and tracked with explicit forward-fill flags.

### Main Objects

For stablecoin quote currency `Q`, we define unadjusted dispersion:

```text
D_Q,t = [log(P_BTC/Q,t) - log(P_BTC/USD,t)] x 10000
```

This measures the visible nominal BTC price difference across quote currencies.

We then adjust the stablecoin leg back to USD:

```text
B_Q,t = [log(P_BTC/Q,t x P_Q/USD,t) - log(P_BTC/USD,t)] x 10000
```

`D_t` captures peg-sensitive quote fragmentation. `B_t` captures the residual arbitrage basis after marking the stablecoin leg to its realized USD value.

### Main Result

During the crisis:

- USDC unadjusted dispersion rises to roughly **+320 bps**.
- USDC adjusted residual remains compressed near **+5 bps**.
- USDC peg deviation half-life is roughly **572 minutes**.
- USDC adjusted residual half-life is roughly **0.6 minutes**.

The implication is direct: cross-market crypto arbitrage corrected quickly, but the stablecoin's underlying reserve-access shock did not.

## Full Paper Narrative

### Abstract

Following the collapse of Silicon Valley Bank in March 2023, USDC traded as low as roughly **$0.87**, creating visible violations of the [law of one price](https://en.wikipedia.org/wiki/Law_of_one_price) across crypto markets. Using 1-minute data from Binance, Coinbase, and Kraken, this paper decomposes Bitcoin price fragmentation into a dominant stablecoin-driven component and a smaller USD-adjusted arbitrage residual.

The headline result is a sharp separation between the two layers. The adjusted BTC basis mean-reverts in about **0.6 minutes** during the crisis, while the USDC peg deviation persists for about **572 minutes**. The submitted bootstrap numbers are now reproduced exactly, and the historical method-label discrepancy is documented explicitly below.

The paper then shows that the reserve shock was not fully quarantined. Peg stress transmits into the adjusted basis through liquidity withdrawal, volatility, cross-exchange basis widening, and sharply reduced post-friction arbitrage. Fiat-quoted BTC/USD remains the main information anchor during the event, while USDC-specific markets show severe tail risk and instability. These empirical channels map directly onto stablecoin policy issues around reserve composition, redemption speed, transparency, and bankruptcy priority.

### 1. Introduction

Stablecoins are the dominant quote currency on many crypto exchanges, but they are not identical to insured bank deposits. When the market questions reserve access or redemption certainty, a stablecoin can deviate from $1 and introduce numeraire risk into every asset quoted against it.

The SVB crisis is a clean natural experiment because Circle disclosed that part of USDC's reserves were held at Silicon Valley Bank. When the bank failed, USDC de-pegged and USDT traded at a premium as traders migrated toward the perceived safer stablecoin. That setting allows the paper to ask whether the observed Bitcoin price gaps were caused by a breakdown in crypto arbitrage itself, or by the stablecoin quote currency losing its peg.

The paper's answer is layered. Nominal BTC price dispersion became very large, but once the stablecoin leg is marked back to its realized USD value, most of the dislocation disappears. The durable failure sits in the reserve-access layer; the crypto arbitrage layer remains fast, though temporarily stressed.

### 2. Data and Processing

The analysis uses 1-minute OHLCV candles from March 1-21, 2023, with March 10-13 marked as the SVB crisis window. The venue selection is intentional:

- **Binance** captures high-volume global USDT trading.
- **Coinbase** captures the U.S. fiat gateway through BTC/USD and USDT/USD.
- **Kraken** provides the most complete set of fiat and stablecoin conversion legs, including BTC/USD, BTC/USDT, BTC/USDC, USDC/USD, and USDT/USD.

All markets are synchronized to a unified UTC minute grid. Short gaps are forward-filled for at most five minutes and tracked with explicit flags, while longer gaps remain missing. This matters because thin crisis markets can create artificial persistence if missing data are handled carelessly. The robustness checks repeat the core estimates under no-forward-fill filters and 5-minute sampling.

### 3. Quantitative Framework

The core measurement problem is that a BTC/USDC price can move either because Bitcoin moved or because USDC moved. The paper separates those effects with two objects:

```text
D_Q,t = [log(P_BTC/Q,t) - log(P_BTC/USD,t)] x 10000
```

`D_Q,t` is the visible, unadjusted dispersion across quote currencies. It is what a trader sees when comparing BTC quoted in a stablecoin to BTC quoted in dollars.

```text
B_Q,t = [log(P_BTC/Q,t x P_Q/USD,t) - log(P_BTC/USD,t)] x 10000
```

`B_Q,t` marks the stablecoin leg back to its actual dollar value. This is the cleaner arbitrage residual. If `D_t` is huge but `B_t` is small, the price gap is mostly a stablecoin-peg problem rather than a persistent BTC arbitrage problem.

To estimate persistence, the paper models deviations with a discrete-time AR(1) representation of an [Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process):

```text
X_t = c + rho X_{t-1} + epsilon_t
half-life = log(2) * Delta t / -log(rho)
```

Because peg deviations are close to unit-root behavior, the submitted paper reports a bootstrap interval rather than relying only on parametric uncertainty. Git history shows that the submitted numbers came from a 10,000-draw parametric AR(1) residual sieve bootstrap; a later edit relabeled the unchanged numbers as a 5,000-draw moving-block result. The pipeline executes both methods and records the difference.

The contagion model then tests whether peg stress predicts changes in the adjusted arbitrage basis:

```text
Delta B_t = a + beta B_{t-1} + lambda S_{t-1} + epsilon_t
```

Here, `S_t` is the stablecoin peg deviation. A significant `lambda` means stablecoin reserve stress is not merely a mechanical conversion issue; it actively leaks into crypto-market microstructure.

### 4. Empirical Results

**Dispersion decomposition.** During the crisis, Kraken USDC unadjusted dispersion averages roughly **+320 bps**, but the adjusted USDC residual averages only about **+5.3 bps**. USDT moves in the opposite direction, with a safe-haven premium visible in the unadjusted measure and a much smaller adjusted residual. This is the main evidence that the large headline price gaps were primarily stablecoin numeraire effects.

**Two-layer persistence.** The adjusted USDC residual mean-reverts in about **0.6 minutes**, while the USDC peg deviation has a half-life near **572 minutes**. The estimated ratio is roughly **940x**. The recovered sieve bootstrap reproduces **567 [216, 1,898]** and rejects equal persistence. This is the paper's central result.

**Contagion intensity.** Before SVB, the USDC peg-stress transmission parameter is statistically indistinguishable from zero. During the crisis, it becomes highly significant. The sign implies that a deeper USDC discount predicts upward pressure in the adjusted residual, consistent with liquidity withdrawal from USDC order books. USDT shows the mirror pattern, consistent with safe-haven demand.

**Cross-exchange dispersion.** The crisis is not isolated to one venue. Cross-exchange BTC/USDT and BTC/USD bases widen, and the Coinbase-Kraken BTC/USD channel shows a persistent fiat gateway premium. This suggests the stablecoin shock interacted with broader exchange-level fragmentation.

**Market microstructure.** Kraken BTC/USDC becomes dramatically more expensive to trade. The Roll effective spread rises from about **1.10 bps** pre-crisis to about **25.64 bps** during the crisis, while BTC/USDT remains much more stable. Daily dollar-volume proxies show that Kraken BTC/USDC is about **26x thinner** than the Coinbase BTC/USD fiat channel during the event.

**Tail risk and correlation.** Crisis USDC adjusted residuals develop extreme tails, with excess kurtosis around **12.9** and a 99th percentile that rises from about **13.6 bps** to **81.4 bps**. Cross-stablecoin adjusted-basis correlation falls from **0.44** before SVB to **0.16** during the crisis, supporting the interpretation that USDC and USDT did not simply respond to a symmetric common shock.

**Price discovery.** Cointegration and VECM evidence show that BTC/USD and BTC/USDT retain a shared long-run price relationship during the crisis. Hasbrouck information-share estimates place BTC/USD leadership around **0.73**, meaning fiat-quoted BTC/USD contributes about 73% of permanent price discovery against BTC/USDT. BTC/USDC is too distressed under strict no-forward-fill filtering to support the same stable cointegration relationship.

**Arbitrage after fees.** Apparent arbitrage opportunities shrink sharply after trading costs and range-based slippage proxies. USDC intra-exchange crisis profitability falls from **29.16%** of minutes under fee-only costs to **6.72%** after fees plus slippage. This explains why large visible dislocations can persist in a headline sense even when adjusted arbitrage residuals are corrected quickly.

### 5. Regulatory Interpretation

The empirical failure chain is:

```text
reserve-access shock -> stablecoin peg deviation -> quote-currency dispersion -> liquidity stress -> reduced executable arbitrage
```

This maps naturally to the GENIUS Act's focus on reserve composition, timely redemption, transparency, and bankruptcy priority. In the paper's counterfactual framing, partial mitigation of the reserve-lockup shock would mechanically compress the observed `D_t` dispersion. However, the evidence also suggests that reserve regulation alone may not eliminate all crypto-market fragmentation, because cross-exchange basis and depth differences persist after the acute peg event.

### 6. Conclusion

The SVB-USDC event did not reveal a simple, persistent failure of crypto arbitrage. It revealed a more specific failure chain: reserve-access risk destabilized the stablecoin quote currency, that peg shock created large nominal BTC price gaps, and market frictions limited how much of the opportunity could be executed in real time.

The adjusted arbitrage layer remained fast, but not frictionless. The reserve layer remained slow, persistent, and systemically important. That distinction is the main contribution of the project.

## Final Figures

These are the exact figures referenced by the final paper.

| Figure | What It Shows |
|---|---|
| ![Stablecoin peg](figures_col/fig_stablecoin_peg.png) | USDC breaks below par while USDT trades at a premium. |
| ![Dispersion decomposition](figures_col/fig_dispersion_vs_adjusted_kraken.png) | Large unadjusted USDC dispersion collapses after USD adjustment. |
| ![Stablecoin substitution](figures_col/fig_stablecoin_substitution_scatter.png) | Crisis observations shift toward USDC discount and USDT premium behavior. |
| ![Two-layer persistence](figures_col/fig_two_layer_persistence.png) | Adjusted basis mean-reverts quickly while peg deviation persists. |
| ![Half-life robustness](figures_col/fig_half_life_robustness.png) | Half-life estimates are stable across sampling and no-forward-fill filters. |
| ![Crisis zoom](figures_col/fig_svb_crisis_zoom.png) | Minute-level view of adjusted residuals and cross-exchange basis during crisis. |
| ![Cross-exchange basis](figures_col/fig_cross_exchange_basis.png) | Cross-exchange BTC basis widens during stress. |
| ![Liquidity](figures_col/fig_liquidity_roll_amihud.png) | Roll spread and Amihud illiquidity by pair and regime. |
| ![Volume share](figures_col/fig_volume_share.png) | Trading volume rotates across quote channels during stress. |
| ![Realized volatility](figures_col/fig_realized_volatility.png) | BTC/USDC volatility spikes far above USD and USDT channels. |
| ![Tail blowout](figures_col/fig_tail_blowout_kde.png) | USDC adjusted residual develops severe crisis tails. |
| ![Correlation heatmap](figures_col/fig_correlation_regime_heatmap.png) | Cross-stablecoin correlation collapses during crisis. |
| ![VAR IRF](figures_col/fig_var_irf.png) | BTC/USD shocks transmit to BTC/USDC more than the reverse. |
| ![Arbitrage after fees](figures_col/fig_arbitrage_after_fees.png) | Apparent arbitrage compresses sharply after fees and slippage proxies. |

## Repository Structure

```text
.
├── iaqf/
│   ├── config.py                  # fixed study and artifact contract
│   ├── data.py                    # raw refresh, validation, and processing
│   ├── metrics.py                 # statistical and microstructure calculations
│   ├── tables.py                  # deterministic table serialization
│   ├── figures.py                 # deterministic plotting
│   └── validation.py              # repository and paper checks
├── run_all.py                     # single reproduction entry point
├── data_raw/                      # 11 committed raw Parquet files
├── data_processed/                # six processed Parquet files
├── tables/                        # 44 generated tables, diagnostics, and audit artifacts
├── figures_col/                   # 14 paper PNGs
├── tests/                         # unit, contract, golden, and integration tests
├── pyproject.toml                 # exact Python and dependency pins
├── uv.lock                        # locked dependency graph
├── IAQF_Inefficient_Markets_2026.tex  # immutable submitted source
└── IAQF_Inefficient_Markets_2026.pdf  # immutable submitted paper
```

## Reproduce the Results

### 1. Install the pinned environment

The reference environment is Python **3.12.4** with
[uv](https://docs.astral.sh/uv/) **0.11.5**.

```bash
uv sync --frozen
```

### 2. Run the full pipeline

Activate the environment, then use the repository's only execution entry point:

```bash
source .venv/bin/activate
python run_all.py
```

The default run is offline: it validates the 11 committed raw files, rebuilds the
six processed frames, writes all 44 table and diagnostic artifacts, regenerates
all 14 figures, and validates the complete artifact surface. Paths are anchored
to the checkout, so the command also works when invoked from another working
directory.

Network retrieval is opt-in:

```bash
python run_all.py --refresh-data
```

The refresh flag replaces the raw cache before running the same deterministic
pipeline. It is unnecessary for reproducing the committed study.

### 3. Validate the repository

```bash
uv run ruff check .
uv run python -m pytest -q
```

The full suite includes exact table and reference-render PNG contracts plus an
isolated two-run idempotence test. CI runs portable checks on Linux and the
golden/integration reproduction checks on the pinned macOS reference platform.

The submitted `IAQF_Inefficient_Markets_2026.tex` and
`IAQF_Inefficient_Markets_2026.pdf` are immutable.
The pipeline validates their SHA-256 hashes and never formats, edits, or compiles
over them.

### Reproduction provenance

The complete-case return sample used by HAC and Granger calculations is restored
from repository history. It reproduces `N=26,489` for both HAC channels and the
reverse-Granger q-value `0.127898`; other paper-facing semantics, including the
Kraken end boundary and existing missing-data choices, remain unchanged.

The history audit found that the March snapshot in this repository contains
byte-identical copies of the final February producer scripts and processed data
from the legacy repository. It also resolves the bootstrap mystery:

- The February manuscript described a **parametric AR(1) residual sieve
  bootstrap** with 10,000 draws. Reconstructing that stated algorithm with the
  historical seed reproduces **567 [216, 1,898]** exactly after paper rounding.
- The final manuscript changed the label to a 5,000-draw, 60-minute
  moving-block bootstrap without changing the reported numbers.
- The committed moving-block implementation produces approximately
  **64 [46, 88]**. This is a method mismatch in the submitted text, not a
  numerical failure of the recovered sieve producer.

The pipeline now runs both algorithms and writes their results to
`tables/half_life_ratio_bootstrap.csv`. It also writes
`tables/paper_claim_audit.csv`, which records one additional substantive
shortfall: the paper says the reverse BTC/USDC-to-BTC/USD IRF confidence interval
contains zero at every horizon, but the plotted model's horizon 3 interval is
`[-0.237665, -0.010469]`. The forward impact response of about 6.13 bps and the
rest of the mapped headline results reproduce.

## Methods Glossary

For readers who want quick conceptual background:

- [Stablecoin](https://en.wikipedia.org/wiki/Stablecoin): digital token designed to track a reference asset such as the U.S. dollar.
- [USDC](https://en.wikipedia.org/wiki/USD_Coin): Circle-issued dollar stablecoin studied in the SVB event.
- [Collapse of Silicon Valley Bank](https://en.wikipedia.org/wiki/Collapse_of_Silicon_Valley_Bank): March 2023 banking shock that triggered the USDC reserve-access concern.
- [Law of one price](https://en.wikipedia.org/wiki/Law_of_one_price): the arbitrage principle behind comparing BTC prices across quote currencies.
- [Arbitrage](https://en.wikipedia.org/wiki/Arbitrage): simultaneous or near-simultaneous trades exploiting price discrepancies.
- [Market microstructure](https://en.wikipedia.org/wiki/Market_microstructure): study of trading frictions, liquidity, spreads, and order-flow effects.
- [Ornstein-Uhlenbeck process](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process): continuous-time mean-reversion model used to motivate basis half-life.
- [Autoregressive model](https://en.wikipedia.org/wiki/Autoregressive_model): discrete-time model used to estimate half-life.
- [Half-life](https://en.wikipedia.org/wiki/Half-life): time required for a deviation to decay by half under mean reversion.
- [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)): resampling method used for robust uncertainty estimates.
- [Newey-West estimator](https://en.wikipedia.org/wiki/Newey%E2%80%93West_estimator): HAC standard-error estimator used for high-frequency serial correlation.
- [Cointegration](https://en.wikipedia.org/wiki/Cointegration): long-run relationship among nonstationary price series.
- [Vector error correction model](https://en.wikipedia.org/wiki/Error_correction_model): model used when cointegrated series adjust toward equilibrium.
- [Granger causality](https://en.wikipedia.org/wiki/Granger_causality): predictive lead-lag test used as secondary evidence for price discovery.
- [Impulse response](https://en.wikipedia.org/wiki/Impulse_response): dynamic response of one series to a shock in another.
- [Chow test](https://en.wikipedia.org/wiki/Chow_test): structural-break test used to validate the SVB regime split.
- [Skewness](https://en.wikipedia.org/wiki/Skewness), [kurtosis](https://en.wikipedia.org/wiki/Kurtosis), and [Jarque-Bera test](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test): distributional diagnostics for crisis tail risk.

## Citations and Source Links

### Event and Policy Sources

- Circle. **"$3.3 Billion of USDC Reserve Risk Removed, Dollar De-peg Closes"**. Circle Pressroom, March 13, 2023. [Official Circle link](https://www.circle.com/pressroom/3-3-billion-of-usdc-reserve-risk-removed-dollar-de-peg-closes)
- Federal Reserve, Treasury, and FDIC. **"Joint Statement by Treasury, Federal Reserve, and FDIC"**. March 12, 2023. [Official Federal Reserve link](https://www.federalreserve.gov/newsevents/pressreleases/monetary20230312b.htm)
- U.S. Congress. **Public Law 119-27: Guiding and Establishing National Innovation for U.S. Stablecoins Act (GENIUS Act)**. July 18, 2025. [Congress PDF](https://www.congress.gov/119/plaws/publ27/PLAW-119publ27.pdf) | [Congress bill page](https://www.congress.gov/bill/119th-congress/senate-bill/1582)
- The White House. **"Fact Sheet: President Donald J. Trump Signs GENIUS Act into Law"**. July 18, 2025. [White House link](https://www.whitehouse.gov/fact-sheets/2025/07/fact-sheet-president-donald-j-trump-signs-genius-act-into-law/)

### Academic References

- Amihud, Y. (2002). **"Illiquidity and stock returns: Cross-section and time-series effects."** *Journal of Financial Markets*, 5(1), 31-56. [DOI](https://doi.org/10.1016/S1386-4181(01)00024-6)
- Chow, G. C. (1960). **"Tests of equality between sets of coefficients in two linear regressions."** *Econometrica*, 28(3), 591-605. [DOI](https://doi.org/10.2307/1910133) | [Econometric Society](https://www.econometricsociety.org/publications/econometrica/1960/07/01/tests-equality-between-sets-coefficients-two-linear-regressions)
- Engle, R. F., & Granger, C. W. J. (1987). **"Co-integration and error correction: Representation, estimation, and testing."** *Econometrica*, 55(2), 251-276. [DOI](https://doi.org/10.2307/1913236) | [Econometric Society](https://www.econometricsociety.org/publications/econometrica/1987/03/01/co-integration-and-error-correction-representation-estimation)
- Granger, C. W. J. (1969). **"Investigating causal relations by econometric models and cross-spectral methods."** *Econometrica*, 37(3), 424-438. [DOI](https://doi.org/10.2307/1912791) | [Econometric Society](https://www.econometricsociety.org/publications/econometrica/browse/1969/08/01/investigating-causal-relations-econometric-models-and-cross)
- Hasbrouck, J. (1995). **"One security, many markets: Determining the contributions to price discovery."** *Journal of Finance*, 50(4), 1175-1199. [DOI](https://doi.org/10.1111/j.1540-6261.1995.tb04054.x)
- Johansen, S. (1988). **"Statistical analysis of cointegration vectors."** *Journal of Economic Dynamics and Control*, 12(2-3), 231-254. [DOI](https://doi.org/10.1016/0165-1889(88)90041-3) | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0165188988900413)
- Newey, W. K., & West, K. D. (1987). **"A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix."** *Econometrica*, 55(3), 703-708. [DOI](https://doi.org/10.2307/1913610) | [NBER working paper](https://www.nber.org/papers/t0055)
- Roll, R. (1984). **"A simple implicit measure of the effective bid-ask spread in an efficient market."** *Journal of Finance*, 39(4), 1127-1139. [DOI](https://doi.org/10.1111/j.1540-6261.1984.tb03897.x)

## Recruiter Notes

This project demonstrates:

- event-study design around a real market stress event;
- high-frequency financial data engineering;
- careful treatment of missing data and forward-fill exposure;
- econometric modeling with HAC inference, VECM, VAR, Granger tests, and an executable two-method bootstrap audit;
- market microstructure reasoning around liquidity, spreads, volatility, and executable arbitrage;
- policy interpretation connecting empirical findings to stablecoin regulation;
- fully scripted reproduction of processed data, tables, diagnostics, and all
  final figures.

## Disclaimer

This repository is an academic research project. It is not investment, legal, or regulatory advice.
