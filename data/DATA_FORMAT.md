# Data Format Reference

This file describes every input the code reads. Short excerpts of most sources are in
`data/samples/` to show the format. Paths below are relative to the repository root.

| # | Source | Path | Used by |
|---|---|---|---|
| 1 | OHLCV daily bars (Binance) | `data/historical_data/{SYM}USDT_*.h5` | A1, execution, all baselines |
| 2 | Candlestick charts | `data/kline_data/with_ma/{SYM}/{SYM}_{YYYYMMDD}.png` | A1 |
| 3 | FRED macro indicators | `data/macro_data/fred_macro_raw.xlsx` | A3 |
| 4 | BTC network and market data (blockchain.com, Binance) | `data/macro_data/crypto_macro_raw.xlsx` | A2 |
| 5 | Crypto Fear & Greed Index (alternative.me) | `data/macro_data/fear_greed_index.csv` | A3 |
| 6 | Indicator list | `data/macro_data/macro_intro.xlsx` | DataManager (selects and cleans the macro columns) |
| 7 | On-chain metrics per token | `data/on_chain_data/onchain_filtered.xlsx` | A2 |
| 8 | Polymarket prediction-market prices | `data/social_media_data/polymarket_crypto_dump/crypto_prices_filtered.csv` | A2 |
| 9 | Twitter/X posts with sentiment labels | `data/social_media_data/Tweets/tweets_all.csv` | A3 |

Tokens: ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC.

---

## 1. OHLCV daily bars (sample: `sample_ohlcv_BTCUSDT.csv`)

**Source:** Binance spot market, daily bars.
**Storage:** one HDF5 file per token, key `stock_data`, named `{SYM}USDT_{N_rows}.h5`. Read with `pandas.read_hdf` (needs the `tables` package).
**Coverage:** BTC from 2017-08-17 to 2025-12-31 (3,059 rows).

| Column | Type | Description |
|---|---|---|
| `Open Time` | date | Bar open date |
| `Open`, `High`, `Low`, `Close` | float | Prices in USDT |
| `Volume` | float | Base-asset volume |
| `Close Time` | datetime | Bar close timestamp |
| `Quote Asset Volume` | float | Quote volume (USDT) |
| `Number of Trades` | int | Trades in the bar |
| `Taker Buy Base Asset Volume`, `Taker Buy Quote Asset Volume` | float | Taker-side buy volume |
| `Ignore` | int | Binance placeholder |

---

## 2. Candlestick charts (sample: `sample_kline_BTC.png`)

**Source:** drawn with `mplfinance` from the OHLCV data.
**Storage:** one PNG per token and date, `data/kline_data/with_ma/{SYM}/{SYM}_{YYYYMMDD}.png`.
**Content:** a 90-day candlestick chart with 5-, 20- and 60-day moving averages and a volume panel.
It is passed to A1 as a base64-encoded image.

---

## 3. FRED macro indicators (sample: `sample_fred_macro.csv`)

**Source:** Federal Reserve Economic Data (FRED). Values are stored as published (levels, not changes).
**Storage:** XLSX, `date` is a plain column. 1,588 rows from 2019-12-01 to 2025-12-31.

| Column | FRED series | Frequency | Description |
|---|---|---|---|
| `us_10y_real_yield` | DFII10 | Daily | 10-year inflation-indexed Treasury yield (%) |
| `usd_index_broad` | DTWEXBGS | Daily | Nominal Broad U.S. Dollar Index |
| `on_rrp` | RRPONTSYD | Daily | Overnight reverse repo, Treasury securities sold by the Fed |
| `vix` | VIXCLS | Daily | CBOE Volatility Index |
| `wti_oil` | DCOILWTICO | Daily | WTI crude oil price (USD per barrel) |
| `cpi` | CPIAUCSL | Monthly | Consumer Price Index, all urban consumers (index) |
| `core_pce` | PCEPILFE | Monthly | PCE excluding food and energy (chain-type price index) |
| `unemployment_rate` | UNRATE | Monthly | Unemployment rate (%) |
| `nonfarm_payrolls` | PAYEMS | Monthly | All employees, total nonfarm (thousands) |
| `industrial_production` | INDPRO | Monthly | Industrial production, total index |

**Cleaning:** daily columns `ffill(limit=5)`, monthly columns `ffill()`.

---

## 4. BTC network and market data (sample: `sample_crypto_macro.csv`)

**Source:** blockchain.com charts (network columns) and Binance (market columns).
**Storage:** XLSX, 2,193 rows from 2019-12-31 to 2025-12-31.

| Column | Source | Description |
|---|---|---|
| `n-unique-addresses` | blockchain.com | Unique addresses used per day |
| `n-transactions` | blockchain.com | Confirmed transactions per day |
| `estimated-transaction-volume-usd` | blockchain.com | Estimated transaction value (USD) |
| `transaction-fees-usd` | blockchain.com | Total transaction fees (USD) |
| `hash-rate` | blockchain.com | Total hash rate (TH/s) |
| `mempool-size` | blockchain.com | Mempool size (bytes) |
| `spot_quote_volume` | Binance | BTC/USDT spot volume |
| `log_return` | Binance | Daily log return of the BTC spot close |
| `realized_vol_30d` | Binance | 30-day realized volatility, annualized |
| `funding_rate` | Binance | Daily average funding rate of the BTC perpetual |
| `basis` | Binance | Relative price gap between the BTC perpetual and spot |

**Cleaning:** `ffill(limit=5)`.

---

## 5. Crypto Fear & Greed Index (sample: `sample_fear_greed_index.csv`)

**Source:** alternative.me.
**Storage:** CSV, newest row first, dates as `DD-MM-YYYY`. Coverage from 2018-02-01.

| Column | Description |
|---|---|
| `date` | Date (`DD-MM-YYYY`) |
| `fng_classification` | Index value, 0 to 100 (integer) |
| `fng_value` | Text label, e.g. `Fear`, `Neutral`, `Greed` |

Note that the column names are swapped relative to their content: the number is in
`fng_classification` and the label in `fng_value`. The code reads them this way.

**Cleaning:** `ffill()`.

---

## 6. Indicator list (`macro_intro.xlsx`, no sample)

One row per macro indicator with columns `id`, `type`, `source`, `index`, `frequency`,
`short_name`, `full_name (description)`, `link`, `threshold`. `DataManager` uses `short_name`
and `frequency` to pick and clean the columns of sources 3 to 5.

---

## 7. On-chain metrics per token (sample: `sample_onchain_filtered.csv`)

**Sources:** Coin Metrics community data, BscScan, DefiLlama (TVL) and Dune (ERC-20 transfers),
depending on the token and metric.
**Storage:** XLSX, 1,831 rows by 40 columns: a date column plus 3 metrics for each of the 13
tokens. Coverage from 2020-12-31 to 2025-12-31.

The first three rows are a header, not data:

| Row | Content |
|---|---|
| 0 | token (`ada`, `bch`, ...) |
| 1 | metric: `AdrActCnt` (active addresses), `TxCnt` (transactions), `TokenTxCnt` (token transfers), `FeeTotNtv` (total fees, native units) or `ChainTVL_USD` (total value locked, USD) |
| 2 | source: `coinmetrics`, `bscscan`, `defillama_tvl` or `dune_erc20` |

Rows from 3 onward hold the daily values.

**Cleaning:** `ffill(limit=5)`.

---

## 8. Polymarket prediction-market prices (sample: `sample_polymarket.csv`)

**Source:** Polymarket API, crypto-related markets.
**Storage:** CSV, 209,128 rows from 2023-02-23 to 2025-12-31.

| Column | Type | Description |
|---|---|---|
| `market_id` | int | Market identifier |
| `slug` | str | Market slug |
| `question` | str | Market question |
| `token_id` | str | CLOB token identifier |
| `ts` | int | Unix timestamp |
| `datetime` | datetime | UTC time of the price |
| `price` | float | Price, read as a probability (0 to 1) |
| `outcome` | str | Outcome label |
| `outcome_norm` | str | Normalized outcome (`yes` / `no`) |
| `date` | date | Observation date |

Aggregated into a per-token daily sentiment signal in `data_layer/social_media_reader.py`.

---

## 9. Twitter/X posts (no sample)

Not included, because the terms of the X platform do not allow redistributing post content.
The code expects a CSV at `data/social_media_data/Tweets/tweets_all.csv`. Our file has 27,103
posts from 2023-01-01 to 2026-02-28. For each decision the code reads only the rows inside that
decision's lookback window (`TweetReader.get_range`).

| Column | Description |
|---|---|
| `date` | Post date |
| `affected_cryptos` | Tokens the post refers to, separated by a vertical bar (`\|`) |
| `sentiment` | `bullish`, `neutral` or `bearish` |
| `confidence` | Confidence of the sentiment label |
| `likes`, `reposts`, `views` | Engagement counts |
| `text` | Post text |

Each post is scored +1 / 0 / −1 by its sentiment and weighted by engagement
(likes + reposts + views / 100), then aggregated per token and day for A3
(see `data_layer/social_media_reader.py`).
