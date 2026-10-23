# Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems

Code for the NeurIPS 2026 paper (poster) by Yunhua Pei, Zerui Ge, Jin Zheng and John Cartlidge.
Paper: https://arxiv.org/abs/2605.24490

Market Regime Council (MRC) assigns credit across specialist LLM agents with exact Shapley values,
computed online over all single, pairwise and grand-coalition portfolios, and turns that credit into
agent weights for daily crypto portfolio decisions over 13 assets.

![MRC framework](assets/framework.jpg)

*The MRC mechanism with N=3 specialist agents for crypto portfolio management.*

---

## Repository layout

```
.
├── run_backtest.py                 Entry point for all backtests
├── requirements.txt
├── assets/framework.jpg            Framework figure from the paper
├── market_regime_council/
│   ├── config.py                   Token universe and global settings
│   ├── agents/                     Market (A1), on-chain (A2), macro (A3) observers and the Agent-4 readout
│   ├── council/
│   │   ├── regime_council.py       Three-stage council, regime-aware multipliers, blend ratios
│   │   └── shapley_tracker.py      Coalition values, exact Shapley values, Bayesian mixture, selective WTA
│   ├── backtest/
│   │   ├── backtest_runner.py      Daily loop, execution overlays, EMA smoothing
│   │   ├── metrics.py              Evaluation metrics
│   │   └── result.py               Tables and plots
│   ├── baselines/                  FinCon, TradingAgents, FinMem, CryptoTrade, FS-ReasoningAgent, A2C, PPO, DQN, MAPPO, MADDPG
│   ├── data_layer/                 Readers for every data source and the DataManager
│   ├── llm/                        Ollama, OpenAI-compatible (vLLM / SGLang), remote API (OpenRouter etc.) and a MockLLM
│   ├── portfolio/                  Weight-driven rebalancing and transaction costs
│   └── prompts/templates.py        All agent prompts
└── data/
    ├── DATA_FORMAT.md              Schema of every input
    └── samples/                    Short excerpts of most inputs
```

Method constants are set as constructor defaults in `RegimeCouncil.__init__`
(`council/regime_council.py`), `ShapleyValueLedger.__init__` (`council/shapley_tracker.py`) and
`BacktestRunner.__init__` (`backtest/backtest_runner.py`). `run_backtest.py` passes some of them
from the command line, for example `--smoothing-alpha`.

---

## Installation

Tested with Python 3.8.

```bash
pip install -r requirements.txt
```

## Smoke test (no data, GPU or API key needed)

```bash
python run_backtest.py --mock --run-mode mrc --start 2023-03-01 --end 2023-03-20
```

Without the data files the run prints warnings about the missing inputs and completes the full
pipeline with placeholder values, which confirms that the installation works.

## Data

All inputs come from public providers: Binance, FRED, blockchain.com, alternative.me, Coin Metrics,
BscScan, DefiLlama, Dune, Polymarket and X. The raw files are not redistributed here because of the
providers' terms of use. `data/DATA_FORMAT.md` documents the source and schema of every input, and
`data/samples/` shows the format of each file.

Place the full files as follows (paths are relative to the repository root):

| Source | Format | Path |
|---|---|---|
| OHLCV, Binance daily bars | HDF5, key `stock_data` | `data/historical_data/{SYM}USDT_*.h5` |
| Candlestick charts | PNG | `data/kline_data/with_ma/{SYM}/{SYM}_{YYYYMMDD}.png` |
| FRED macro indicators | XLSX | `data/macro_data/fred_macro_raw.xlsx` |
| BTC network and market data (blockchain.com, Binance) | XLSX | `data/macro_data/crypto_macro_raw.xlsx` |
| Crypto Fear & Greed Index (alternative.me) | CSV | `data/macro_data/fear_greed_index.csv` |
| Indicator list used to select the macro columns | XLSX | `data/macro_data/macro_intro.xlsx` |
| On-chain metrics per token | XLSX with a 3-row header | `data/on_chain_data/onchain_filtered.xlsx` |
| Polymarket prediction-market prices | CSV | `data/social_media_data/polymarket_crypto_dump/crypto_prices_filtered.csv` |
| Twitter/X posts with sentiment labels | CSV | `data/social_media_data/Tweets/tweets_all.csv` |

Tokens: ADA, BCH, BNB, BTC, DOGE, ETH, LINK, LTC, PAXG, TRX, XLM, XRP, ZEC.
Evaluation window: 2023-03-01 to 2025-12-31 (1,037 daily decisions).

## Running MRC with an LLM

The main runs reported in the paper used a local Ollama server with `qwen3-vl:8b`. A full MRC run
with the paper's primary temperature of 0.7:

```bash
ollama pull qwen3-vl:8b
ollama serve

python run_backtest.py --run-mode mrc --model-name qwen3-vl:8b \
    --temperature 0.7 --seed 4 --smoothing-alpha 0.8 \
    --start 2023-03-01 --end 2025-12-31
```

Pass `--temperature 0.7` explicitly, because the command-line default is 0.9. `--seed 4` and
`--smoothing-alpha 0.8` are the defaults and are shown only for completeness. LLM sampling at
temperature 0.7 is stochastic, so the paper reports the mean and standard deviation over five
runs. On a single NVIDIA RTX 3090 one trading day took about 23 s (median), so a full 1,037-day
run takes about 7 hours.

If the LLM backend cannot be reached or rejects the request, the run does not stop. That day falls
back to an equal-weight portfolio and the log shows `[Backtest] Council failed on <date>`. Check the
log for this line to make sure that every day was decided by the LLM council.

Other backends. The backbone ablation in the paper's appendix ran the other Qwen3-VL variants
through OpenRouter (`qwen/qwen3-vl-8b-instruct`, `qwen/qwen3-vl-8b-thinking`,
`qwen/qwen3-vl-32b-instruct`, `qwen/qwen3-vl-30b-a3b-instruct`, `qwen/qwen3-vl-30b-a3b-thinking`,
`qwen/qwen3-vl-235b-a22b-instruct`, `qwen/qwen3-vl-235b-a22b-thinking`):

```bash
# OpenRouter or another OpenAI-compatible remote API
python run_backtest.py --use-openrouter --api-key $OPENROUTER_API_KEY --model-name qwen/qwen3-vl-32b-instruct ...
# vLLM / SGLang server
python run_backtest.py --use-local-llm --api-base http://localhost:8000/v1 --model-name <model> ...
```

## Baselines

```bash
# DRL baselines only (NumPy). --mock skips the LLM connection check; the DRL baselines never call an LLM.
python run_backtest.py --mock --run-mode baselines --drl-baselines a2c ppo dqn mappo maddpg \
    --start 2023-03-01 --end 2025-12-31

# MRC together with the LLM baselines
python run_backtest.py --run-mode all \
    --baselines fincon trading_agents finmem cryptotrade fs_reasoning \
    --start 2023-03-01 --end 2025-12-31
```

## Outputs

Each run writes to `logs/`:

- `backtest_<timestamp>.log`: the main log, including the per-period causal trace (Shapley values, multipliers, blend ratios and overlay effects);
- `backtest_<timestamp>_dialogue.log`: every prompt and LLM response;
- `backtest_<timestamp>.json` and `backtest_<timestamp>_stages.json`: per-period records and the portfolios of every coalition;
- `cumulative_returns_<timestamp>.png`.

`backtest/metrics.py` reports cumulative return, annualized return, annualized volatility, Sharpe,
Sortino, maximum drawdown, Calmar ratio, annualized turnover, win rate, and information ratio
against the equal-weight portfolio.

## Acknowledgements

This work was funded by UKRI EPSRC (grant No. EP/Y028392/1): AI for Collective Intelligence (AI4CI).

## License

MIT, see `LICENSE`.

## Citation

```bibtex
@misc{pei2026marketregimecouncildynamic,
      title={Market Regime Council for Dynamic Credit Assignment in Multi-Agent LLM Decision Systems}, 
      author={Yunhua Pei and Zerui Ge and Jin Zheng and John Cartlidge},
      year={2026},
      eprint={2605.24490},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.24490}, 
}
```
