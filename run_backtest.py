#!/usr/bin/env python3
import argparse
import asyncio
import datetime
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from market_regime_council.config import BacktestConfig, DataConfig, ALL_TOKENS
from market_regime_council.data_layer.data_manager import DataManager
from market_regime_council.council import RegimeCouncil
from market_regime_council.backtest import BacktestRunner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Market Regime Council — Backtest Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )


    parser.add_argument("--start", default="2023-03-01",
                        help="Backtest start date (YYYY-MM-DD)")

    parser.add_argument("--end", default="2025-12-31",
                        help="Backtest end date (YYYY-MM-DD)")

    parser.add_argument("--lookback", type=int, default=90,
                        help="Lookback window in days (controls OHLCV history + candlestick image selection)")
    parser.add_argument("--kline-style", default="with_ma",
                        choices=["with_ma", "without_ma"],
                        help="Candlestick chart style")
    parser.add_argument("--rebalance-freq", type=int, default=1,
                        help="Rebalancing frequency in days")


    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--use-openrouter", action="store_true",
        help="Use OpenRouter API (requires OPENROUTER_API_KEY or --api-key; "
             "default model: qwen/qwen2.5-vl-32b-instruct)",
    )
    llm_group.add_argument(
        "--use-local-llm", action="store_true",
        help="Use vLLM / SGLang OpenAI-compatible REST API (default: localhost:8000/v1)"
    )
    llm_group.add_argument(
        "--use-api-llm", action="store_true",
        help="Use a remote API LLM (OpenAI / DeepSeek / Anthropic etc.; requires --api-base + --api-key)"
    )
    llm_group.add_argument(
        "--mock", action="store_true",
        help="Use MockLLM — no GPU or API key required; for smoke testing"
    )


    parser.add_argument("--model-name", default="",
                        help="Model name (Ollama default: qwen3-vl:8b; OpenRouter default: qwen/qwen2.5-vl-32b-instruct)")
    parser.add_argument("--api-base", default="",
                        help="API base URL (empty: Ollama→localhost:11434, vLLM→localhost:8000/v1)")
    parser.add_argument("--api-key", default="",
                        help="API key for remote LLM backends")
    parser.add_argument("--max-images", type=int, default=1,
                        help="Number of candlestick images per Ollama multimodal call (default 1)")
    parser.add_argument("--ollama-timeout", type=int, default=600,
                        help="Ollama per-request timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="LLM sampling temperature (0=greedy for reproducibility)")
    parser.add_argument("--seed", type=int, default=4,
                        help="LLM random seed (use with temperature=0 for maximum reproducibility)")


    parser.add_argument(
        "--run-mode",
        choices=["all", "mrc", "baselines"],
        default="baselines",
        help=(
            "Execution mode: "
            "all=MRC+Baselines, "
            "mrc=MRC only (skips all baseline LLM calls), "
            "baselines=Baselines only (MRC uses equal-weight placeholder, no LLM). Default: baselines"
        ),
    )


    parser.add_argument(
        "--baselines",
        nargs="*",
        choices=["fincon", "trading_agents", "finmem", "cryptotrade", "fs_reasoning"],
        # default=["fincon", "trading_agents", "finmem", "cryptotrade", "fs_reasoning"],
        metavar="BASELINE",
        help=(
            "LLM baselines to run (ignored when --run-mode mrc). "
            "Choices: fincon (NeurIPS 2024), trading_agents (ICML 2025 WS), "
            "finmem (AAAI 2024), cryptotrade (EMNLP 2024), "
            "fs_reasoning (ICLR 2025 WS). "
            "Default: none. Example: --baselines fincon finmem cryptotrade"
        ),
    )
    # DRL baselines (no GPU/LLM required; NumPy-only)
    parser.add_argument(
        "--drl-baselines",
        nargs="*",
        choices=["a2c", "ppo", "dqn", "mappo", "maddpg"],
        default=["a2c", "ppo", "dqn", "mappo", "maddpg"],
        metavar="DRL",
        help=(
            "DRL baselines to run (no GPU or LLM required; pure NumPy). "
            "Choices: a2c (Advantage Actor-Critic), ppo (Proximal Policy Optimization), "
            "dqn (Deep Q-Network), mappo (Multi-Agent PPO), maddpg (Multi-Agent DDPG). "
            "Default: all. "
            "Example: --drl-baselines a2c ppo dqn mappo maddpg"
        ),
    )


    parser.add_argument(
        "--allow-short", action="store_true", default=False,
        help="Allow Agent 4 to output negative weights (short positions). Disabled by default (long-only).",
    )
    parser.add_argument(
        "--max-short", type=float, default=0.20,
        help="Maximum absolute short weight per token (default 0.20 = up to 20%% short).",
    )


    parser.add_argument(
        "--smoothing-alpha", type=float, default=0.8,
        help=(
            "EMA smoothing coefficient for portfolio weights (0–1). "
            "1.0=no smoothing, 0.8=fast LLM signal tracking, 0.5=50/50 blend. "
            "Reduces turnover and dampens per-period LLM noise."
        ),
    )


    parser.add_argument("--save-csv", default="",
                        help="Save portfolio_history to CSV")
    parser.add_argument("--save-plot", default="",
                        help="Save cumulative return chart to file")
    parser.add_argument("--save-json", default="",
                        help="Save full summary results to JSON (includes shapley_history); empty=skip")
    parser.add_argument("--save-shapley-csv", default="",
                        help="Save per-period Shapley Value snapshots to CSV")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose logging")

    return parser.parse_args()


async def build_llm(args):
    """"""
    if args.use_openrouter:
        from market_regime_council.llm.api_llm import ApiLLM
        model = args.model_name or "qwen/qwen2.5-vl-32b-instruct"
        llm = ApiLLM(
            api_key=args.api_key,
            base_url="https://openrouter.ai/api/v1",
            model_name=model,
            temperature=args.temperature,
            seed=args.seed,
            force_json=True,
            extra_headers={
                "X-Title": "Market Regime Council",
            },
        )
        print(f"[LLM] OpenRouter: {model}  (temperature={args.temperature}, seed={args.seed})")
        print(f"[LLM] API key   : {'set via arg' if args.api_key else 'read from OPENROUTER_API_KEY / OPENAI_API_KEY'}")

    elif args.use_local_llm:
        from market_regime_council.llm.local_llm import LocalLLM
        api_base = args.api_base or "http://localhost:8000/v1"
        model = args.model_name or "qwen3-vl"
        llm = LocalLLM(api_base=api_base, model_name=model, use_api=True)
        print(f"[LLM] LocalLLM (vLLM/SGLang): {model} @ {api_base}")

    elif args.use_api_llm:
        from market_regime_council.llm.api_llm import ApiLLM
        base_url = args.api_base or "https://api.openai.com/v1"
        model = args.model_name or "gpt-4o"
        llm = ApiLLM(
            api_key=args.api_key,
            base_url=base_url,
            model_name=model,
            temperature=args.temperature,
            seed=args.seed,
            force_json=True,
        )
        print(f"[LLM] Remote API: {model} @ {base_url}  (temperature={args.temperature}, seed={args.seed})")

    elif args.mock:
        from market_regime_council.llm.api_llm import MockLLM
        llm = MockLLM()
        print("[LLM] MockLLM (no GPU/API required)")

    else:
        from market_regime_council.llm.ollama_llm import OllamaLLM
        host = args.api_base or "http://localhost:11434"
        model = args.model_name or "qwen3-vl:8b"
        llm = OllamaLLM(
            model_name=model,
            host=host,
            timeout=args.ollama_timeout,
            force_json=True,
            max_images=args.max_images,
            temperature=args.temperature,
            seed=args.seed,
        )
        print(f"[LLM] Ollama: {model} @ {host}  (force_json=True, max_images={args.max_images}, temperature={args.temperature}, seed={args.seed})")
        print("[LLM] Checking Ollama connection...")
        ok = await llm.health_check()
        if not ok:
            sys.exit(1)
        print("[LLM] Ollama OK\n")

    return llm


async def main_async(args, run_ts: str = None):

    try:
        start_date = datetime.date.fromisoformat(args.start)
        end_date   = datetime.date.fromisoformat(args.end)
    except ValueError as e:
        print(f"[Error] Invalid date format: {e}")
        sys.exit(1)
    if start_date >= end_date:
        print("[Error] start_date must be before end_date")
        sys.exit(1)

    bt_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_freq_days=args.rebalance_freq,
        allow_short=args.allow_short,
        max_short_weight=args.max_short,
    )
    data_config = DataConfig(
        tokens=ALL_TOKENS,
        lookback_days=args.lookback,
        kline_style=args.kline_style,
    )


    llm = await build_llm(args)

    # ---- DataManager ----
    data_manager = DataManager(config=data_config)
    health = await data_manager.health_check()
    print(f"[Data] Health check: {health}")

    # ---- Council ----


    council = RegimeCouncil(
        llm=llm,
        data_manager=data_manager,
        lookback_days=args.lookback,
        allow_short=args.allow_short,
        max_short_weight=args.max_short,
    )
    print("[Council] RegimeCouncil — 3-stage Shapley Value-based reweighting (EW-coalition values, halflife=252)")


    run_mode = args.run_mode  # "all" | "mrc" | "baselines"
    run_main_council = run_mode in ("all", "mrc")   # False → baselines-only

    # ---- Baselines (LLM + DRL) ----


    baseline_councils = {}

    if run_mode != "mrc":

        if args.baselines:
            from market_regime_council.baselines import (
                FinConCouncil, TradingAgentsCouncil, FinMemCouncil,
                CryptoTradeCouncil, FSReasoningAgentCouncil,
            )
            _llm_map = {
                "fincon":         (FinConCouncil,           "FinCon          (NeurIPS 2024)"),
                "trading_agents": (TradingAgentsCouncil,    "TradingAgents   (ICML 2025 Workshop)"),
                "finmem":         (FinMemCouncil,           "FinMem          (AAAI 2024)"),
                "cryptotrade":    (CryptoTradeCouncil,      "CryptoTrade     (EMNLP 2024, reflection)"),
                "fs_reasoning":   (FSReasoningAgentCouncil, "FS-ReasoningAgent (ICLR 2025 WS, fact/subjectivity)"),
            }
            for bl_name in args.baselines:
                cls, desc = _llm_map[bl_name]
                baseline_councils[bl_name] = cls(llm=llm, lookback_days=args.lookback)
                print(f"[Baseline/LLM] {desc}")


        if getattr(args, "drl_baselines", None):
            from market_regime_council.baselines import (
                A2CBaseline, PPOBaseline, DQNBaseline,
                MAPPOBaseline, MADDPGBaseline,
            )
            _drl_map = {
                "a2c":    (A2CBaseline,    "A2C       (Advantage Actor-Critic)"),
                "ppo":    (PPOBaseline,    "PPO       (Proximal Policy Optimization)"),
                "dqn":    (DQNBaseline,    "DQN       (Deep Q-Network, 10 discrete templates)"),
                "mappo":  (MAPPOBaseline,  "MAPPO     (Multi-Agent PPO, 3 temporal agents)"),
                "maddpg": (MADDPGBaseline, "MADDPG    (Multi-Agent DDPG, cooperative)"),
            }
            for drl_name in args.drl_baselines:
                cls, desc = _drl_map[drl_name]
                baseline_councils[drl_name] = cls(lookback=args.lookback, seed=args.seed)
                print(f"[Baseline/DRL] {desc}")

    llm_bl  = [n for n in baseline_councils if n in ("fincon", "trading_agents", "finmem", "cryptotrade", "fs_reasoning")]
    drl_bl  = [n for n in baseline_councils if n in ("a2c", "ppo", "dqn", "mappo", "maddpg")]
    mode_desc = {
        "all": (
            f"MRC + LLM[{', '.join(llm_bl) or 'none'}]"
            f" + DRL[{', '.join(drl_bl) or 'none'}]"
        ),
        "mrc":       "MRC only (no baselines)",
        "baselines": (
            f"Baselines only — LLM[{', '.join(llm_bl) or 'none'}]"
            f" + DRL[{', '.join(drl_bl) or 'none'}]"
        ),
    }[run_mode]
    print(f"[Mode] {mode_desc}")


    _ts = run_ts or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    _incremental_json = os.path.join(_log_dir, f"backtest_{_ts}.json")
    _stages_log = os.path.join(_log_dir, f"backtest_{_ts}_stages.json")
    print(f"[Output] Incremental JSON → {_incremental_json}")
    print(f"[Output] Stages log      → {_stages_log}")

    # ---- Runner ----
    runner = BacktestRunner(
        council=council,
        config=bt_config,
        data_manager=data_manager,
        lookback_days=args.lookback,
        kline_style=args.kline_style,
        baseline_councils=baseline_councils,
        run_main_council=run_main_council,
        smoothing_alpha=args.smoothing_alpha,
        incremental_json_path=_incremental_json,
        stages_log_path=_stages_log,
    )

    print(f"\n[Backtest] {start_date} → {end_date} | "
          f"{len(ALL_TOKENS)} tokens | lookback={args.lookback}d | "
          f"rebalance every {args.rebalance_freq}d\n")

    result = await runner.run_async()


    if args.save_csv:
        import pandas as pd
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        pd.DataFrame(result.portfolio_history).to_csv(args.save_csv, index=False)
        print(f"[Output] portfolio_history → {args.save_csv}")

    if args.save_json:
        os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"[Output] Full results → {args.save_json}")


    os.makedirs(_log_dir, exist_ok=True)
    _auto_plot = os.path.join(_log_dir, f"cumulative_returns_{_ts}.png")
    result.plot_cumulative_returns(save_path=_auto_plot)

    if args.save_plot:
        os.makedirs(os.path.dirname(args.save_plot) or ".", exist_ok=True)
        result.plot_cumulative_returns(save_path=args.save_plot)

    if args.save_shapley_csv and result.shapley_history:
        import pandas as pd
        os.makedirs(os.path.dirname(args.save_shapley_csv) or ".", exist_ok=True)
        rows = []
        for snap in result.shapley_history:
            row = {"eval_date": snap.get("eval_date", "")}
            for a, sv in snap.get("stage1", {}).items():
                row[f"s1_shapley_{a}"] = sv
            for a, w in snap.get("agent_weights_used", {}).items():
                row[f"s1_weight_{a}"] = w
            for p, sv in snap.get("stage2", {}).items():
                row[f"s2_shapley_{p}"] = sv
            for p, w in snap.get("pair_weights_used", {}).items():
                row[f"s2_weight_{p}"] = w
            row["s3_shapley"]       = snap.get("stage3", 0.0)
            row["n_return_periods"] = snap.get("n_return_periods", 0)
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.save_shapley_csv, index=False)
        print(f"[Output] Shapley history → {args.save_shapley_csv}")

    return result


def _setup_auto_log() -> str:
    """"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"backtest_{timestamp}.log")


    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logging.getLogger().addHandler(file_handler)


    dialogue_path = os.path.join(log_dir, f"backtest_{timestamp}_dialogue.log")
    dialogue_handler = logging.FileHandler(dialogue_path, encoding="utf-8")
    dialogue_handler.setLevel(logging.DEBUG)
    dialogue_handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    dialogue_logger = logging.getLogger("llm.dialogue")
    dialogue_logger.addHandler(dialogue_handler)
    dialogue_logger.setLevel(logging.DEBUG)
    dialogue_logger.propagate = False

    return log_path, timestamp


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


    log_path, run_ts = _setup_auto_log()
    log_dir = os.path.dirname(log_path)
    dialogue_path = os.path.join(log_dir, f"backtest_{run_ts}_dialogue.log")
    print(f"[Log] Main log      → {log_path}")
    print(f"[Log] Dialogue log  → {dialogue_path}")

    asyncio.run(main_async(args, run_ts=run_ts))



if __name__ == "__main__":
    main()