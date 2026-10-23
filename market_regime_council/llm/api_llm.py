""""""
from __future__ import annotations

import base64
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from .base import BaseLLM, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class ApiLLM(BaseLLM):
    """"""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://openrouter.ai/api/v1",
        model_name: str = "qwen/qwen2.5-vl-32b-instruct",
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 2048,
        force_json: bool = True,
        timeout: int = 180,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model_name=model_name)

        self.api_key     = api_key or os.environ.get("OPENROUTER_API_KEY", "") \
                                   or os.environ.get("OPENAI_API_KEY", "")
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.seed        = seed
        self.max_tokens  = max_tokens
        self.force_json  = force_json
        self.timeout     = timeout
        self.extra_headers = extra_headers or {}

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _build_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ValueError(
                "ApiLLM: api_key not set.\n"
                "  Option 1: --api-key sk-or-...\n"
                "  Option 2: export OPENROUTER_API_KEY=sk-or-..."
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        headers.update(self.extra_headers)
        return headers

    @staticmethod
    def _encode_image(path: str) -> str:
        """"""
        ext = os.path.splitext(path)[-1].lower().lstrip(".")
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png",  "webp": "image/webp",
                "gif": "image/gif"}.get(ext, "image/png")
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    def _to_api_messages(self, messages: List[LLMMessage]) -> List[Dict]:
        """"""
        api_msgs = []
        for m in messages:
            if isinstance(m.content, str):
                api_msgs.append({"role": m.role, "content": m.content})
            else:
                # multimodal content list
                parts = []
                for part in m.content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("file://"):

                            local_path = url.replace("file://", "")
                            url = self._encode_image(local_path)
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                api_msgs.append({"role": m.role, "content": parts})
        return api_msgs

    def _build_payload(self, api_messages: List[Dict]) -> Dict:
        payload: Dict[str, Any] = {
            "model":       self.model_name,
            "messages":    api_messages,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
            "seed":        self.seed,
        }
        if self.force_json:
            payload["response_format"] = {"type": "json_object"}
        return payload

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        import httpx

        api_messages = self._to_api_messages(messages)
        payload = self._build_payload(api_messages)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._build_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        text = data["choices"][0]["message"]["content"] or ""
        return LLMResponse(text=text, raw=data, usage=data.get("usage", {}))

    async def _chat_multimodal_core(
        self, system_prompt: str, text: str,
        image_paths: List[str] = None, **kwargs
    ) -> str:
        """"""
        import httpx

        content: List[Dict] = [{"type": "text", "text": text}]
        for p in (image_paths or []):
            if p and os.path.isfile(p):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._encode_image(p)},
                })

        api_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content},
        ]
        payload = self._build_payload(api_messages)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._build_headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"] or ""


class MockLLM(BaseLLM):
    """"""

    def __init__(self):
        super().__init__(model_name="mock-llm")

    async def chat(self, messages: List[LLMMessage], temperature: float = 0.7,
                   max_tokens: int = 4096, **kwargs) -> LLMResponse:
        system_text = ""
        for m in messages:
            content = m.content if isinstance(m.content, str) else str(m.content)
            if m.role == "system":
                system_text = content

        if "investment" in system_text.lower() or "agent 4" in system_text.lower():
            return LLMResponse(text=self._mock_investment_decision())
        elif "coordinator" in system_text.lower() or "discussion coordinator" in system_text.lower():
            return LLMResponse(text=self._mock_discussion_summary())
        elif "market observer" in system_text.lower():
            return LLMResponse(text=self._mock_market_analysis())
        elif "on-chain" in system_text.lower():
            return LLMResponse(text=self._mock_onchain_analysis())
        elif "environment observer" in system_text.lower():
            return LLMResponse(text=self._mock_environment_analysis())
        else:
            return LLMResponse(text=self._mock_generic())

    def _mock_market_analysis(self) -> str:
        regime = random.choice(["bullish", "bearish", "sideways", "high_volatility"])
        return json.dumps({
            "agent": "market_observer",
            "regime_assessment": regime,
            "scores": {
                "market": round(random.uniform(0.1, 0.5), 2),
                "trade": round(random.uniform(0.05, 0.3), 2),
                "environment": round(random.uniform(0.1, 0.4), 2),
                "other": round(random.uniform(0.01, 0.1), 2),
            },
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "key_observations": [
                f"BTC showing {regime} pattern on 4h chart",
                "Volume increasing on recent candles",
                f"RSI at {random.randint(25, 75)}, indicating moderate momentum",
            ],
            "token_signals": {
                "BTC": random.choice(["buy", "hold", "sell"]),
                "ETH": random.choice(["buy", "hold", "sell"]),
                "SOL": random.choice(["buy", "hold", "sell"]),
            },
        }, indent=2)

    def _mock_onchain_analysis(self) -> str:
        return json.dumps({
            "agent": "onchain_observer",
            "scores": {
                "market": round(random.uniform(0.05, 0.3), 2),
                "trade": round(random.uniform(0.2, 0.5), 2),
                "environment": round(random.uniform(0.05, 0.2), 2),
                "other": round(random.uniform(0.01, 0.1), 2),
            },
            "confidence": round(random.uniform(0.5, 0.9), 2),
            "key_observations": [
                "Whale accumulation detected for BTC",
                "Exchange outflow increasing (bullish signal)",
                f"Social sentiment score: {round(random.uniform(-0.5, 0.8), 2)}",
            ],
        }, indent=2)

    def _mock_environment_analysis(self) -> str:
        return json.dumps({
            "agent": "environment_observer",
            "scores": {
                "market": round(random.uniform(0.05, 0.2), 2),
                "trade": round(random.uniform(0.05, 0.15), 2),
                "environment": round(random.uniform(0.3, 0.6), 2),
                "other": round(random.uniform(0.01, 0.1), 2),
            },
            "confidence": round(random.uniform(0.5, 0.85), 2),
            "key_observations": [
                "DXY weakening - positive for crypto",
                "VIX stable, low market fear",
                "Fed holding rates steady this quarter",
            ],
        }, indent=2)

    def _mock_investment_decision(self) -> str:
        return json.dumps({
            "agent": "investment_analyst",
            "final_regime": random.choice(["bullish", "bearish", "sideways"]),
            "dominator": {
                "market": round(random.uniform(0.15, 0.35), 2),
                "trade": round(random.uniform(0.1, 0.3), 2),
                "environment": round(random.uniform(0.2, 0.4), 2),
                "other": round(random.uniform(0.01, 0.1), 2),
            },
            "recommendations": [
                {"token": "BTC", "action": random.choice(["buy", "hold", "sell"]),
                 "weight": round(random.uniform(0.1, 0.4), 2)},
                {"token": "ETH", "action": random.choice(["buy", "hold", "sell"]),
                 "weight": round(random.uniform(0.1, 0.3), 2)},
                {"token": "SOL", "action": random.choice(["buy", "hold", "sell"]),
                 "weight": round(random.uniform(0.05, 0.2), 2)},
            ],
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "risk_level": random.choice(["low", "medium", "high"]),
        }, indent=2)

    def _mock_discussion_summary(self) -> str:
        return json.dumps({
            "discussion_round": 1,
            "consensus_reached": random.choice([True, False]),
            "summary": "Agents discussed current market conditions and synthesized views.",
            "revised_weights": {
                "market": round(random.uniform(0.15, 0.4), 2),
                "trade": round(random.uniform(0.1, 0.3), 2),
                "environment": round(random.uniform(0.2, 0.4), 2),
                "other": round(random.uniform(0.01, 0.1), 2),
            },
        }, indent=2)

    def _mock_generic(self) -> str:
        return json.dumps({"response": "Mock LLM generic response", "status": "ok"})
