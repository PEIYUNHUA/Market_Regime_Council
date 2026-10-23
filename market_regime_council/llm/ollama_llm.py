""""""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import time
from typing import List

import requests

from .base import BaseLLM, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class OllamaLLM(BaseLLM):
    """"""

    def __init__(
        self,
        model_name: str = "qwen3-vl:8b",
        host: str = "http://localhost:11434",
        timeout: int = 600,
        force_json: bool = True,
        max_images: int = 2,
        num_predict: int = 2048,
        num_ctx: int = 8192,
        temperature: float = 0.0,
        seed: int = 42,
        think: bool = False,
    ):
        super().__init__(model_name=model_name)
        self.host        = host.rstrip("/")
        self.timeout     = timeout
        self.force_json  = force_json
        self.max_images  = max_images
        self.num_predict = num_predict
        self.num_ctx     = num_ctx
        self.temperature = temperature
        self.seed        = seed
        self.think       = think

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    def _call(self, messages: list) -> str:
        """"""
        url = f"{self.host}/api/chat"

        msgs = list(messages)

        payload: dict = {
            "model":    self.model_name,
            "messages": msgs,
            "stream":   False,
            "options":  {
                "num_predict": self.num_predict,
                "num_ctx":     self.num_ctx,
                "temperature": self.temperature,
                "seed":        self.seed,
            },
        }

        if self.force_json:


            payload["think"] = False
        else:
            payload["think"] = self.think

        r = requests.post(url, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            try:
                err = r.json()
            except Exception:
                err = r.text
            raise RuntimeError(f"Ollama API error {r.status_code}: {err}")

        data    = r.json()
        msg     = data.get("message", {})
        content = msg.get("content", "").strip()

        if not content:

            thinking = msg.get("thinking", "").strip()
            if thinking:
                logger.debug("[OllamaLLM] content empty; using message.thinking")
                content = thinking
            else:
                logger.warning(
                    "[OllamaLLM] Both content and thinking are empty "
                    "(model=%s, num_ctx=%d, num_predict=%d). stop_reason=%s",
                    self.model_name, self.num_ctx, self.num_predict,
                    data.get("done_reason", "?"),
                )

        # Strip <think>...</think> reasoning blocks when force_json=True.
        # Root cause: qwen3-vl outputs <think> in /api/chat content even with
        # think=False for multimodal calls. The thinking block may contain
        # JSON-like snippets ({...}) that fool _parse_json before the real JSON.
        # The model DOES complete the think block + generate JSON within num_predict,
        # but the parser finds false JSON inside <think> first and fails.
        if self.force_json and "<think>" in content:
            stripped = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if stripped:
                logger.debug(
                    "[OllamaLLM] _call: stripped <think> block (%d → %d chars)",
                    len(content), len(stripped),
                )
                content = stripped
            else:
                logger.warning(
                    "[OllamaLLM] _call: <think> stripping removed all content; keeping original"
                )

        logger.debug(
            "[OllamaLLM] response len=%d first50=%r",
            len(content), content[:50],
        )
        return content

    def _call_generate(self, messages: list, images: List[str] = None) -> str:
        """"""
        system_content = ""
        user_content = ""
        for m in messages:
            role = m.get("role", "")
            if role == "system":
                system_content = m.get("content", "")
            elif role == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    user_content = content

        # Qwen3 ChatML template, incomplete assistant turn ends with `{"`.
        # Why `{"` and not just `{`:
        #   `{` alone → model interprets as complete (empty) JSON object and emits
        #   <|im_end|> immediately → Ollama strips special tokens → response = "".
        #   `{"` → model is inside an open string literal; it MUST generate the key
        #   name next — cannot legally output EOS mid-string.
        # The model continues: `key_name": value, ...}`, giving full valid JSON.
        prompt = (
            f"<|im_start|>system\n{system_content}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f'<|im_start|>assistant\n{{"'
        )

        url = f"{self.host}/api/generate"
        payload = {
            "model":  self.model_name,
            "prompt": prompt,
            "raw":    True,
            "stream": False,
            "options": {
                "num_predict": self.num_predict,
                "num_ctx":     self.num_ctx,
                "temperature": self.temperature,
                "seed":        self.seed,
            },
        }

        # Pass images to /api/generate — Ollama processes them through the vision
        # encoder even in raw=True mode (images field is independent of templating)
        if images:
            payload["images"] = images

        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            if r.status_code != 200:
                logger.warning(
                    "[OllamaLLM] _call_generate HTTP %d, falling back to _call()", r.status_code
                )
                return self._call(messages)
            data = r.json()
            response = data.get("response", "").strip()
        except Exception as e:
            logger.warning("[OllamaLLM] _call_generate failed (%s), falling back to _call()", e)
            return self._call(messages)

        if not response:
            # `{"` primary also returned empty (rare).
            # Fallback: use `{` prefix with slightly higher temperature to
            # break the deterministic EOS path.
            logger.warning(
                "[OllamaLLM] _call_generate empty (rare), retrying with '{' + higher temp..."
            )
            retry_prompt = (
                f"<|im_start|>system\n{system_content}<|im_end|>\n"
                f"<|im_start|>user\n{user_content}<|im_end|>\n"
                f"<|im_start|>assistant\n{{"
            )
            retry_payload = dict(payload)
            retry_payload["prompt"] = retry_prompt
            retry_payload["options"] = dict(payload["options"])
            retry_payload["options"]["temperature"] = max(0.3, self.temperature)
            retry_payload["options"]["seed"] = self.seed
            try:
                r2 = requests.post(url, json=retry_payload, timeout=self.timeout)
                if r2.status_code == 200:
                    r2_resp = r2.json().get("response", "").strip()
                    if r2_resp:
                        content = "{" + r2_resp
                        logger.debug(
                            "[OllamaLLM] _call_generate (fallback retry) len=%d first50=%r",
                            len(content), content[:50],
                        )
                        return content
            except Exception as e2:
                logger.warning("[OllamaLLM] _call_generate fallback retry failed (%s)", e2)

        # For multimodal calls: if still empty, images in raw=True mode may
        # cause EOS.  Retry without images (model still has stats table text).
        if not response and images:
            logger.warning(
                "[OllamaLLM] _call_generate: retrying without images (text-only)..."
            )
            payload_no_img = {k: v for k, v in payload.items() if k != "images"}
            try:
                r3 = requests.post(url, json=payload_no_img, timeout=self.timeout)
                if r3.status_code == 200:
                    r3_resp = r3.json().get("response", "").strip()
                    if r3_resp:
                        content = '{"' + r3_resp
                        logger.debug(
                            "[OllamaLLM] _call_generate (no-img retry) len=%d first50=%r",
                            len(content), content[:50],
                        )
                        return content
            except Exception as e3:
                logger.warning("[OllamaLLM] _call_generate no-img retry failed (%s)", e3)

        if not response:
            logger.warning(
                "[OllamaLLM] _call_generate: all attempts empty, falling back to _call()"
            )
            return self._call(messages)

        # Prepend the `{"` we injected as the assistant prefix
        content = '{"' + response
        logger.debug(
            "[OllamaLLM] _call_generate len=%d first50=%r", len(content), content[:50]
        )
        return content

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def chat(self, messages: List[LLMMessage], **kwargs) -> LLMResponse:
        ollama_msgs = []
        for m in messages:
            if isinstance(m.content, str):
                ollama_msgs.append({"role": m.role, "content": m.content})
            else:
                text_parts: List[str] = []
                image_b64s: List[str] = []
                for part in m.content:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:image"):
                            image_b64s.append(url.split(",", 1)[-1])
                        elif url.startswith("file://"):
                            image_b64s.append(_img_to_b64(url.replace("file://", "")))
                msg_dict: dict = {"role": m.role, "content": "\n".join(text_parts)}
                if image_b64s:
                    msg_dict["images"] = image_b64s[: self.max_images]
                ollama_msgs.append(msg_dict)

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, lambda: self._call(ollama_msgs))
        return LLMResponse(text=text, raw={})

    async def _chat_text_core(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        t0 = time.time()
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        loop = asyncio.get_event_loop()
        if self.force_json:
            text = await loop.run_in_executor(None, lambda: self._call_generate(msgs))
        else:
            text = await loop.run_in_executor(None, lambda: self._call(msgs))
        logger.debug("[OllamaLLM] chat_text %.1fs len=%d", time.time() - t0, len(text))
        return text

    async def _chat_multimodal_core(
        self, system_prompt: str, text: str, image_paths: list = None, **kwargs
    ) -> str:
        b64s: List[str] = []
        for p in image_paths:
            if p and os.path.isfile(p):
                b64s.append(_img_to_b64(p))
            if len(b64s) >= self.max_images:
                break

        loop = asyncio.get_event_loop()
        if self.force_json:
            # Route through _call_generate() + images to enforce JSON via prefill.
            # /api/chat with think=False does NOT suppress <think> for multimodal
            # calls — the model consumes all num_predict tokens on thinking content,
            # leaving no room for JSON output.
            # /api/generate processes images via its `images` field independently
            # of the raw=True prompt templating.
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": text},
            ]
            return await loop.run_in_executor(
                None, lambda: self._call_generate(msgs, images=b64s or None)
            )
        else:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": text, "images": b64s},
            ]
            return await loop.run_in_executor(None, lambda: self._call(msgs))

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """"""
        try:
            msgs = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": 'Reply with valid JSON: {"status": "ok"}'},
            ]
            payload = {
                "model": self.model_name, "messages": msgs, "stream": False,
                "think": False, "format": "json",
                "options": {"num_predict": 32, "num_ctx": 512, "temperature": 0.0},
            }
            r = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=30
            )
            r.raise_for_status()
            content = r.json()["message"].get("content", "").strip()
            logger.info("[OllamaLLM] Health check OK: %s", content[:80])
            return True
        except Exception as e:
            logger.error(
                "[OllamaLLM] Cannot reach Ollama at %s — %s: %s\n"
                "  → Make sure Ollama is running: ollama serve\n"
                "  → Pull the model first:        ollama pull %s",
                self.host, type(e).__name__, e, self.model_name,
            )
            return False
