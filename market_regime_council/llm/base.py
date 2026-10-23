from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


_dialogue_logger = logging.getLogger("llm.dialogue")


@dataclass
class LLMMessage:
    role: str                               # "system" | "user" | "assistant"
    content: Union[str, List[Dict]]


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    usage: Dict[str, int] = field(default_factory=dict)


class BaseLLM(ABC):

    def __init__(self, model_name: str = ""):
        self.model_name = model_name

    @abstractmethod
    async def chat(self, messages: List[LLMMessage], temperature: float = 0.7,
                   max_tokens: int = 4096, **kwargs) -> LLMResponse:
        ...

    async def chat_text(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """"""
        _SEP = "=" * 80
        _dialogue_logger.info(
            "\n%s\n[chat_text] model=%s\n--- SYSTEM PROMPT ---\n%s\n--- USER PROMPT ---\n%s",
            _SEP, self.model_name, system_prompt, user_prompt,
        )
        text = await self._chat_text_core(system_prompt, user_prompt, **kwargs)
        _dialogue_logger.info("--- RESPONSE ---\n%s\n%s", text, _SEP)
        return text

    async def _chat_text_core(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """"""
        msgs = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        resp = await self.chat(msgs, **kwargs)
        return resp.text

    async def chat_multimodal(self, system_prompt: str, text: str,
                              image_paths: List[str] = None, **kwargs) -> str:
        """"""
        _SEP = "=" * 80
        _dialogue_logger.info(
            "\n%s\n[chat_multimodal] model=%s | images=%s\n--- SYSTEM PROMPT ---\n%s\n--- USER PROMPT ---\n%s",
            _SEP, self.model_name, image_paths, system_prompt, text,
        )
        result = await self._chat_multimodal_core(system_prompt, text, image_paths, **kwargs)
        _dialogue_logger.info("--- RESPONSE ---\n%s\n%s", result, _SEP)
        return result

    async def _chat_multimodal_core(self, system_prompt: str, text: str,
                                    image_paths: List[str] = None, **kwargs) -> str:
        """"""
        content: List[Dict] = [{"type": "text", "text": text}]
        for img_path in (image_paths or []):
            content.append({"type": "image_url", "image_url": {"url": f"file://{img_path}"}})
        msgs = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=content),
        ]
        resp = await self.chat(msgs, **kwargs)
        return resp.text
