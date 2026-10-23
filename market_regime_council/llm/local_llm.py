""""""
from __future__ import annotations

import base64
import logging
from typing import Any, Dict, List

from .base import BaseLLM, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)


class LocalLLM(BaseLLM):

    def __init__(self, model_path: str = "", model_name: str = "qwen3-vl",
                 api_base: str = "http://localhost:8000/v1",
                 device: str = "cuda", dtype: str = "bfloat16", use_api: bool = True):
        super().__init__(model_name=model_name)
        self.model_path = model_path
        self.api_base = api_base.rstrip("/")
        self.device = device
        self.dtype = dtype
        self.use_api = use_api
        self._model = None
        self._processor = None

    async def chat(self, messages: List[LLMMessage], temperature: float = 0.7,
                   max_tokens: int = 4096, **kwargs) -> LLMResponse:
        if self.use_api:
            return await self._chat_via_api(messages, temperature, max_tokens, **kwargs)
        else:
            return await self._chat_via_transformers(messages, temperature, max_tokens, **kwargs)


    async def _chat_via_api(self, messages: List[LLMMessage], temperature: float,
                            max_tokens: int, **kwargs) -> LLMResponse:
        import httpx  # pip install httpx

        api_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                api_messages.append({"role": msg.role, "content": msg.content})
            else:
                content_parts = []
                for part in msg.content:
                    if part["type"] == "text":
                        content_parts.append(part)
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("file://"):
                            img_path = url.replace("file://", "")
                            b64 = self._encode_image(img_path)
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            })
                        else:
                            content_parts.append(part)
                api_messages.append({"role": msg.role, "content": content_parts})

        payload = {
            "model": self.model_name, "messages": api_messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.api_base}/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return LLMResponse(text=data["choices"][0]["message"]["content"],
                           raw=data, usage=data.get("usage", {}))


    def _init_transformers(self):
        """"""
        if self._model is not None:
            return
        logger.info(f"Loading model from {self.model_path} ...")


        # from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        # import torch
        # dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        # self._model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     self.model_path, torch_dtype=dtype_map.get(self.dtype, torch.bfloat16),
        #     device_map="auto", attn_implementation="flash_attention_2",
        # )
        # self._processor = AutoProcessor.from_pretrained(self.model_path)

        logger.warning("Transformers direct loading not yet enabled. "
                       "Uncomment _init_transformers() code to activate.")

    async def _chat_via_transformers(self, messages: List[LLMMessage], temperature: float,
                                     max_tokens: int, **kwargs) -> LLMResponse:
        self._init_transformers()


        # from qwen_vl_utils import process_vision_info
        # formatted = []
        # for msg in messages:
        #     if isinstance(msg.content, str):
        #         formatted.append({"role": msg.role, "content": [{"type": "text", "text": msg.content}]})
        #     else:
        #         formatted.append({"role": msg.role, "content": msg.content})
        # text_input = self._processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=True)
        # image_inputs, video_inputs = process_vision_info(formatted)
        # inputs = self._processor(text=[text_input], images=image_inputs, videos=video_inputs,
        #                          padding=True, return_tensors="pt").to(self.device)
        # output_ids = self._model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature)
        # output_text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        # return LLMResponse(text=output_text)

        return LLMResponse(text="[LocalLLM transformers mode not yet enabled]")

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
