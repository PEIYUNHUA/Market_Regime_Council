""""""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..llm.base import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class AgentEvaluation:
    """"""
    agent_name: str
    raw_text: str
    parsed: Dict[str, Any]
    scores: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    key_observations: List[str] = field(default_factory=list)
    # Which data dimension is currently most informative (LLM-voted, independent of market state)
    # technical | onchain | macro | other
    dominant_dimension: str = ""


class BaseAgent(ABC):

    def __init__(self, name: str, llm: BaseLLM, system_prompt: str = ""):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self._history: List[AgentEvaluation] = []

    @abstractmethod
    async def analyze(self, data: Any) -> AgentEvaluation:
        ...

    async def discuss(self, own_eval: AgentEvaluation,
                      other_evals: List[AgentEvaluation]) -> AgentEvaluation:
        """"""
        discussion_prompt = f"You previously assessed:\n{own_eval.raw_text}\n\nOther agents' assessments:\n"
        for ev in other_evals:
            discussion_prompt += f"\n--- {ev.agent_name} ---\n{ev.raw_text}\n"
        discussion_prompt += ("\nConsidering these perspectives, revise your assessment if needed. "
                              "Respond in the same JSON format.")

        response = await self.llm.chat_text(system_prompt=self.system_prompt, user_prompt=discussion_prompt)
        parsed = self._parse_json(response)
        eval_result = AgentEvaluation(
            agent_name=self.name, raw_text=response, parsed=parsed,
            scores=parsed.get("scores", {}),
            confidence=parsed.get("confidence", 0.0),
            key_observations=parsed.get("key_observations", []),
        )
        self._history.append(eval_result)
        return eval_result

    def _parse_json(self, text: str) -> Dict[str, Any]:

        clean = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)

        clean = re.sub(r"<think>[\s\S]*$", "", clean, flags=re.IGNORECASE).strip()


        for candidate in (clean, text):
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass


        for src in (clean, text):
            repaired = BaseAgent._repair_json(src)
            if repaired != src:
                try:
                    result = json.loads(repaired)
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, ValueError):
                    pass


        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean)
        if match:
            try:
                result = json.loads(match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                pass


        for src in (clean, text):
            best = self._extract_best_json_object(src)
            if best:
                try:
                    result = json.loads(best)
                    if isinstance(result, dict) and not result.get("parse_error"):
                        return result
                except (json.JSONDecodeError, ValueError):
                    repaired = self._repair_json(best)
                    try:
                        result = json.loads(repaired)
                        if isinstance(result, dict):
                            return result
                    except (json.JSONDecodeError, ValueError):
                        pass


        for src in (clean, text):
            try:
                import json_repair  # type: ignore
                result = json_repair.repair_json(src, return_objects=True)
                if isinstance(result, dict) and result:
                    return result
            except ImportError:
                break
            except Exception:
                break

        snippet = (clean or text)[:300].replace("\n", " ")
        logger.warning(
            f"[{self.name}] Failed to parse JSON from LLM output. "
            f"First 300 chars: {snippet!r}"
        )
        return {"raw_text": text, "parse_error": True}

    @staticmethod
    def _extract_best_json_object(text: str) -> Optional[str]:
        """"""
        candidates: List[tuple] = []  # (key_count, json_str)
        pos = 0
        while True:
            start = text.find("{", pos)
            if start == -1:
                break
            depth = 0
            in_string = False
            escape_next = False
            end = -1
            for i in range(start, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end == -1:
                break
            candidate = text[start:end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    candidates.append((len(parsed), candidate))
            except (json.JSONDecodeError, ValueError):
                pass
            pos = start + 1

        if not candidates:
            return None

        return max(candidates, key=lambda x: x[0])[1]

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
        """"""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape_next = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    @staticmethod
    def _repair_json(text: str) -> str:
        """"""

        text = re.sub(r",\s*([}\]])", r"\1", text)

        text = re.sub(r"\bTrue\b", "true", text)
        text = re.sub(r"\bFalse\b", "false", text)
        text = re.sub(r"\bNone\b", "null", text)


        text = re.sub(r"(\d)\s+\.(\d)", r"\1.\2", text)

        text = re.sub(r"(?<=\d):(?=\d)", ".", text)

        text = re.sub(r"\b\d+\s*-\s*(\d+\.\d+)\b", r"\1", text)
        return text

    @property
    def history(self) -> List[AgentEvaluation]:
        return self._history
