"""LLM providers. Core modules depend only on :class:`~treeengine.core.protocols.LLMProvider`.

The LLM is used for exactly four things in V0.1: node summaries, structure fallback,
tree-retrieval reasoning and final answers. Every caller must work when ``llm is None``.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.protocols import LLMProvider


class CallableLLM:
    """Wrap any ``fn(prompt, system) -> str`` (handy for tests or custom SDK glue)."""

    def __init__(self, fn: Callable[[str, str | None], str]) -> None:
        self.fn = fn
        self.calls: list[tuple[str, str | None]] = []

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append((prompt, system))
        return self.fn(prompt, system)


class OpenAICompatibleLLM:
    """Minimal ``/v1/chat/completions`` client using only the standard library.

    Works with OpenAI, gateways such as new-api / one-api, vLLM, Ollama (``/v1``), etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.last_usage: dict[str, int] | None = None

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self.extra_body,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        usage = data.get("usage") or {}
        self.last_usage = {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        }
        return str(data["choices"][0]["message"]["content"] or "")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: CJK chars ~1 token each, other text ~4 chars per token."""
    cjk = sum(1 for ch in text if "\u3040" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af")
    return cjk + max(0, len(text) - cjk) // 4


@dataclass
class LLMUsage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated: bool = True  # False once a provider reported real usage for every call
    by_purpose: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> LLMUsage:
        return LLMUsage(
            self.calls,
            self.input_tokens,
            self.output_tokens,
            self.estimated,
            dict(self.by_purpose),
        )

    def since(self, before: LLMUsage) -> LLMUsage:
        purposes = {
            k: v - before.by_purpose.get(k, 0)
            for k, v in self.by_purpose.items()
            if v - before.by_purpose.get(k, 0)
        }
        return LLMUsage(
            self.calls - before.calls,
            self.input_tokens - before.input_tokens,
            self.output_tokens - before.output_tokens,
            self.estimated,
            purposes,
        )


class MeteredLLM:
    """Wraps any provider and counts calls / tokens (real usage when the provider exposes
    ``last_usage``, otherwise an estimate). The factory wraps every LLM in one of these."""

    def __init__(self, inner: LLMProvider) -> None:
        self.inner = inner
        self.usage = LLMUsage()
        self._saw_estimate = False

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        reply = self.inner.complete(
            prompt, system=system, max_tokens=max_tokens, temperature=temperature
        )
        real = getattr(self.inner, "last_usage", None)
        if isinstance(real, dict) and real.get("input_tokens"):
            self.usage.input_tokens += int(real["input_tokens"])
            self.usage.output_tokens += int(real.get("output_tokens") or 0)
        else:
            self._saw_estimate = True
            self.usage.input_tokens += estimate_tokens((system or "") + prompt)
            self.usage.output_tokens += estimate_tokens(reply)
        self.usage.estimated = self._saw_estimate
        self.usage.calls += 1
        purpose = _purpose(system)
        self.usage.by_purpose[purpose] = self.usage.by_purpose.get(purpose, 0) + 1
        return reply


def _purpose(system: str | None) -> str:
    from . import prompts

    return {
        prompts.TREE_SELECT_SYSTEM: "tree_navigation",
        prompts.TREE_REASON_SYSTEM: "tree_reasoning",
        prompts.ANSWER_SYSTEM: "answer",
        prompts.SUMMARY_SYSTEM: "summary",
        prompts.STRUCTURE_SYSTEM: "structure",
        prompts.PLANNER_SYSTEM: "planner",
    }.get(system or "", "other")


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM reply. Returns None on failure."""
    candidates = [m.group(1) for m in _JSON_FENCE.finditer(text)] + [text]
    for c in candidates:
        c = c.strip()
        try:
            return json.loads(c)
        except (json.JSONDecodeError, ValueError):
            pass
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            i, j = c.find(open_ch), c.rfind(close_ch)
            if 0 <= i < j:
                try:
                    return json.loads(c[i : j + 1])
                except (json.JSONDecodeError, ValueError):
                    continue
    return None
