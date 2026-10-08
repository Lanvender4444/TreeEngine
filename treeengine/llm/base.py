"""Pluggable LLM provider protocol. Core modules depend only on :class:`LLMProvider`.

The LLM is used for exactly four things in V0.1: node summaries, structure fallback,
tree-retrieval reasoning and final answers. Every caller must work when ``llm is None``.
"""

from __future__ import annotations

import json
import re
import urllib.request
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str: ...


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
        return str(data["choices"][0]["message"]["content"] or "")


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
