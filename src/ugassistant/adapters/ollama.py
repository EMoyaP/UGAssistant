from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

from ugassistant.domain.ports import LLMAdapter, LLMEngineStatus, LLMMessage


class OllamaAdapter(LLMAdapter):
    """Small adapter for the local Ollama HTTP API; it has no cloud fallback."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds

    async def status(self) -> LLMEngineStatus:
        try:
            payload = await asyncio.to_thread(self._request, "/api/tags", None)
        except RuntimeError as exc:
            return LLMEngineStatus(False, False, str(exc))
        models = payload.get("models", [])
        names = {
            str(item.get("name", ""))
            for item in models
            if isinstance(item, dict)
        }
        model_available = self._model in names
        detail = "ready" if model_available else f"model_not_installed:{self._model}"
        return LLMEngineStatus(True, model_available, detail)

    async def chat(
        self,
        messages: tuple[LLMMessage, ...],
        *,
        max_tokens: int,
        temperature: float,
        repeat_penalty: float,
        think: bool,
        context_tokens: int,
    ) -> str:
        payload = await asyncio.to_thread(
            self._request,
            "/api/chat",
            {
                "model": self._model,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
                "stream": False,
                "think": think,
                "keep_alive": "5m",
                "options": {
                    "num_predict": max_tokens,
                    "num_ctx": context_tokens,
                    "temperature": temperature,
                    "repeat_penalty": repeat_penalty,
                },
            },
        )
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama returned an invalid chat response")
        return str(message["content"])

    def _request(self, path: str, payload: dict[str, object] | None) -> dict[str, object]:
        request = urllib.request.Request(
            f"{self._base_url}{path}",
            data=(json.dumps(payload).encode("utf-8") if payload is not None else None),
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ollama_unreachable:{exc.reason}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("ollama_invalid_response") from exc
        if not isinstance(data, dict):
            raise RuntimeError("ollama_invalid_response")
        return data
