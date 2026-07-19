from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ugassistant.domain.ports import LLMAdapter, LLMMessage


@dataclass(frozen=True)
class ConversationStatus:
    available: bool = False
    model_available: bool = False
    busy: bool = False
    phase: str = "not_scanned"
    detail: str = "not_scanned"
    question: str = ""
    answer: str = ""
    language: str | None = None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


ConversationStatusCallback = Callable[[ConversationStatus], Awaitable[None]]


class ConversationService:
    def __init__(
        self,
        adapter: LLMAdapter,
        *,
        inference_lock: asyncio.Lock,
        max_history_turns: int = 3,
        max_response_characters: int = 320,
        max_tokens: int = 160,
        temperature: float = 0.4,
        on_status: ConversationStatusCallback | None = None,
    ) -> None:
        self._adapter = adapter
        self._inference_lock = inference_lock
        self._max_history_messages = max(0, max_history_turns) * 2
        self._max_response_characters = max(32, max_response_characters)
        self._max_tokens = max(16, max_tokens)
        self._temperature = min(max(temperature, 0.0), 2.0)
        self._on_status = on_status
        self._history: list[LLMMessage] = []
        self._lock = asyncio.Lock()
        self._status = ConversationStatus()

    @property
    def status(self) -> ConversationStatus:
        return self._status

    async def refresh(self) -> ConversationStatus:
        engine = await self._adapter.status()
        if not self._lock.locked():
            self._status = ConversationStatus(
                available=engine.available,
                model_available=engine.model_available,
                phase="ready" if engine.available and engine.model_available else "unavailable",
                detail=engine.detail,
            )
            await self._publish()
        return self._status

    async def answer(self, question: str, language: str) -> str:
        normalized_question = " ".join(question.split())
        normalized_language = language.casefold()
        if normalized_language not in {"es", "fr"}:
            raise ValueError(f"Unsupported conversation language: {language}")
        if not normalized_question:
            raise ValueError("Question cannot be empty")
        if self._lock.locked():
            raise RuntimeError("Conversation is already active")

        async with self._lock:
            engine = await self._adapter.status()
            if not engine.available or not engine.model_available:
                raise RuntimeError(f"Ollama is unavailable: {engine.detail}")
            await self._set_status(
                available=True,
                model_available=True,
                busy=True,
                phase="thinking",
                detail="ollama_inference",
                question=normalized_question,
                language=normalized_language,
            )
            messages = (
                LLMMessage("system", self._system_prompt(normalized_language)),
                *self._history[-self._max_history_messages :],
                LLMMessage("user", normalized_question),
            )
            try:
                async with self._inference_lock:
                    raw_answer = await self._adapter.chat(
                        tuple(messages),
                        max_tokens=self._max_tokens,
                        temperature=self._temperature,
                    )
                answer = self._short_response(raw_answer)
                if not answer:
                    raise RuntimeError("Ollama returned an empty response")
                self._history.extend(
                    (
                        LLMMessage("user", normalized_question),
                        LLMMessage("assistant", answer),
                    )
                )
                if self._max_history_messages:
                    self._history = self._history[-self._max_history_messages :]
                else:
                    self._history = []
                await self._set_status(
                    available=True,
                    model_available=True,
                    busy=False,
                    phase="completed",
                    detail="response_ready",
                    question=normalized_question,
                    answer=answer,
                    language=normalized_language,
                )
                return answer
            except Exception as exc:
                await self._set_status(
                    available=engine.available,
                    model_available=engine.model_available,
                    busy=False,
                    phase="error",
                    detail=str(exc),
                    question=normalized_question,
                    language=normalized_language,
                )
                raise

    @staticmethod
    def _system_prompt(language: str) -> str:
        if language == "fr":
            return (
                "Tu es UGAssistant, un assistant local. Reponds uniquement en "
                "francais, de maniere utile et breve, en deux phrases maximum."
            )
        return (
            "Eres UGAssistant, un asistente local. Responde solo en espanol, "
            "de forma util y breve, en un maximo de dos frases."
        )

    def _short_response(self, response: str) -> str:
        compact = " ".join(response.split())
        if len(compact) <= self._max_response_characters:
            return compact
        shortened = compact[: self._max_response_characters].rsplit(" ", 1)[0]
        return shortened.rstrip(".,;: ") + "."

    async def _set_status(self, **values: object) -> None:
        self._status = ConversationStatus(**values)  # type: ignore[arg-type]
        await self._publish()

    async def _publish(self) -> None:
        if self._on_status is not None:
            await self._on_status(self._status)
