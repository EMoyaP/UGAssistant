from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import re
from typing import Literal

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
ResponseDetail = Literal["short", "complete"]


class ConversationService:
    def __init__(
        self,
        adapter: LLMAdapter,
        *,
        inference_lock: asyncio.Lock,
        max_history_turns: int = 3,
        max_response_characters: int = 320,
        max_tokens: int = 160,
        complete_max_response_characters: int = 1200,
        complete_max_tokens: int = 384,
        temperature: float = 0.4,
        on_status: ConversationStatusCallback | None = None,
    ) -> None:
        self._adapter = adapter
        self._inference_lock = inference_lock
        self._max_history_messages = max(0, max_history_turns) * 2
        self._max_response_characters = max(32, max_response_characters)
        self._max_tokens = max(16, max_tokens)
        self._complete_max_response_characters = max(
            self._max_response_characters,
            complete_max_response_characters,
        )
        self._complete_max_tokens = max(self._max_tokens, complete_max_tokens)
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

    async def answer(
        self,
        question: str,
        language: str,
        response_detail: ResponseDetail = "short",
    ) -> str:
        normalized_question = " ".join(question.split())
        normalized_language = language.casefold()
        if normalized_language not in {"es", "fr"}:
            raise ValueError(f"Unsupported conversation language: {language}")
        if not normalized_question:
            raise ValueError("Question cannot be empty")
        if response_detail not in {"short", "complete"}:
            raise ValueError(f"Unsupported response detail: {response_detail}")
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
                LLMMessage(
                    "system",
                    self._system_prompt(normalized_language, response_detail),
                ),
                *self._history[-self._max_history_messages :],
                LLMMessage("user", normalized_question),
            )
            try:
                async with self._inference_lock:
                    raw_answer = await self._adapter.chat(
                        tuple(messages),
                        max_tokens=self._max_tokens_for(response_detail),
                        temperature=self._temperature,
                    )
                answer = self._compact_response(raw_answer, response_detail)
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
    def _system_prompt(language: str, response_detail: ResponseDetail) -> str:
        if response_detail == "complete":
            if language == "fr":
                return (
                    "Tu es UGAssistant, un assistant local. Reponds uniquement en "
                    "francais avec une reponse complete, exacte et pratique. Inclus le "
                    "contexte et les etapes necessaires pour resoudre la demande. "
                    "Utilise des phrases claires sans markdown, asterisques, titres ou "
                    "listes a puces et evite les details "
                    "inventes."
                )
            return (
                "Eres UGAssistant, un asistente local. Responde solo en espanol con "
                "una respuesta completa, exacta y practica. Incluye el contexto y los "
                "pasos necesarios para resolver la duda. Usa frases claras sin markdown, "
                "asteriscos, almohadillas o listas con guiones y evita inventar detalles."
            )
        if language == "fr":
            return (
                "Tu es UGAssistant, un assistant local. Reponds uniquement en "
                "francais, de maniere utile et breve, en deux phrases maximum, sans markdown."
            )
        return (
            "Eres UGAssistant, un asistente local. Responde solo en espanol, "
            "de forma util y breve, en un maximo de dos frases y sin markdown."
        )

    def _max_tokens_for(self, response_detail: ResponseDetail) -> int:
        return (
            self._complete_max_tokens
            if response_detail == "complete"
            else self._max_tokens
        )

    def _compact_response(
        self,
        response: str,
        response_detail: ResponseDetail,
    ) -> str:
        compact = " ".join(self._plain_text(response).split())
        max_characters = (
            self._complete_max_response_characters
            if response_detail == "complete"
            else self._max_response_characters
        )
        if len(compact) <= max_characters:
            return compact
        shortened = compact[:max_characters].rsplit(" ", 1)[0]
        return shortened.rstrip(".,;: ") + "."

    @staticmethod
    def _plain_text(response: str) -> str:
        text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", response)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
        text = re.sub(r"(?m)^\s*[-+*]\s+", "", text)
        text = re.sub(r"(?m)^\s*>\s?", "", text)
        text = text.replace("```", "").replace("`", "")
        text = text.replace("**", "").replace("__", "").replace("~~", "")
        text = re.sub(r"(?<!\w)[*_~]+|[*_~]+(?!\w)", "", text)
        return text.replace("#", "")

    async def _set_status(self, **values: object) -> None:
        self._status = ConversationStatus(**values)  # type: ignore[arg-type]
        await self._publish()

    async def _publish(self) -> None:
        if self._on_status is not None:
            await self._on_status(self._status)
