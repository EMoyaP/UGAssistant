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
        short_context_tokens: int = 2048,
        complete_context_tokens: int = 4096,
        max_tokens: int = 384,
        complete_max_tokens: int = 2048,
        temperature: float = 0.2,
        repeat_penalty: float = 1.08,
        on_status: ConversationStatusCallback | None = None,
    ) -> None:
        self._adapter = adapter
        self._inference_lock = inference_lock
        self._max_history_messages = max(0, max_history_turns) * 2
        self._short_context_tokens = min(max(short_context_tokens, 1024), 4096)
        self._complete_context_tokens = min(
            max(complete_context_tokens, self._short_context_tokens),
            8192,
        )
        self._max_tokens = max(16, max_tokens)
        self._complete_max_tokens = max(self._max_tokens, complete_max_tokens)
        self._temperature = min(max(temperature, 0.0), 2.0)
        self._repeat_penalty = min(max(repeat_penalty, 1.0), 1.5)
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
            messages = self._build_messages(
                normalized_question,
                normalized_language,
                response_detail,
            )
            try:
                async with self._inference_lock:
                    raw_answer = await self._adapter.chat(
                        tuple(messages),
                        max_tokens=self._max_tokens_for(response_detail),
                        temperature=self._temperature,
                        repeat_penalty=self._repeat_penalty,
                        # Gemma 3 4B does not expose Ollama's thinking capability.
                        think=False,
                        context_tokens=self._context_tokens_for(response_detail),
                    )
                answer = self._clean_response(raw_answer)
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
                    "francais avec une reponse complete, pratique et proportionnee a la "
                    "demande. Utilise seulement des informations que tu peux soutenir par "
                    "des connaissances generales fiables. N'invente pas de faits, ingredients, "
                    "etapes, dates, chiffres, noms, citations ou sources. Ne presente rien "
                    "comme authentique, traditionnel, officiel, correct, sur ou garanti si tu "
                    "ne peux pas le soutenir. Omet les details incertains et dis clairement "
                    "quand une verification est necessaire. Ne donne pas ton raisonnement "
                    "interne. Utilise un texte naturel pour la voix, sans markdown, asterisques, "
                    "titres, liens ou listes a puces. Termine naturellement sans promettre une "
                    "exactitude factuelle que tu ne peux pas verifier."
                )
            return (
                "Eres UGAssistant, un asistente local. Responde solo en espanol con una "
                "respuesta completa, practica y proporcionada a la pregunta. Usa solo "
                "informacion que puedas sostener con conocimiento general fiable. No inventes "
                "datos, ingredientes, pasos, fechas, cifras, nombres, citas ni fuentes. No "
                "afirmes que algo es autentico, tradicional, oficial, correcto, seguro o "
                "garantizado si no puedes sostenerlo. Omite los detalles inciertos y explica "
                "con claridad cuando sea necesaria una verificacion. No muestres razonamiento "
                "interno. Usa texto natural apto para voz, sin markdown, asteriscos, "
                "almohadillas, enlaces ni listas con guiones. Termina de forma natural sin "
                "prometer una exactitud factual que no puedes verificar."
            )
        if language == "fr":
            return (
                "Tu es UGAssistant, un assistant local. Reponds uniquement en "
                "francais, de maniere utile et breve, en trois phrases maximum, sans "
                "markdown. N'invente pas de faits et omets les details incertains au lieu "
                "de les completer. Ne presente rien comme authentique, traditionnel, "
                "correct ou garanti sans pouvoir le soutenir. Ne montre pas ton raisonnement interne."
            )
        return (
            "Eres UGAssistant, un asistente local. Responde solo en espanol, "
            "de forma util y breve, en un maximo de tres frases y sin markdown. "
            "No inventes datos: omite los detalles inciertos en vez de rellenarlos. "
            "No afirmes que algo es autentico, tradicional, correcto o garantizado si "
            "no puedes sostenerlo, y no muestres tu razonamiento interno."
        )

    def _max_tokens_for(self, response_detail: ResponseDetail) -> int:
        return (
            self._complete_max_tokens
            if response_detail == "complete"
            else self._max_tokens
        )

    def _context_tokens_for(self, response_detail: ResponseDetail) -> int:
        return (
            self._complete_context_tokens
            if response_detail == "complete"
            else self._short_context_tokens
        )

    def _build_messages(
        self,
        question: str,
        language: str,
        response_detail: ResponseDetail,
    ) -> tuple[LLMMessage, ...]:
        system = LLMMessage("system", self._system_prompt(language, response_detail))
        current_question = LLMMessage("user", question)
        context_tokens = self._context_tokens_for(response_detail)
        output_reserve = self._max_tokens_for(response_detail)
        # Ollama's context window includes the generated response. Keep a small
        # margin for tokenisation differences between Spanish and French.
        input_budget = max(256, context_tokens - output_reserve - 128)
        selected_history: list[LLMMessage] = []
        used_tokens = self._estimated_tokens(system.content) + self._estimated_tokens(
            current_question.content
        )
        for message in reversed(self._history[-self._max_history_messages :]):
            message_tokens = self._estimated_tokens(message.content)
            if used_tokens + message_tokens > input_budget:
                break
            selected_history.append(message)
            used_tokens += message_tokens
        selected_history.reverse()
        return (system, *selected_history, current_question)

    @staticmethod
    def _estimated_tokens(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def _clean_response(self, response: str) -> str:
        return " ".join(self._plain_text(response).split())

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
