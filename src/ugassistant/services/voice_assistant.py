from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ugassistant.services.audio import AudioDeviceService, AudioStatus
from ugassistant.services.conversation import ConversationService
from ugassistant.services.recognition import VoiceRecognitionService
from ugassistant.services.audio import NoSpeechDetectedError
from ugassistant.services.speech import SpeechService


logger = logging.getLogger("ugassistant.voice_assistant")


@dataclass(frozen=True)
class VoiceAssistantStatus:
    enabled: bool = True
    busy: bool = False
    phase: str = "waiting_for_wake_word"
    detail: str = "microphone_disabled"
    wake_transcript: str = ""
    question: str = ""
    answer: str = ""
    language: str | None = None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


VoiceAssistantStatusCallback = Callable[[VoiceAssistantStatus], Awaitable[None]]


class VoiceAssistantService:
    def __init__(
        self,
        audio_service: AudioDeviceService,
        recognition_service: VoiceRecognitionService,
        speech_service: SpeechService,
        conversation_service: ConversationService,
        *,
        spanish_wake_words: tuple[str, ...] = ("hola",),
        french_wake_words: tuple[str, ...] = ("salut",),
        spanish_greeting: str = "¿Qué desea?",
        french_greeting: str = "Que puis-je faire pour vous ?",
        follow_up_wait_seconds: float = 0.0,
        on_status: VoiceAssistantStatusCallback | None = None,
    ) -> None:
        self._audio_service = audio_service
        self._recognition_service = recognition_service
        self._speech_service = speech_service
        self._conversation_service = conversation_service
        self._wake_words = {
            "es": self._normalized_words(spanish_wake_words),
            "fr": self._normalized_words(french_wake_words),
        }
        self._greetings = {"es": spanish_greeting, "fr": french_greeting}
        self._follow_up_wait_seconds = max(follow_up_wait_seconds, 0.0)
        self._on_status = on_status
        self._task: asyncio.Task[None] | None = None
        self._end_requested = False
        self._status = VoiceAssistantStatus()

    @property
    def status(self) -> VoiceAssistantStatus:
        return self._status

    @property
    def wake_words(self) -> dict[str, tuple[str, ...]]:
        return {
            language: tuple(sorted(words))
            for language, words in self._wake_words.items()
        }

    def configure_wake_words(self, spanish: str, french: str) -> None:
        self._wake_words = {
            "es": self._normalized_words((spanish,)),
            "fr": self._normalized_words((french,)),
        }

    def request_end_session(self) -> bool:
        if not self._status.busy:
            return False
        self._end_requested = True
        return True

    def observe_audio(self, status: AudioStatus) -> None:
        if not status.monitoring:
            if not self._status.busy:
                self._set_status_now(phase="waiting_for_wake_word", detail="microphone_disabled")
            return
        if not self._status.busy and self._status.detail == "microphone_disabled":
            self._set_status_now(
                phase="waiting_for_wake_word",
                detail="monitoring_wake_word",
            )
        if (
            not status.sound_detected
            or not self._audio_service.has_monitor_audio_buffer
            or self._status.busy
            or self._speech_service.status.busy
        ):
            return
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._handle_wake_word())
            self._task.add_done_callback(self._log_task_error)

    async def shutdown(self) -> None:
        self._recognition_service.cancel()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _handle_wake_word(self) -> None:
        try:
            self._end_requested = False
            await self._set_status( busy=True, phase="detecting_wake_word", detail="transcribing_wake_word")
            wake = await self._recognition_service.recognize_once(
                initial_chunks=self._audio_service.take_monitor_audio_buffer()
            )
            wake_language = self._wake_language(wake.text)
            if wake_language is None:
                await self._set_status(
                    busy=False,
                    phase="waiting_for_wake_word",
                    detail="wake_word_not_detected",
                    wake_transcript=wake.text,
                    language=wake.language,
                )
                return
            if not self._audio_service.status.output_enabled:
                raise RuntimeError("Audio output is disabled")
            await self._set_status(
                busy=True,
                phase="greeting",
                detail="wake_word_detected",
                wake_transcript=wake.text,
                language=wake.language,
            )
            await self._speech_service.select_language(
                "fr_FR" if wake_language == "fr" else "es_ES"
            )
            await self._speech_service.speak(self._greetings[wake_language])
            await self._set_status(
                busy=True,
                phase="listening_for_question",
                detail="waiting_for_question",
                wake_transcript=wake.text,
                language=wake.language,
            )
            question = await self._recognition_service.recognize_once()
            await self._set_status(
                busy=True,
                phase="thinking",
                detail="asking_ollama",
                wake_transcript=wake.text,
                question=question.text,
                language=question.language,
            )
            answer = await self._conversation_service.answer(
                question.text,
                question.language,
            )
            await self._speech_service.select_language(
                "fr_FR" if question.language.casefold() == "fr" else "es_ES"
            )
            await self._set_status(
                busy=True,
                phase="speaking",
                detail="answer_ready",
                wake_transcript=wake.text,
                question=question.text,
                answer=answer,
                language=question.language,
            )
            await self._complete_answer_loop(
                wake.text,
                question.text,
                answer,
                question.language,
            )
        except asyncio.CancelledError:
            await self._set_status(busy=False, phase="waiting_for_wake_word", detail="cancelled")
            raise
        except Exception as exc:
            logger.exception("voice_assistant_turn_failed")
            await self._set_status(busy=False, phase="error", detail=str(exc))

    def _wake_language(self, transcript: str) -> str | None:
        normalized = self._normalize(transcript)
        for language, wake_words in self._wake_words.items():
            if any(
                re.search(rf"\b{re.escape(wake_word)}\b", normalized)
                for wake_word in wake_words
            ):
                return language
        return None

    async def _complete_answer_loop(
        self,
        wake_transcript: str,
        question: str,
        answer: str,
        language: str,
    ) -> None:
        await self._speech_service.speak(answer)
        locale = "fr_FR" if language.casefold() == "fr" else "es_ES"
        farewell = "De acuerdo, hasta luego." if locale == "es_ES" else "D'accord, a bientot."
        if self._end_requested:
            await self._speech_service.speak(farewell)
            await self._set_status(
                busy=False,
                phase="waiting_for_wake_word",
                detail="ended_by_gesture",
                wake_transcript=wake_transcript,
                question=question,
                answer=answer,
                language=language,
            )
            return
        if self._follow_up_wait_seconds <= 0:
            await self._set_status(
                busy=False,
                phase="waiting_for_wake_word",
                detail="completed",
                wake_transcript=wake_transcript,
                question=question,
                answer=answer,
                language=language,
            )
            return

        prompt = (
            "Souhaitez-vous une precision ou avez-vous une autre question ?"
            if locale == "fr_FR"
            else "Quieres que aclare algo o tienes otra pregunta?"
        )
        while True:
            await self._speech_service.speak(prompt)
            await self._set_status(
                busy=True,
                phase="listening_for_follow_up",
                detail="waiting_for_follow_up",
                wake_transcript=wake_transcript,
                question=question,
                answer=answer,
                language=language,
            )
            try:
                follow_up = await self._recognition_service.recognize_once(
                    wait_for_speech_seconds=self._follow_up_wait_seconds
                )
            except NoSpeechDetectedError:
                await self._speech_service.speak(farewell)
                break
            if self._end_requested:
                await self._speech_service.speak(farewell)
                break
            if self._is_session_end(follow_up.text):
                await self._speech_service.speak(farewell)
                break
            question = follow_up.text
            language = follow_up.language
            locale = "fr_FR" if language.casefold() == "fr" else "es_ES"
            prompt = (
                "Souhaitez-vous une precision ou avez-vous une autre question ?"
                if locale == "fr_FR"
                else "Quieres que aclare algo o tienes otra pregunta?"
            )
            farewell = "De acuerdo, hasta luego." if locale == "es_ES" else "D'accord, a bientot."
            await self._set_status(
                busy=True,
                phase="thinking",
                detail="asking_ollama_follow_up",
                wake_transcript=wake_transcript,
                question=question,
                answer=answer,
                language=language,
            )
            answer = await self._conversation_service.answer(question, language)
            await self._speech_service.select_language(locale)
            await self._set_status(
                busy=True,
                phase="speaking",
                detail="follow_up_ready",
                wake_transcript=wake_transcript,
                question=question,
                answer=answer,
                language=language,
            )
            await self._speech_service.speak(answer)

        await self._set_status(
            busy=False,
            phase="waiting_for_wake_word",
            detail="completed",
            wake_transcript=wake_transcript,
            question=question,
            answer=answer,
            language=language,
        )

    def _is_session_end(self, transcript: str) -> bool:
        normalized = self._normalize(transcript)
        endings = ("no", "nada mas", "no gracias", "adios", "hasta luego", "fin")
        return any(ending in normalized for ending in endings)

    @staticmethod
    def _normalized_words(words: tuple[str, ...]) -> frozenset[str]:
        return frozenset(
            VoiceAssistantService._normalize(word)
            for word in words
            if VoiceAssistantService._normalize(word)
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return "".join(
            character
            for character in unicodedata.normalize("NFD", value.casefold())
            if unicodedata.category(character) != "Mn"
        )

    def _set_status_now(self, **values: object) -> None:
        self._status = VoiceAssistantStatus(**{**self._status.__dict__, **values})  # type: ignore[arg-type]
        if self._on_status is not None:
            task = asyncio.create_task(self._on_status(self._status))
            task.add_done_callback(self._log_task_error)

    async def _set_status(self, **values: object) -> None:
        self._status = VoiceAssistantStatus(**{**self._status.__dict__, **values})  # type: ignore[arg-type]
        if self._on_status is not None:
            await self._on_status(self._status)

    @staticmethod
    def _log_task_error(task: asyncio.Task[object]) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("voice_assistant_background_task_failed", exc_info=task.exception())
