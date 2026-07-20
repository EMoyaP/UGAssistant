from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ugassistant.domain.timer_alarm import build_timer_alarm_wav
from ugassistant.domain.timers import TimerSnapshot
from ugassistant.services.audio import AudioDeviceService, AudioStatus
from ugassistant.services.conversation import ConversationService
from ugassistant.services.recognition import VoiceRecognitionService
from ugassistant.services.audio import NoSpeechDetectedError
from ugassistant.services.speech import SpeechService
from ugassistant.services.timers import TimerService


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
    response_detail: str = "short"
    timers: tuple[TimerSnapshot, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            **self.__dict__,
            "timers": [timer.to_dict() for timer in self.timers],
        }


VoiceAssistantStatusCallback = Callable[[VoiceAssistantStatus], Awaitable[None]]


class VoiceAssistantService:
    def __init__(
        self,
        audio_service: AudioDeviceService,
        recognition_service: VoiceRecognitionService,
        speech_service: SpeechService,
        conversation_service: ConversationService,
        *,
        timer_service: TimerService | None = None,
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
        self._timer_service = timer_service
        self._wake_words = {
            "es": self._normalized_words(spanish_wake_words),
            "fr": self._normalized_words(french_wake_words),
        }
        self._greetings = {"es": spanish_greeting, "fr": french_greeting}
        self._follow_up_wait_seconds = max(follow_up_wait_seconds, 0.0)
        self._on_status = on_status
        self._task: asyncio.Task[None] | None = None
        self._end_requested = False
        self._status = VoiceAssistantStatus(
            timers=timer_service.timers if timer_service is not None else (),
        )

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

    async def update_timers(self, timers: tuple[TimerSnapshot, ...]) -> None:
        await self._set_status(timers=timers)

    async def notify_timer_expired(self, timer: TimerSnapshot) -> None:
        """Play a local alarm, then announce the completed timer."""
        locale = "fr_FR" if timer.language.casefold() == "fr" else "es_ES"
        was_monitoring = self._audio_service.status.monitoring
        monitoring_paused = False
        try:
            if self._speech_service.status.busy:
                await self._speech_service.interrupt()
            if was_monitoring and not self._audio_service.status.recording:
                await self._audio_service.disable_monitoring()
                monitoring_paused = True
            await self._audio_service.play_wav(build_timer_alarm_wav())
            await self._speech_service.select_language(locale)
            message = (
                f"Temporizador de {self._format_timer_duration(timer.duration_seconds, locale)} finalizado."
                if locale == "es_ES"
                else f"Minuteur de {self._format_timer_duration(timer.duration_seconds, locale)} termine."
            )
            await self._speech_service.speak(message)
        except Exception:
            logger.exception("timer_expiration_notification_failed label=%s", timer.label)
        finally:
            if monitoring_paused and not self._audio_service.status.monitoring:
                try:
                    await self._audio_service.enable_monitoring()
                except Exception:
                    logger.exception("timer_monitor_resume_failed")

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
                initial_chunks=self._audio_service.take_monitor_audio_buffer(),
                use_selected_language=False,
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
            inline_question = self._wake_remainder(wake.text, wake_language)
            if await self._handle_timer_request(
                wake.text,
                inline_question,
                wake_language,
            ):
                return
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
            question = await self._recognition_service.recognize_once(
                language_hint=wake_language
            )
            if await self._handle_timer_request(wake.text, question.text, question.language):
                return
            await self._set_status(
                busy=True,
                phase="asking_response_detail",
                detail="waiting_for_response_detail",
                wake_transcript=wake.text,
                question=question.text,
                language=question.language,
            )
            response_detail = await self._ask_response_detail(question.language)
            if self._end_requested:
                await self._set_status(
                    busy=False,
                    phase="waiting_for_wake_word",
                    detail="ended_by_gesture",
                    wake_transcript=wake.text,
                    question=question.text,
                    language=question.language,
                    response_detail=response_detail,
                )
                return
            if response_detail == "complete":
                await self._announce_complete_response(question.language)
            await self._set_status(
                busy=True,
                phase="thinking",
                detail="asking_ollama",
                wake_transcript=wake.text,
                question=question.text,
                language=question.language,
                response_detail=response_detail,
            )
            answer = await self._conversation_service.answer(
                question.text,
                question.language,
                response_detail,
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
                response_detail=response_detail,
            )
            await self._complete_answer_loop(
                wake.text,
                question.text,
                answer,
                question.language,
                response_detail,
            )
        except asyncio.CancelledError:
            await self._finish_interruption(
                "interrupted" if self._end_requested else "cancelled"
            )
            if not self._end_requested:
                raise
        except NoSpeechDetectedError:
            await self._finish_interruption("no_speech_detected")
        except Exception as exc:
            if "did not recognize any text" in str(exc).casefold():
                await self._finish_interruption("no_speech_recognized")
                return
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

    async def _finish_interruption(self, detail: str) -> None:
        await self._set_status(
            busy=False,
            phase="waiting_for_wake_word",
            detail=detail,
            wake_transcript="",
            question="",
            answer="",
            language=None,
            response_detail="short",
        )

    def _wake_remainder(self, transcript: str, language: str) -> str:
        for wake_word in self._wake_words.get(language, frozenset()):
            match = re.search(rf"\b{re.escape(wake_word)}\b", transcript, re.IGNORECASE)
            if match is not None:
                return transcript[match.end():].lstrip(" ,.:;!?-\u2026")
        return ""

    async def _handle_timer_request(
        self,
        wake_transcript: str,
        question: str,
        language: str,
    ) -> bool:
        command = self._timer_command(question)
        if command is None or self._timer_service is None:
            return False
        action, duration_seconds = command
        locale = "fr_FR" if language.casefold() == "fr" else "es_ES"
        if action == "create":
            if duration_seconds is None:
                duration_seconds = await self._ask_timer_duration(
                    wake_transcript,
                    language,
                    action="create",
                )
            if duration_seconds is None:
                return True
            timer = await self._timer_service.create(
                duration_seconds,
                language=language,
            )
            description = self._format_timer_duration(duration_seconds, locale)
            response = (
                f"Activando temporizador de {description} que empieza ya."
                if timer.label == 1
                else f"Activando temporizador {timer.label} de {description} que empieza ya."
            )
            if locale == "fr_FR":
                response = (
                    f"Minuteur de {description} active, il commence maintenant."
                    if timer.label == 1
                    else f"Minuteur {timer.label} de {description} active, il commence maintenant."
                )
            await self._speech_service.select_language(locale)
            await self._speech_service.speak(response)
            await self._set_timer_status(
                detail="timer_started",
                wake_transcript=wake_transcript,
                language=language,
            )
            return True

        timers = self._timer_service.timers
        if not timers:
            response = (
                "No hay temporizadores activos."
                if locale == "es_ES"
                else "Il n'y a aucun minuteur actif."
            )
            await self._speech_service.select_language(locale)
            await self._speech_service.speak(response)
            await self._set_timer_status(
                detail="timer_not_found",
                wake_transcript=wake_transcript,
                language=language,
            )
            return True

        selected_label = await self._select_timer_label(
            action,
            timers,
            wake_transcript,
            language,
        )
        if selected_label is None:
            return True
        if action == "cancel":
            await self._timer_service.cancel(selected_label)
            response = (
                f"He cancelado el temporizador {selected_label}."
                if locale == "es_ES"
                else f"J'ai annule le minuteur {selected_label}."
            )
            await self._speech_service.select_language(locale)
            await self._speech_service.speak(response)
            await self._set_timer_status(
                detail="timer_cancelled",
                wake_transcript=wake_transcript,
                language=language,
            )
            return True

        duration_seconds = await self._ask_timer_duration(
            wake_transcript,
            language,
            action="modify",
        )
        if duration_seconds is None:
            return True
        await self._timer_service.modify(selected_label, duration_seconds)
        description = self._format_timer_duration(duration_seconds, locale)
        response = (
            f"He modificado el temporizador {selected_label} a {description}."
            if locale == "es_ES"
            else f"J'ai modifie le minuteur {selected_label} a {description}."
        )
        await self._speech_service.select_language(locale)
        await self._speech_service.speak(response)
        await self._set_timer_status(
            detail="timer_modified",
            wake_transcript=wake_transcript,
            language=language,
        )
        return True

    async def _select_timer_label(
        self,
        action: str,
        timers: tuple[TimerSnapshot, ...],
        wake_transcript: str,
        language: str,
    ) -> int | None:
        if len(timers) == 1:
            return timers[0].label
        locale = "fr_FR" if language.casefold() == "fr" else "es_ES"
        verb = "modificar" if action == "modify" else "cancelar"
        prompt = (
            f"Que temporizador quieres {verb}? Indica su numero."
            if locale == "es_ES"
            else "Quel minuteur voulez-vous modifier ? Indiquez son numero."
        )
        await self._speech_service.select_language(locale)
        await self._speech_service.speak(prompt)
        await self._set_status(
            busy=True,
            phase="listening_for_timer_selection",
            detail="waiting_for_timer_selection",
            wake_transcript=wake_transcript,
            language=language,
        )
        try:
            selection = await self._recognition_service.recognize_once(
                language_hint=language
            )
        except NoSpeechDetectedError:
            await self._set_timer_status(
                detail="timer_selection_timeout",
                wake_transcript=wake_transcript,
                language=language,
            )
            return None
        label = self._timer_label(selection.text)
        if label is not None and any(timer.label == label for timer in timers):
            return label
        response = (
            "No he identificado ese temporizador."
            if locale == "es_ES"
            else "Je n'ai pas reconnu ce minuteur."
        )
        await self._speech_service.select_language(locale)
        await self._speech_service.speak(response)
        await self._set_timer_status(
            detail="timer_selection_invalid",
            wake_transcript=wake_transcript,
            language=language,
        )
        return None

    async def _ask_timer_duration(
        self,
        wake_transcript: str,
        language: str,
        *,
        action: str,
    ) -> int | None:
        locale = "fr_FR" if language.casefold() == "fr" else "es_ES"
        prompt = (
            "De cuanto tiempo?" if action == "create" else "Cual es el nuevo tiempo?"
        )
        if locale == "fr_FR":
            prompt = "Pour combien de temps ?" if action == "create" else "Quelle est la nouvelle duree ?"
        await self._speech_service.select_language(locale)
        await self._speech_service.speak(prompt)
        await self._set_status(
            busy=True,
            phase="listening_for_timer_duration",
            detail="waiting_for_timer_duration",
            wake_transcript=wake_transcript,
            language=language,
        )
        try:
            duration = await self._recognition_service.recognize_once(
                language_hint=language
            )
        except NoSpeechDetectedError:
            await self._set_timer_status(
                detail="timer_duration_timeout",
                wake_transcript=wake_transcript,
                language=language,
            )
            return None
        duration_seconds = self._parse_timer_duration(duration.text)
        if duration_seconds is not None:
            return duration_seconds
        response = (
            "No he entendido la duracion del temporizador."
            if locale == "es_ES"
            else "Je n'ai pas compris la duree du minuteur."
        )
        await self._speech_service.select_language(locale)
        await self._speech_service.speak(response)
        await self._set_timer_status(
            detail="timer_duration_invalid",
            wake_transcript=wake_transcript,
            language=language,
        )
        return None

    async def _set_timer_status(
        self,
        *,
        detail: str,
        wake_transcript: str,
        language: str,
    ) -> None:
        await self._set_status(
            busy=False,
            phase="waiting_for_wake_word",
            detail=detail,
            wake_transcript=wake_transcript,
            question="",
            answer="",
            language=language,
            response_detail="short",
        )

    @classmethod
    def _timer_command(cls, transcript: str) -> tuple[str, int | None] | None:
        normalized = cls._normalize(transcript)
        if not any(word in normalized for word in ("temporizador", "minuteur", "timer")):
            return None
        if any(word in normalized for word in ("cancel", "anula", "anular", "annule", "arrete")):
            return ("cancel", None)
        if any(word in normalized for word in ("modifica", "modificar", "cambia", "ajusta", "modifier")):
            return ("modify", None)
        return ("create", cls._parse_timer_duration(transcript))

    @classmethod
    def _parse_timer_duration(cls, transcript: str) -> int | None:
        normalized = cls._normalize(transcript)
        number_words = {
            "un": 1,
            "una": 1,
            "uno": 1,
            "dos": 2,
            "tres": 3,
            "cuatro": 4,
            "cinco": 5,
            "seis": 6,
            "siete": 7,
            "ocho": 8,
            "nueve": 9,
            "diez": 10,
            "once": 11,
            "doce": 12,
            "trece": 13,
            "catorce": 14,
            "quince": 15,
            "seize": 16,
            "dix": 10,
            "onze": 11,
            "douze": 12,
            "treize": 13,
            "quatorze": 14,
            "quinze": 15,
        }
        amount_pattern = r"(\d+|" + "|".join(number_words) + r")"
        matches = re.findall(
            amount_pattern
            + r"\s*(segundo(?:s)?|seconde(?:s)?|minuto(?:s)?|minute(?:s)?|hora(?:s)?|heure(?:s)?)\b",
            normalized,
        )
        if not matches:
            return None
        duration_seconds = 0
        for raw_amount, raw_unit in matches:
            amount = int(raw_amount) if raw_amount.isdigit() else number_words[raw_amount]
            if raw_unit.startswith(("hora", "heure")):
                duration_seconds += amount * 3600
            elif raw_unit.startswith(("minuto", "minute")):
                duration_seconds += amount * 60
            else:
                duration_seconds += amount
        return duration_seconds or None

    @classmethod
    def _timer_label(cls, transcript: str) -> int | None:
        normalized = cls._normalize(transcript)
        numeric = re.search(r"\b(\d+)\b", normalized)
        if numeric is not None:
            return int(numeric.group(1))
        numbers = {
            "uno": 1,
            "un": 1,
            "dos": 2,
            "tres": 3,
            "cuatro": 4,
            "cinco": 5,
            "seis": 6,
            "siete": 7,
            "ocho": 8,
            "nueve": 9,
            "diez": 10,
            "un": 1,
            "deux": 2,
            "trois": 3,
        }
        return next((value for word, value in numbers.items() if re.search(rf"\b{word}\b", normalized)), None)

    @staticmethod
    def _format_timer_duration(duration_seconds: int, locale: str) -> str:
        hours, remainder = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        parts: list[str] = []
        if locale == "fr_FR":
            if hours:
                parts.append(f"{hours} heure" + ("s" if hours != 1 else ""))
            if minutes:
                parts.append(f"{minutes} minute" + ("s" if minutes != 1 else ""))
            if seconds:
                parts.append(f"{seconds} seconde" + ("s" if seconds != 1 else ""))
            return " et ".join(parts)
        if hours:
            parts.append(f"{hours} hora" + ("s" if hours != 1 else ""))
        if minutes:
            parts.append(f"{minutes} minuto" + ("s" if minutes != 1 else ""))
        if seconds:
            parts.append(f"{seconds} segundo" + ("s" if seconds != 1 else ""))
        return " y ".join(parts)

    async def _complete_answer_loop(
        self,
        wake_transcript: str,
        question: str,
        answer: str,
        language: str,
        response_detail: str,
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
                response_detail=response_detail,
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
                response_detail=response_detail,
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
                response_detail=response_detail,
            )
            try:
                follow_up = await self._recognition_service.recognize_once(
                    wait_for_speech_seconds=self._follow_up_wait_seconds,
                    language_hint=language,
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
                response_detail=response_detail,
            )
            answer = await self._conversation_service.answer(
                question,
                language,
                response_detail,
            )
            await self._speech_service.select_language(locale)
            await self._set_status(
                busy=True,
                phase="speaking",
                detail="follow_up_ready",
                wake_transcript=wake_transcript,
                question=question,
                answer=answer,
                language=language,
                response_detail=response_detail,
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
            response_detail=response_detail,
        )

    async def _ask_response_detail(self, language: str) -> str:
        locale = "fr_FR" if language.casefold() == "fr" else "es_ES"
        prompt = (
            "Voulez-vous une reponse courte ou complete ?"
            if locale == "fr_FR"
            else "Quieres una respuesta corta o completa?"
        )
        await self._speech_service.select_language(locale)
        await self._speech_service.speak(prompt)
        await self._set_status(
            busy=True,
            phase="listening_for_response_detail",
            detail="waiting_for_response_detail",
            language=language,
        )
        try:
            choice = await self._recognition_service.recognize_once(
                wait_for_speech_seconds=(
                    self._follow_up_wait_seconds
                    if self._follow_up_wait_seconds > 0
                    else 5.0
                ),
                language_hint=language,
            )
        except NoSpeechDetectedError:
            return "complete"
        return self._response_detail(choice.text)

    async def _announce_complete_response(self, language: str) -> None:
        locale = "fr_FR" if language.casefold() == "fr" else "es_ES"
        message = (
            "D'accord, cela me prendra un peu plus de temps a preparer."
            if locale == "fr_FR"
            else "De acuerdo, tardare un poco mas en prepararla."
        )
        await self._speech_service.select_language(locale)
        await self._set_status(
            busy=True,
            phase="announcing_complete_response",
            detail="complete_response_selected",
            language=language,
            response_detail="complete",
        )
        await self._speech_service.speak(message)

    @classmethod
    def _response_detail(cls, transcript: str) -> str:
        normalized = cls._normalize(transcript)
        short_words = ("corta", "corto", "breve", "court", "courte")
        if any(re.search(rf"\b{word}\b", normalized) for word in short_words):
            return "short"
        return "complete"

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
