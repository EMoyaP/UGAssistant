from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import logging

from ugassistant.domain.ports import STTAdapter, TranscriptionResult
from ugassistant.services.audio import (
    AudioCaptureCancelledError,
    AudioDeviceService,
    NoSpeechDetectedError,
)
from ugassistant.services.speech import SpeechService


logger = logging.getLogger("ugassistant.recognition")


class RecognitionBusyError(RuntimeError):
    pass


class UnsupportedRecognitionLanguageError(RuntimeError):
    pass


@dataclass(frozen=True)
class RecognitionStatus:
    available: bool = False
    busy: bool = False
    phase: str = "not_scanned"
    detail: str = "not_scanned"
    transcript: str = ""
    language: str | None = None
    audio_duration_ms: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "busy": self.busy,
            "phase": self.phase,
            "detail": self.detail,
            "transcript": self.transcript,
            "language": self.language,
            "audio_duration_ms": self.audio_duration_ms,
        }


RecognitionStatusCallback = Callable[[RecognitionStatus], Awaitable[None]]


class VoiceRecognitionService:
    SUPPORTED_LANGUAGES = frozenset({"es", "fr"})

    def __init__(
        self,
        adapter: STTAdapter,
        audio_service: AudioDeviceService,
        speech_service: SpeechService,
        *,
        accepted_languages: tuple[str, ...] = ("es", "fr"),
        sample_rate: int = 16000,
        wait_for_speech_seconds: float = 8.0,
        silence_seconds: float = 1.5,
        max_recording_seconds: float = 20.0,
        pre_roll_seconds: float = 0.3,
        inference_lock: asyncio.Lock | None = None,
        on_status: RecognitionStatusCallback | None = None,
    ) -> None:
        self._adapter = adapter
        self._audio_service = audio_service
        self._speech_service = speech_service
        self._accepted_languages = {
            language.casefold() for language in accepted_languages
        }
        unsupported_languages = (
            self._accepted_languages - self.SUPPORTED_LANGUAGES
        )
        if unsupported_languages:
            unsupported = ", ".join(sorted(unsupported_languages))
            raise ValueError(f"Unsupported configured STT languages: {unsupported}")
        self._sample_rate = sample_rate
        self._wait_for_speech_seconds = wait_for_speech_seconds
        self._silence_seconds = silence_seconds
        self._max_recording_seconds = max_recording_seconds
        self._pre_roll_seconds = pre_roll_seconds
        self._on_status = on_status
        self._lock = asyncio.Lock()
        self._inference_lock = inference_lock or asyncio.Lock()
        self._cancel_event = asyncio.Event()
        self._status = RecognitionStatus()

    @property
    def status(self) -> RecognitionStatus:
        return self._status

    async def refresh(self) -> RecognitionStatus:
        engine_status = await self._adapter.status()
        if not self._lock.locked():
            self._status = RecognitionStatus(
                available=engine_status.available,
                phase="ready" if engine_status.available else "unavailable",
                detail=engine_status.detail,
            )
            await self._publish()
        return self._status

    async def recognize_and_repeat(self) -> RecognitionStatus:
        await self._recognize(repeat=True)
        return self._status

    async def recognize_once(
        self,
        *,
        initial_chunks: tuple[tuple[float, bytes], ...] = (),
        wait_for_speech_seconds: float | None = None,
    ) -> TranscriptionResult:
        return await self._recognize(
            repeat=False,
            initial_chunks=initial_chunks,
            wait_for_speech_seconds=wait_for_speech_seconds,
        )

    async def _recognize(
        self,
        *,
        repeat: bool,
        initial_chunks: tuple[tuple[float, bytes], ...] = (),
        wait_for_speech_seconds: float | None = None,
    ) -> TranscriptionResult:
        if self._lock.locked():
            raise RecognitionBusyError("Voice recognition is already active")

        async with self._lock:
            engine_status = None
            resume_monitoring = False
            self._cancel_event = asyncio.Event()
            try:
                engine_status = await self._adapter.status()
                if not engine_status.available:
                    raise RuntimeError(
                        f"STT is unavailable: {engine_status.detail}"
                    )
                if self._audio_service.status.selected_input_index is None:
                    raise RuntimeError("No audio input device selected")
                if repeat and not self._audio_service.status.output_enabled:
                    raise RuntimeError("Audio output is disabled")

                resume_monitoring = self._audio_service.status.monitoring
                if resume_monitoring:
                    await self._audio_service.disable_monitoring()
                await self._set_status(
                    available=True,
                    busy=True,
                    phase="listening",
                    detail="waiting_for_speech",
                )
                wav_bytes = await self._audio_service.capture_utterance(
                    target_sample_rate=self._sample_rate,
                    wait_for_speech_seconds=(
                        self._wait_for_speech_seconds
                        if wait_for_speech_seconds is None
                        else wait_for_speech_seconds
                    ),
                    silence_seconds=self._silence_seconds,
                    max_recording_seconds=self._max_recording_seconds,
                    pre_roll_seconds=self._pre_roll_seconds,
                    initial_chunks=initial_chunks,
                    cancel_event=self._cancel_event,
                )
                await self._set_status(
                    available=True,
                    busy=True,
                    phase="transcribing",
                    detail="whisper_inference",
                )
                async with self._inference_lock:
                    result = await self._adapter.transcribe(
                        wav_bytes,
                        language_hint=self._speech_service.status.selected_language,
                    )
                self._validate_language(result)
                await self._set_status(
                    available=True,
                    busy=True,
                    phase="recognized",
                    detail="transcription_ready",
                    transcript=result.text,
                    language=result.language,
                    audio_duration_ms=result.duration_ms,
                )
                if repeat:
                    await self._speech_service.select_language(
                        self._voice_locale(result.language)
                    )
                    await self._speech_service.speak(result.text)
                await self._set_status(
                    available=True,
                    busy=False,
                    phase="completed",
                    detail=("recognized_and_repeated" if repeat else "recognized"),
                    transcript=result.text,
                    language=result.language,
                    audio_duration_ms=result.duration_ms,
                )
                return result
            except AudioCaptureCancelledError:
                await self._set_status(
                    available=True,
                    busy=False,
                    phase="cancelled",
                    detail="recognition_cancelled",
                )
                raise
            except NoSpeechDetectedError:
                await self._set_status(
                    available=True,
                    busy=False,
                    phase="timeout",
                    detail="no_speech_detected",
                )
                raise
            except UnsupportedRecognitionLanguageError as exc:
                await self._set_status(
                    available=True,
                    busy=False,
                    phase="unsupported_language",
                    detail=str(exc),
                )
                raise
            except Exception as exc:
                await self._set_status(
                    available=(
                        engine_status.available
                        if engine_status is not None
                        else False
                    ),
                    busy=False,
                    phase="error",
                    detail=str(exc),
                )
                raise
            finally:
                if resume_monitoring:
                    try:
                        await self._audio_service.enable_monitoring()
                    except Exception:
                        logger.exception("audio_monitor_restore_failed")

    def cancel(self) -> RecognitionStatus:
        self._cancel_event.set()
        self._audio_service.cancel_capture()
        return self._status

    async def shutdown(self) -> None:
        self.cancel()

    def _validate_language(self, result: TranscriptionResult) -> None:
        if result.language.casefold() not in self._accepted_languages:
            raise UnsupportedRecognitionLanguageError(
                f"Unsupported detected language: {result.language}"
            )

    @staticmethod
    def _voice_locale(language: str) -> str:
        locales = {"es": "es_ES", "fr": "fr_FR"}
        return locales[language.casefold()]

    async def _set_status(
        self,
        *,
        available: bool,
        busy: bool,
        phase: str,
        detail: str,
        transcript: str = "",
        language: str | None = None,
        audio_duration_ms: int | None = None,
    ) -> None:
        self._status = RecognitionStatus(
            available=available,
            busy=busy,
            phase=phase,
            detail=detail,
            transcript=transcript,
            language=language,
            audio_duration_ms=audio_duration_ms,
        )
        await self._publish()

    async def _publish(self) -> None:
        if self._on_status is not None:
            await self._on_status(self._status)
