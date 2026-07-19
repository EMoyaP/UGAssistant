from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ugassistant.domain.lip_sync import LipSyncTrack, build_lip_sync_track
from ugassistant.domain.ports import TTSAdapter, TTSVoice
from ugassistant.services.audio import AudioDeviceService


logger = logging.getLogger("ugassistant.speech")


class SpeechBusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class SpeechStatus:
    voices: tuple[TTSVoice, ...] = ()
    languages: tuple[str, ...] = ()
    selected_voice_id: str | None = None
    selected_language: str | None = None
    available: bool = False
    output_ready: bool = False
    ready: bool = False
    busy: bool = False
    phase: str = "not_scanned"
    detail: str = "not_scanned"
    mouth_cue_interval_ms: int | None = None
    audio_duration_ms: int | None = None
    mouth_levels: tuple[float, ...] = ()
    speech_rate: float = 0.85

    def to_dict(self) -> dict[str, object]:
        return {
            "voices": [voice.to_dict() for voice in self.voices],
            "languages": list(self.languages),
            "selected_voice_id": self.selected_voice_id,
            "selected_language": self.selected_language,
            "available": self.available,
            "output_ready": self.output_ready,
            "ready": self.ready,
            "busy": self.busy,
            "phase": self.phase,
            "detail": self.detail,
            "mouth_cue_interval_ms": self.mouth_cue_interval_ms,
            "audio_duration_ms": self.audio_duration_ms,
            "mouth_levels": list(self.mouth_levels),
            "speech_rate": self.speech_rate,
        }


SpeechStatusCallback = Callable[[SpeechStatus], Awaitable[None]]
BalanceProvider = Callable[[], float]


class SpeechService:
    def __init__(
        self,
        adapter: TTSAdapter,
        audio_service: AudioDeviceService,
        *,
        default_voice_id: str,
        default_speech_rate: float = 0.85,
        max_text_length: int = 500,
        output_guard_seconds: float = 0.2,
        on_status: SpeechStatusCallback | None = None,
        balance_provider: BalanceProvider | None = None,
    ) -> None:
        self._adapter = adapter
        self._audio_service = audio_service
        self._default_voice_id = default_voice_id
        self._speech_rate = min(max(default_speech_rate, 0.6), 1.3)
        self._max_text_length = max(1, max_text_length)
        self._output_guard_seconds = max(0.0, output_guard_seconds)
        self._on_status = on_status
        self._balance_provider = balance_provider
        self._selected_voice_id: str | None = None
        self._selection_initialized = False
        self._voices: tuple[TTSVoice, ...] = ()
        self._lock = asyncio.Lock()
        self._lip_sync_track: LipSyncTrack | None = None
        self._active_task: asyncio.Task[object] | None = None
        self._status = SpeechStatus()

    @property
    def status(self) -> SpeechStatus:
        return self._status

    async def refresh(self) -> SpeechStatus:
        self._voices = tuple(await self._adapter.list_voices())
        voice_ids = {voice.voice_id for voice in self._voices}
        if not self._selection_initialized or self._selected_voice_id not in voice_ids:
            preferred = next(
                (
                    voice
                    for voice in self._voices
                    if voice.voice_id == self._default_voice_id
                ),
                None,
            )
            fallback = next((voice for voice in self._voices if voice.available), None)
            selected = preferred or fallback or (self._voices[0] if self._voices else None)
            self._selected_voice_id = selected.voice_id if selected else None
            self._selection_initialized = True
        self._status = self._build_status(phase="ready", busy=False)
        await self._publish()
        return self._status

    async def select_voice(self, voice_id: str) -> SpeechStatus:
        await self.refresh()
        selected = next(
            (voice for voice in self._voices if voice.voice_id == voice_id),
            None,
        )
        if selected is None:
            raise ValueError(f"Unknown TTS voice: {voice_id}")
        if not selected.available:
            raise RuntimeError(f"TTS voice is unavailable: {selected.detail}")
        self._selected_voice_id = voice_id
        self._selection_initialized = True
        self._status = self._build_status(phase="ready", busy=False)
        await self._publish()
        return self._status

    async def select_language(self, language: str) -> SpeechStatus:
        await self.refresh()
        selected = next(
            (
                voice
                for voice in self._voices
                if voice.language == language and voice.available
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"Unsupported TTS language: {language}")
        self._selected_voice_id = selected.voice_id
        self._selection_initialized = True
        self._status = self._build_status(phase="ready", busy=False)
        await self._publish()
        return self._status

    async def set_speech_rate(self, speech_rate: float) -> SpeechStatus:
        if not 0.6 <= speech_rate <= 1.3:
            raise ValueError("Speech rate must be between 60% and 130%")
        if self._lock.locked():
            raise SpeechBusyError("Cannot change speech rate while speaking")
        self._speech_rate = speech_rate
        self._status = self._build_status(phase="ready", busy=False)
        await self._publish()
        return self._status

    async def interrupt(self, *, cancel_task: bool = True) -> SpeechStatus:
        await self._audio_service.stop_output()
        active_task = self._active_task
        if (
            cancel_task
            and active_task is not None
            and active_task is not asyncio.current_task()
        ):
            active_task.cancel()
        if self._status.busy:
            self._lip_sync_track = None
            self._status = self._build_status(
                phase="ready",
                busy=False,
                detail="interrupted",
            )
            await self._publish()
        return self._status

    async def speak(self, text: str) -> SpeechStatus:
        normalized_text = " ".join(text.split())
        if not normalized_text:
            raise ValueError("Text to synthesize cannot be empty")
        if len(normalized_text) > self._max_text_length:
            raise ValueError(
                f"Text exceeds the {self._max_text_length} character limit"
            )
        if self._lock.locked():
            raise SpeechBusyError("Another speech synthesis is already active")

        async with self._lock:
            current_task = asyncio.current_task()
            self._active_task = current_task
            await self.refresh()
            voice = self._selected_voice()
            if voice is None or not voice.available:
                detail = voice.detail if voice is not None else "voice_not_selected"
                raise RuntimeError(f"TTS is unavailable: {detail}")
            if not self._audio_service.status.output_enabled:
                raise RuntimeError("Audio output is disabled")
            if self._audio_service.status.selected_output_index is None:
                raise RuntimeError("No audio output device selected")

            resume_monitoring = self._audio_service.status.monitoring
            if resume_monitoring:
                await self._audio_service.disable_monitoring()
            try:
                self._status = self._build_status(
                    phase="synthesizing",
                    busy=True,
                )
                await self._publish()
                wav_bytes = await self._adapter.synthesize(
                    normalized_text,
                    voice.voice_id,
                    self._speech_rate,
                )
                self._lip_sync_track = build_lip_sync_track(wav_bytes)
                self._status = self._build_status(phase="playing", busy=True)
                await self._publish()
                balance = 0.0
                if self._balance_provider is not None:
                    balance = min(max(float(self._balance_provider()), -1.0), 1.0)
                await self._audio_service.play_wav(wav_bytes, balance=balance)
                if self._output_guard_seconds:
                    await asyncio.sleep(self._output_guard_seconds)
                self._lip_sync_track = None
                self._status = self._build_status(phase="ready", busy=False)
                await self._publish()
                return self._status
            except asyncio.CancelledError:
                self._lip_sync_track = None
                self._status = self._build_status(
                    phase="ready",
                    busy=False,
                    detail="interrupted",
                )
                await self._publish()
                raise
            except Exception as exc:
                self._lip_sync_track = None
                self._status = self._build_status(
                    phase="error",
                    busy=False,
                    detail=str(exc),
                )
                await self._publish()
                raise
            finally:
                if self._active_task is current_task:
                    self._active_task = None
                if resume_monitoring:
                    try:
                        await self._audio_service.enable_monitoring()
                    except Exception:
                        logger.exception("audio_monitor_resume_after_speech_failed")

    def _selected_voice(self) -> TTSVoice | None:
        return next(
            (
                voice
                for voice in self._voices
                if voice.voice_id == self._selected_voice_id
            ),
            None,
        )

    def _build_status(
        self,
        *,
        phase: str,
        busy: bool,
        detail: str | None = None,
    ) -> SpeechStatus:
        selected = self._selected_voice()
        languages = tuple(sorted({voice.language for voice in self._voices}))
        available = selected is not None and selected.available
        output_ready = (
            self._audio_service.status.output_enabled
            and self._audio_service.status.selected_output_index is not None
        )
        if detail is None:
            if selected is None:
                detail = "voice_not_selected"
            elif not selected.available:
                detail = selected.detail
            elif not output_ready:
                detail = "audio_output_not_ready"
            else:
                detail = phase
        return SpeechStatus(
            voices=self._voices,
            languages=languages,
            selected_voice_id=selected.voice_id if selected else None,
            selected_language=selected.language if selected else None,
            available=available,
            output_ready=output_ready,
            ready=available and output_ready and not busy,
            busy=busy,
            phase=phase,
            detail=detail,
            mouth_cue_interval_ms=(
                self._lip_sync_track.interval_ms
                if self._lip_sync_track is not None
                else None
            ),
            audio_duration_ms=(
                self._lip_sync_track.duration_ms
                if self._lip_sync_track is not None
                else None
            ),
            mouth_levels=(
                self._lip_sync_track.levels
                if self._lip_sync_track is not None
                else ()
            ),
            speech_rate=self._speech_rate,
        )

    async def _publish(self) -> None:
        if self._on_status is not None:
            await self._on_status(self._status)
