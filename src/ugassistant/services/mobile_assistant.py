from __future__ import annotations

import asyncio
import base64
from collections.abc import Callable

from ugassistant.domain.ports import STTAdapter, TTSAdapter
from ugassistant.services.conversation import ConversationService
from ugassistant.services.mobile_access import MobileAccessStore


class MobileAudioError(ValueError):
    pass


class MobileAssistantService:
    """Runs phone audio through the existing local STT, LLM and TTS engines."""

    def __init__(
        self,
        *,
        access_store: MobileAccessStore,
        stt_adapter: STTAdapter,
        tts_adapter: TTSAdapter,
        conversation_factory: Callable[[], ConversationService],
        inference_lock: asyncio.Lock,
        speech_rate: float,
    ) -> None:
        self._access_store = access_store
        self._stt_adapter = stt_adapter
        self._tts_adapter = tts_adapter
        self._conversation_factory = conversation_factory
        self._inference_lock = inference_lock
        self._speech_rate = speech_rate
        self._sessions: dict[str, ConversationService] = {}
        self._lock = asyncio.Lock()

    async def ask(
        self,
        *,
        access_id: str,
        token: str,
        device_id: str,
        device_label: str,
        wav_bytes: bytes,
    ) -> dict[str, object]:
        if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
            raise MobileAudioError("mobile_audio_must_be_pcm_wav")
        if len(wav_bytes) > 8 * 1024 * 1024:
            raise MobileAudioError("mobile_audio_too_large")
        await asyncio.to_thread(
            self._access_store.authorize,
            access_id,
            token,
            device_id,
            device_label,
        )
        async with self._lock:
            async with self._inference_lock:
                transcription = await self._stt_adapter.transcribe(wav_bytes)
            language = transcription.language.casefold()
            if language not in {"es", "fr"}:
                language = "es"
            session = self._sessions.get(access_id)
            if session is None:
                session = self._conversation_factory()
                self._sessions[access_id] = session
            answer = await session.answer(transcription.text, language, "short")
            voice_id = await self._voice_for_language(language)
            audio = await self._tts_adapter.synthesize(
                answer,
                voice_id=voice_id,
                speech_rate=self._speech_rate,
            )
        return {
            "transcript": transcription.text,
            "language": language,
            "answer": answer,
            "audio_wav_base64": base64.b64encode(audio).decode("ascii"),
        }

    async def _voice_for_language(self, language: str) -> str:
        voices = await self._tts_adapter.list_voices()
        voice = next(
            (
                candidate
                for candidate in voices
                if candidate.available and candidate.language.casefold().startswith(language)
            ),
            None,
        )
        if voice is None:
            raise RuntimeError(f"mobile_voice_unavailable:{language}")
        return voice.voice_id
