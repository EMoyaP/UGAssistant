from __future__ import annotations

import asyncio
from array import array
from collections.abc import Callable
import unittest

from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedLLMAdapter,
    SimulatedSTTAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.services.audio import AudioDeviceService, AudioStatus
from ugassistant.services.conversation import ConversationService
from ugassistant.services.recognition import VoiceRecognitionService
from ugassistant.services.speech import SpeechService
from ugassistant.services.voice_assistant import VoiceAssistantService


class VoiceAssistantServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_wake_word_asks_response_detail_before_answering(self) -> None:
        adapter = SimulatedAudioAdapter()
        assistant: VoiceAssistantService | None = None

        async def on_audio(status: AudioStatus) -> None:
            if assistant is not None:
                assistant.observe_audio(status)

        audio = AudioDeviceService(
            adapter,
            activation_threshold=0.02,
            release_threshold=0.01,
            activation_samples=1,
            block_duration_ms=50,
            on_status=on_audio,
        )
        speech_adapter = SimulatedTTSAdapter()
        speech = SpeechService(
            speech_adapter,
            audio,
            default_voice_id="es_ES-davefx-medium",
            output_guard_seconds=0.0,
        )
        inference_lock = asyncio.Lock()
        recognition = VoiceRecognitionService(
            SimulatedSTTAdapter(
                responses=[
                    ("Hola", "es"),
                    ("Que hora es?", "es"),
                    ("completa", "es"),
                ]
            ),
            audio,
            speech,
            wait_for_speech_seconds=1.0,
            silence_seconds=0.1,
            max_recording_seconds=2.0,
            inference_lock=inference_lock,
        )
        conversation = ConversationService(
            SimulatedLLMAdapter(response="Son las diez."),
            inference_lock=inference_lock,
        )
        assistant = VoiceAssistantService(audio, recognition, speech, conversation)
        await audio.refresh()
        await speech.refresh()
        await audio.enable_monitoring()

        speech_chunk = array("h", [5000] * 800).tobytes()
        silence_chunk = array("h", [0] * 800).tobytes()
        adapter.emit_input_audio(0.15, speech_chunk)
        await self._wait_for(lambda: adapter._on_audio_chunk is not None)
        adapter.emit_input_audio(0.0, silence_chunk)
        adapter.emit_input_audio(0.0, silence_chunk)
        await self._wait_for(lambda: assistant.status.phase == "listening_for_question")

        adapter.emit_input_audio(0.15, speech_chunk)
        await self._wait_for(lambda: adapter._on_audio_chunk is not None)
        adapter.emit_input_audio(0.0, silence_chunk)
        adapter.emit_input_audio(0.0, silence_chunk)
        await self._wait_for(
            lambda: assistant.status.phase == "listening_for_response_detail"
        )

        adapter.emit_input_audio(0.15, speech_chunk)
        await self._wait_for(lambda: adapter._on_audio_chunk is not None)
        adapter.emit_input_audio(0.0, silence_chunk)
        adapter.emit_input_audio(0.0, silence_chunk)
        await self._wait_for(lambda: assistant.status.phase == "waiting_for_wake_word")

        self.assertEqual(assistant.status.answer, "Son las diez.")
        self.assertEqual(assistant.status.response_detail, "complete")
        self.assertIn(
            ("Quieres una respuesta corta o completa?", "es_ES-davefx-medium"),
            speech_adapter.synthesized,
        )
        await assistant.shutdown()
        await audio.shutdown()

    async def test_detects_spanish_and_french_wake_words(self) -> None:
        audio = AudioDeviceService(SimulatedAudioAdapter())
        speech = SpeechService(
            SimulatedTTSAdapter(),
            audio,
            default_voice_id="es_ES-davefx-medium",
        )
        recognition = VoiceRecognitionService(
            SimulatedSTTAdapter(),
            audio,
            speech,
            inference_lock=asyncio.Lock(),
        )
        service = VoiceAssistantService(
            audio,
            recognition,
            speech,
            ConversationService(
                SimulatedLLMAdapter(),
                inference_lock=asyncio.Lock(),
            ),
            spanish_wake_words=("hola",),
            french_wake_words=("salut",),
        )

        self.assertEqual(service._wake_language("hola!"), "es")
        self.assertEqual(service._wake_language("salut"), "fr")
        self.assertIsNone(service._wake_language("bonjour"))
        self.assertEqual(service._response_detail("respuesta corta"), "short")
        self.assertEqual(service._response_detail("reponse courte"), "short")
        self.assertEqual(service._response_detail("respuesta completa"), "complete")

    async def test_reports_waiting_for_wake_word_when_monitoring_is_enabled(self) -> None:
        audio = AudioDeviceService(SimulatedAudioAdapter())
        speech = SpeechService(
            SimulatedTTSAdapter(),
            audio,
            default_voice_id="es_ES-davefx-medium",
        )
        recognition = VoiceRecognitionService(
            SimulatedSTTAdapter(),
            audio,
            speech,
            inference_lock=asyncio.Lock(),
        )
        service = VoiceAssistantService(
            audio,
            recognition,
            speech,
            ConversationService(
                SimulatedLLMAdapter(),
                inference_lock=asyncio.Lock(),
            ),
        )

        await audio.refresh()
        await audio.enable_monitoring()
        service.observe_audio(audio.status)

        self.assertEqual(service.status.detail, "monitoring_wake_word")
        await audio.shutdown()

    async def _wait_for(self, predicate: Callable[[], bool], timeout: float = 1.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while loop.time() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("Timed out waiting for voice assistant state")


if __name__ == "__main__":
    unittest.main()
