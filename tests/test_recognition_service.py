from __future__ import annotations

import asyncio
from array import array
import unittest

from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedSTTAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.services.audio import AudioDeviceService
from ugassistant.services.recognition import VoiceRecognitionService
from ugassistant.services.recognition import UnsupportedRecognitionLanguageError
from ugassistant.services.speech import SpeechService


class VoiceRecognitionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_detects_french_selects_tom_and_repeats_transcript(self) -> None:
        audio_adapter = SimulatedAudioAdapter()
        audio_service = AudioDeviceService(
            audio_adapter,
            activation_threshold=0.02,
            release_threshold=0.01,
            activation_samples=1,
            block_duration_ms=50,
        )
        tts_adapter = SimulatedTTSAdapter()
        speech_service = SpeechService(
            tts_adapter,
            audio_service,
            default_voice_id="es_ES-davefx-medium",
            output_guard_seconds=0.0,
        )
        stt_adapter = SimulatedSTTAdapter(
            text="Bonjour depuis UGAssistant",
            language="fr",
        )
        recognition_service = VoiceRecognitionService(
            stt_adapter,
            audio_service,
            speech_service,
            wait_for_speech_seconds=1.0,
            silence_seconds=0.1,
            max_recording_seconds=2.0,
        )
        await audio_service.refresh()
        await speech_service.refresh()
        await audio_service.enable_monitoring()

        task = asyncio.create_task(recognition_service.recognize_and_repeat())
        for _attempt in range(50):
            if audio_adapter._on_audio_chunk is not None:
                break
            await asyncio.sleep(0.01)
        speech_chunk = array("h", [5000] * 800).tobytes()
        silence_chunk = array("h", [0] * 800).tobytes()
        audio_adapter.emit_input_audio(0.15, speech_chunk)
        audio_adapter.emit_input_audio(0.0, silence_chunk)
        audio_adapter.emit_input_audio(0.0, silence_chunk)
        status = await task

        self.assertEqual(status.phase, "completed")
        self.assertEqual(status.language, "fr")
        self.assertEqual(status.transcript, "Bonjour depuis UGAssistant")
        self.assertEqual(
            tts_adapter.synthesized,
            [("Bonjour depuis UGAssistant", "fr_FR-tom-medium")],
        )
        self.assertTrue(audio_service.status.monitoring)
        self.assertEqual(len(stt_adapter.transcribed_audio), 1)

    async def test_rejects_language_other_than_spanish_or_french(self) -> None:
        audio_adapter = SimulatedAudioAdapter()
        audio_service = AudioDeviceService(
            audio_adapter,
            activation_threshold=0.02,
            release_threshold=0.01,
            activation_samples=1,
            block_duration_ms=50,
        )
        tts_adapter = SimulatedTTSAdapter()
        speech_service = SpeechService(
            tts_adapter,
            audio_service,
            default_voice_id="es_ES-davefx-medium",
            output_guard_seconds=0.0,
        )
        recognition_service = VoiceRecognitionService(
            SimulatedSTTAdapter(text="Hello", language="en"),
            audio_service,
            speech_service,
            wait_for_speech_seconds=1.0,
            silence_seconds=0.1,
            max_recording_seconds=2.0,
        )
        await audio_service.refresh()
        await speech_service.refresh()

        task = asyncio.create_task(recognition_service.recognize_and_repeat())
        for _attempt in range(50):
            if audio_adapter._on_audio_chunk is not None:
                break
            await asyncio.sleep(0.01)
        speech_chunk = array("h", [5000] * 800).tobytes()
        silence_chunk = array("h", [0] * 800).tobytes()
        audio_adapter.emit_input_audio(0.15, speech_chunk)
        audio_adapter.emit_input_audio(0.0, silence_chunk)
        audio_adapter.emit_input_audio(0.0, silence_chunk)

        with self.assertRaises(UnsupportedRecognitionLanguageError):
            await task
        self.assertEqual(recognition_service.status.phase, "unsupported_language")
        self.assertEqual(tts_adapter.synthesized, [])


if __name__ == "__main__":
    unittest.main()
