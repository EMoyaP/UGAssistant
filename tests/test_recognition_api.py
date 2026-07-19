from __future__ import annotations

import asyncio
from array import array
from pathlib import Path
import tempfile
import unittest
from typing import Any, Callable

from fastapi import FastAPI

from ugassistant.adapters.preferences import YAMLPreferenceStore
from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedSTTAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import create_app
from ugassistant.config import AppSettings


def route_endpoint(app: FastAPI, path: str) -> Callable[..., Any]:
    return next(route.endpoint for route in app.routes if route.path == path)  # type: ignore[attr-defined]


class RecognitionApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_endpoint_transcribes_french_repeats_and_persists_voice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            audio_adapter = SimulatedAudioAdapter()
            tts_adapter = SimulatedTTSAdapter()
            store = YAMLPreferenceStore(root / "data" / "preferences.yaml")
            app = create_app(
                AppSettings(
                    project_root=root,
                    audio_activation_threshold=0.02,
                    audio_release_threshold=0.01,
                    audio_activation_samples=2,
                    audio_block_duration_ms=50,
                    stt_wait_for_speech_seconds=1.0,
                    stt_silence_seconds=0.1,
                    stt_max_recording_seconds=2.0,
                    stt_pre_roll_seconds=0.0,
                    tts_output_guard_seconds=0.0,
                ),
                SimulatedCameraAdapter(),
                audio_adapter,
                tts_adapter,
                SimulatedSTTAdapter(
                    text="Bonjour depuis UGAssistant",
                    language="fr",
                ),
                preference_store=store,
            )

            async with app.router.lifespan_context(app):
                recognition_task = asyncio.create_task(
                    route_endpoint(app, "/api/stt/recognize")()
                )
                for _attempt in range(50):
                    if audio_adapter._on_audio_chunk is not None:
                        break
                    await asyncio.sleep(0.01)

                speech_chunk = array("h", [5000] * 800).tobytes()
                silence_chunk = array("h", [0] * 800).tobytes()
                audio_adapter.emit_input_audio(0.15, speech_chunk)
                audio_adapter.emit_input_audio(0.15, speech_chunk)
                audio_adapter.emit_input_audio(0.0, silence_chunk)
                audio_adapter.emit_input_audio(0.0, silence_chunk)
                response = await recognition_task

                self.assertEqual(response["phase"], "completed")
                self.assertEqual(response["language"], "fr")
                self.assertEqual(
                    response["transcript"],
                    "Bonjour depuis UGAssistant",
                )
                self.assertEqual(
                    tts_adapter.synthesized,
                    [("Bonjour depuis UGAssistant", "fr_FR-tom-medium")],
                )
                self.assertEqual(app.state.state_machine.state.value, "IDLE")

            preferences = store.load()
            self.assertEqual(preferences.language, "fr_FR")
            self.assertEqual(preferences.voice_id, "fr_FR-tom-medium")


if __name__ == "__main__":
    unittest.main()
