from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from ugassistant.adapters.preferences import YAMLPreferenceStore
from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
)
from ugassistant.api.app import create_app
from ugassistant.config import AppSettings
from ugassistant.domain.state_machine import AssistantState


class AudioStateIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_sound_changes_state_to_listening_and_silence_restores_idle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_adapter = SimulatedAudioAdapter()
            app = create_app(
                settings=AppSettings(
                    audio_activation_threshold=0.02,
                    audio_release_threshold=0.01,
                    audio_activation_samples=1,
                    audio_silence_seconds=0.01,
                ),
                camera_adapter=SimulatedCameraAdapter(),
                audio_adapter=audio_adapter,
                preference_store=YAMLPreferenceStore(
                    Path(temporary_directory) / "preferences.yaml"
                ),
            )
            service = app.state.audio_service
            await service.refresh()
            await service.enable_monitoring()

            audio_adapter.emit_input_level(0.08)
            await asyncio.sleep(0.01)
            listening = app.state.state_machine.snapshot()
            await asyncio.sleep(0.02)
            audio_adapter.emit_input_level(0.0)
            await asyncio.sleep(0.01)
            idle = app.state.state_machine.snapshot()
            await service.shutdown()

        self.assertEqual(listening.state, AssistantState.LISTENING)
        self.assertEqual(listening.detail, "audio-activity")
        self.assertEqual(idle.state, AssistantState.IDLE)
        self.assertEqual(idle.detail, "audio-silence")


if __name__ == "__main__":
    unittest.main()
