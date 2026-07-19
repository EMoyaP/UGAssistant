from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import create_app
from ugassistant.config import AppSettings
from ugassistant.domain.ports import CombinedGesture, CombinedGestureDetection
from ugassistant.services.camera import CameraStatus


class BlockingAudioAdapter(SimulatedAudioAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.playback_started = asyncio.Event()
        self.playback_released = asyncio.Event()

    async def play_wav(
        self,
        device_index: int,
        wav_bytes: bytes,
        volume: float,
        balance: float = 0.0,
    ) -> None:
        await super().play_wav(device_index, wav_bytes, volume, balance)
        self.playback_started.set()
        await self.playback_released.wait()

    async def stop_output(self) -> None:
        await super().stop_output()
        self.playback_released.set()


class SilenceGestureTests(unittest.IsolatedAsyncioTestCase):
    async def test_pointing_at_mouth_interrupts_current_speech_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            audio_adapter = BlockingAudioAdapter()
            app = create_app(
                AppSettings(project_root=Path(temporary_directory)),
                SimulatedCameraAdapter(),
                audio_adapter,
                SimulatedTTSAdapter(),
            )
            async with app.router.lifespan_context(app):
                await app.state.audio_service.enable_output()
                speaking = asyncio.create_task(
                    app.state.speech_service.speak("Deten esta lectura")
                )
                await asyncio.wait_for(audio_adapter.playback_started.wait(), timeout=1.0)
                status = CameraStatus(
                    enabled=True,
                    available=True,
                    model_ready=True,
                    hand_model_ready=True,
                    person_detected=True,
                    combined_gestures=(
                        CombinedGestureDetection(
                            gesture=CombinedGesture.POINTING_AT_MOUTH,
                            handedness="Right",
                            confidence=0.9,
                        ),
                    ),
                )

                await app.state.camera_service._on_status(status)  # type: ignore[misc]
                await app.state.camera_service._on_status(status)  # type: ignore[misc]

                with self.assertRaises(asyncio.CancelledError):
                    await speaking
                self.assertEqual(audio_adapter.output_stop_count, 1)
