from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from typing import Any, Callable

from fastapi import FastAPI

from ugassistant.adapters.preferences import YAMLPreferenceStore
from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import create_app
from ugassistant.config import AppSettings
from ugassistant.domain.preferences import DevicePreference, UserPreferences


def route_endpoint(app: FastAPI, path: str) -> Callable[..., Any]:
    return next(route.endpoint for route in app.routes if route.path == path)  # type: ignore[attr-defined]


class PreferenceRestartTests(unittest.IsolatedAsyncioTestCase):
    async def test_migrates_a_removed_voice_to_the_same_language(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = YAMLPreferenceStore(root / "data" / "preferences.yaml")
            store.save(
                UserPreferences(
                    voice_id="fr_FR-siwis-medium",
                    language="fr_FR",
                )
            )
            app = create_app(
                AppSettings(project_root=root),
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
                preference_store=store,
            )

            async with app.router.lifespan_context(app):
                self.assertEqual(
                    app.state.speech_service.status.selected_voice_id,
                    "fr_FR-tom-medium",
                )

            migrated = store.load()
            self.assertEqual(migrated.voice_id, "fr_FR-tom-medium")
            self.assertEqual(migrated.language, "fr_FR")

    async def test_restores_settings_after_a_server_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = YAMLPreferenceStore(root / "data" / "preferences.yaml")
            store.save(
                UserPreferences(
                    camera_device=DevicePreference("Simulated camera", 999),
                    camera_enabled=True,
                    microphone_device=DevicePreference(
                        "Simulated microphone",
                        999,
                        "Simulated",
                    ),
                    microphone_enabled=True,
                    speaker_device=DevicePreference(
                        "Simulated speakers",
                        999,
                        "Simulated",
                    ),
                    speaker_enabled=True,
                    output_volume=0.42,
                )
            )
            settings = AppSettings(project_root=root)

            first_app = create_app(
                settings,
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
                preference_store=store,
            )
            async with first_app.router.lifespan_context(first_app):
                self.assertTrue(first_app.state.camera_service.status.enabled)
                self.assertEqual(
                    first_app.state.camera_service.status.selected_device_index,
                    0,
                )
                self.assertTrue(first_app.state.audio_service.status.monitoring)
                self.assertEqual(
                    first_app.state.audio_service.status.selected_input_index,
                    0,
                )
                self.assertEqual(
                    first_app.state.audio_service.status.selected_output_index,
                    1,
                )
                self.assertEqual(first_app.state.audio_service.status.output_volume, 0.42)

                await route_endpoint(
                    first_app,
                    "/api/audio/output/volume/{volume_percent}",
                )(37)
                await route_endpoint(first_app, "/api/camera/disable")()
                await route_endpoint(first_app, "/api/audio/disable")()
                await route_endpoint(first_app, "/api/audio/output/disable")()
                await route_endpoint(
                    first_app,
                    "/api/tts/select/{voice_id}",
                )("fr_FR-tom-medium")
                await route_endpoint(
                    first_app,
                    "/api/tts/speed/{speed_percent}",
                )(80)

            second_app = create_app(
                settings,
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
                preference_store=store,
            )
            async with second_app.router.lifespan_context(second_app):
                camera_status = second_app.state.camera_service.status
                audio_status = second_app.state.audio_service.status
                self.assertFalse(camera_status.enabled)
                self.assertEqual(camera_status.selected_device_index, 0)
                self.assertFalse(audio_status.monitoring)
                self.assertFalse(audio_status.output_enabled)
                self.assertEqual(audio_status.selected_input_index, 0)
                self.assertEqual(audio_status.selected_output_index, 1)
                self.assertEqual(audio_status.output_volume, 0.37)
                self.assertEqual(
                    second_app.state.speech_service.status.selected_voice_id,
                    "fr_FR-tom-medium",
                )
                self.assertEqual(
                    second_app.state.speech_service.status.selected_language,
                    "fr_FR",
                )
                self.assertEqual(second_app.state.speech_service.status.speech_rate, 0.8)


if __name__ == "__main__":
    unittest.main()
