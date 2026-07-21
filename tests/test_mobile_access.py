from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI

from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedLLMAdapter,
    SimulatedSTTAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import create_app
from ugassistant.config import AppSettings
from ugassistant.services.mobile_access import MobileAccessDeniedError, MobileAccessStore
from ugassistant.services.conversation import ConversationService
from ugassistant.services.mobile_assistant import MobileAssistantService


class MobileAccessStoreTests(unittest.TestCase):
    def test_binds_a_persistent_token_to_its_first_device_and_revokes_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = MobileAccessStore(Path(temporary_directory) / "mobile.sqlite3")
            issued = store.issue("https://192.168.1.10:8443")

            device = store.authorize(
                issued["access_id"], issued["token"], "phone-a", "Pixel"
            )
            devices = store.list_devices()

            self.assertEqual(device.label, "Pixel")
            self.assertEqual(devices[0]["label"], "Pixel")
            with self.assertRaises(MobileAccessDeniedError):
                store.authorize(
                    issued["access_id"], issued["token"], "phone-b", "Otro"
                )

            store.revoke(issued["access_id"])
            self.assertEqual(store.list_devices(), [])
            with self.assertRaises(MobileAccessDeniedError):
                store.authorize(
                    issued["access_id"], issued["token"], "phone-a", "Pixel"
                )


def route_endpoint(app: FastAPI, path: str):
    return next(route.endpoint for route in app.routes if route.path == path)  # type: ignore[attr-defined]


class MobileAccessApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_issues_a_qr_credential_and_lists_the_pending_device(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(
                AppSettings(project_root=Path(temporary_directory)),
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
            )
            issued = await route_endpoint(app, "/api/mobile/access")()
            devices = await route_endpoint(app, "/api/mobile/devices")()

        self.assertTrue(issued["url"].startswith("https://"))
        self.assertIn("/?access=", issued["url"])
        self.assertNotIn("/mobile?", issued["url"])
        self.assertIn("<svg", issued["qr_svg"])
        self.assertEqual(devices["devices"][0]["label"], "Pendiente")


class MobileAssistantServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_transcribes_answers_and_synthesizes_without_host_playback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = MobileAccessStore(Path(temporary_directory) / "mobile.sqlite3")
            issued = store.issue("https://192.168.1.10:8443")
            stt = SimulatedSTTAdapter(text="Bonjour", language="fr")
            tts = SimulatedTTSAdapter()
            llm = SimulatedLLMAdapter("Bonjour depuis le modele local.")
            inference_lock = asyncio.Lock()
            service = MobileAssistantService(
                access_store=store,
                stt_adapter=stt,
                tts_adapter=tts,
                conversation_factory=lambda: ConversationService(
                    llm,
                    inference_lock=inference_lock,
                ),
                inference_lock=inference_lock,
                speech_rate=0.85,
            )

            result = await service.ask(
                access_id=issued["access_id"],
                token=issued["token"],
                device_id="android-1",
                device_label="Pixel",
                wav_bytes=b"RIFF" + (b"\x00" * 40),
            )

        self.assertEqual(result["transcript"], "Bonjour")
        self.assertEqual(result["answer"], "Bonjour depuis le modele local.")
        self.assertTrue(result["audio_wav_base64"])
        self.assertEqual(tts.synthesized, [("Bonjour depuis le modele local.", "fr_FR-tom-medium")])


if __name__ == "__main__":
    unittest.main()
