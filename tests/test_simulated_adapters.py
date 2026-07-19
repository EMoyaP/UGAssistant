from __future__ import annotations

import unittest

from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedLLMAdapter,
    SimulatedSTTAdapter,
    SimulatedTTSAdapter,
)


class SimulatedAdaptersTests(unittest.IsolatedAsyncioTestCase):
    async def test_simulated_llm_returns_text(self) -> None:
        adapter = SimulatedLLMAdapter()

        response = await adapter.generate("hola")

        self.assertIn("hola", response)

    async def test_simulated_stt_returns_transcript(self) -> None:
        adapter = SimulatedSTTAdapter()

        transcript = await adapter.transcribe(b"sample wav")

        self.assertEqual(transcript.language, "es")
        self.assertIn("transcripcion", transcript.text)

    async def test_simulated_tts_returns_bytes(self) -> None:
        adapter = SimulatedTTSAdapter()

        voices = await adapter.list_voices()
        audio = await adapter.synthesize("bonjour", "fr_FR-tom-medium")

        self.assertIsInstance(audio, bytes)
        self.assertEqual(
            {(voice.voice_id, voice.language) for voice in voices},
            {
                ("es_ES-davefx-medium", "es_ES"),
                ("fr_FR-tom-medium", "fr_FR"),
            },
        )

    async def test_simulated_camera_presence(self) -> None:
        adapter = SimulatedCameraAdapter(person_detected=True)
        await adapter.open()

        presence = await adapter.read_presence()

        self.assertTrue(presence.available)
        self.assertTrue(presence.person_detected)
        self.assertEqual(presence.face_center_x, 0.5)

        frame = await adapter.read_frame()
        self.assertEqual(frame.width, 640)
        await adapter.close()
        self.assertFalse((await adapter.read_presence()).available)

    async def test_simulated_audio_devices(self) -> None:
        adapter = SimulatedAudioAdapter()

        devices = await adapter.list_devices()

        self.assertEqual({device.kind for device in devices}, {"input", "output"})
        self.assertEqual({device.device_index for device in devices}, {0, 1})
        self.assertTrue(all(device.available for device in devices))


if __name__ == "__main__":
    unittest.main()
