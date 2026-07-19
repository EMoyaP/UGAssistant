from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ugassistant.adapters.piper_tts import PiperTTSAdapter, PiperVoiceConfig


class FakePiperProcess:
    returncode = 0

    def __init__(self) -> None:
        self.input_bytes = b""

    async def communicate(self, input_bytes: bytes) -> tuple[bytes, bytes]:
        self.input_bytes = input_bytes
        return b"", b""

    def kill(self) -> None:
        self.returncode = -1


class PiperTTSAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_voice_as_ready_only_when_all_files_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / "piper.exe"
            model = root / "voice.onnx"
            config = root / "voice.onnx.json"
            executable.touch()
            model.touch()
            config.touch()
            adapter = PiperTTSAdapter(
                executable,
                (
                    PiperVoiceConfig(
                        voice_id="es_ES-davefx-medium",
                        display_name="DaveFX",
                        language="es_ES",
                        model_path=model,
                        config_path=config,
                    ),
                ),
            )

            voices = await adapter.list_voices()

            self.assertTrue(voices[0].available)
            self.assertEqual(voices[0].language, "es_ES")

    async def test_synthesis_passes_utf8_text_and_locked_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / "piper.exe"
            model = root / "voice.onnx"
            config = root / "voice.onnx.json"
            executable.touch()
            model.touch()
            config.touch()
            adapter = PiperTTSAdapter(
                executable,
                (
                    PiperVoiceConfig(
                        voice_id="es_ES-davefx-medium",
                        display_name="DaveFX",
                        language="es_ES",
                        model_path=model,
                        config_path=config,
                        length_scale=1.0,
                        noise_scale=0.667,
                        noise_w=0.8,
                    ),
                ),
            )
            process = FakePiperProcess()
            captured_command: tuple[object, ...] = ()

            async def create_process(*command: object, **_kwargs: object) -> FakePiperProcess:
                nonlocal captured_command
                captured_command = command
                output_index = command.index("--output_file") + 1
                Path(str(command[output_index])).write_bytes(b"RIFF" + bytes(40))
                return process

            with patch(
                "ugassistant.adapters.piper_tts.asyncio.create_subprocess_exec",
                new=create_process,
            ):
                audio = await adapter.synthesize(
                    "Hola, que tal?",
                    "es_ES-davefx-medium",
                    speech_rate=0.8,
                )

            self.assertTrue(audio.startswith(b"RIFF"))
            self.assertEqual(process.input_bytes, "Hola, que tal?\n".encode("utf-8"))
            self.assertIn("--length_scale", captured_command)
            self.assertIn("1.25", captured_command)
            self.assertIn("0.667", captured_command)
            self.assertIn("0.8", captured_command)


if __name__ == "__main__":
    unittest.main()
