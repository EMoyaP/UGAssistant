from __future__ import annotations

from array import array
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from ugassistant.adapters.whisper_cpp_stt import WhisperCppSTTAdapter


class FakeWhisperProcess:
    returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""

    def kill(self) -> None:
        self.returncode = -1


def sample_wav() -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(array("h", [1000] * 16000).tobytes())
    return output.getvalue()


class WhisperCppSTTAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_missing_runtime_without_starting_a_process(self) -> None:
        adapter = WhisperCppSTTAdapter(Path("missing.exe"), Path("missing.bin"))

        status = await adapter.status()

        self.assertFalse(status.available)
        self.assertEqual(status.detail, "whisper_runtime_missing")

    async def test_transcribes_json_and_detects_language(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / "whisper-cli.exe"
            model = root / "ggml-base.bin"
            executable.touch()
            model.touch()
            adapter = WhisperCppSTTAdapter(executable, model, threads=3)
            captured_command: tuple[object, ...] = ()

            async def create_process(
                *command: object,
                **_kwargs: object,
            ) -> FakeWhisperProcess:
                nonlocal captured_command
                captured_command = command
                output_base = Path(str(command[command.index("--output-file") + 1]))
                output_base.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "result": {"language": "fr"},
                            "transcription": [
                                {"text": " Bonjour,"},
                                {"text": " comment allez-vous ?"},
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                return FakeWhisperProcess()

            with patch(
                "ugassistant.adapters.whisper_cpp_stt.asyncio.create_subprocess_exec",
                new=create_process,
            ):
                result = await adapter.transcribe(sample_wav())

        self.assertEqual(result.language, "fr")
        self.assertEqual(result.text, "Bonjour, comment allez-vous ?")
        self.assertEqual(result.duration_ms, 1000)
        self.assertIn("auto", captured_command)
        self.assertIn("--no-gpu", captured_command)
        self.assertIn("3", captured_command)

    async def test_uses_french_voice_hint_when_candidate_scores_are_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable = root / "whisper-cli.exe"
            model = root / "ggml-base.bin"
            executable.touch()
            model.touch()
            adapter = WhisperCppSTTAdapter(
                executable,
                model,
                candidate_languages=("es", "fr"),
            )
            captured_commands: list[tuple[object, ...]] = []

            async def create_process(
                *command: object,
                **_kwargs: object,
            ) -> FakeWhisperProcess:
                captured_commands.append(command)
                language = str(command[command.index("--language") + 1])
                output_base = Path(str(command[command.index("--output-file") + 1]))
                if language == "auto":
                    payload = {
                        "result": {"language": "id"},
                        "transcription": [{"text": " satu dua tiga"}],
                    }
                elif language == "es":
                    payload = {
                        "result": {"language": "es"},
                        "transcription": [
                            {
                                "text": " uno dos tres",
                                "tokens": [
                                    {"text": "[_BEG_]", "p": 0.01},
                                    {"text": "<|es|>", "p": 0.99},
                                    {"text": " uno", "p": 0.82},
                                    {"text": " dos", "p": 0.8},
                                    {"text": " tres", "p": 0.81},
                                ],
                            }
                        ],
                    }
                else:
                    payload = {
                        "result": {"language": "fr"},
                        "transcription": [
                            {
                                "text": " un deux trois",
                                "tokens": [
                                    {"text": "[_BEG_]", "p": 0.99},
                                    {"text": "<|fr|>", "p": 0.99},
                                    {"text": " un", "p": 0.8},
                                    {"text": " deux", "p": 0.79},
                                    {"text": " trois", "p": 0.8},
                                ],
                            }
                        ],
                    }
                output_base.with_suffix(".json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                return FakeWhisperProcess()

            with patch(
                "ugassistant.adapters.whisper_cpp_stt.asyncio.create_subprocess_exec",
                new=create_process,
            ):
                result = await adapter.transcribe(
                    sample_wav(),
                    language_hint="fr_FR",
                )

        self.assertEqual(result.language, "fr")
        self.assertEqual(result.text, "un deux trois")
        self.assertEqual(
            [
                str(command[command.index("--language") + 1])
                for command in captured_commands
            ],
            ["auto", "es", "fr"],
        )
        self.assertNotIn("--output-json-full", captured_commands[0])
        self.assertIn("--output-json-full", captured_commands[1])
        self.assertIn("--output-json-full", captured_commands[2])


if __name__ == "__main__":
    unittest.main()
