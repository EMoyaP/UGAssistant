from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile

from ugassistant.domain.ports import TTSVoice


logger = logging.getLogger("ugassistant.tts.piper")


class PiperUnavailableError(RuntimeError):
    pass


class PiperSynthesisError(RuntimeError):
    pass


@dataclass(frozen=True)
class PiperVoiceConfig:
    voice_id: str
    display_name: str
    language: str
    model_path: Path
    config_path: Path
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w: float = 0.8


class PiperTTSAdapter:
    def __init__(
        self,
        executable_path: Path,
        voices: tuple[PiperVoiceConfig, ...],
        *,
        process_timeout_seconds: float = 45.0,
    ) -> None:
        if not voices:
            raise ValueError("At least one Piper voice must be configured")
        self._executable_path = executable_path
        self._voices = {voice.voice_id: voice for voice in voices}
        self._process_timeout_seconds = max(process_timeout_seconds, 1.0)

    async def list_voices(self) -> list[TTSVoice]:
        runtime_ready = self._executable_path.is_file()
        result: list[TTSVoice] = []
        for voice in self._voices.values():
            if not runtime_ready:
                detail = "piper_runtime_missing"
            elif not voice.model_path.is_file():
                detail = "voice_model_missing"
            elif not voice.config_path.is_file():
                detail = "voice_config_missing"
            else:
                detail = "ready"
            result.append(
                TTSVoice(
                    voice_id=voice.voice_id,
                    display_name=voice.display_name,
                    language=voice.language,
                    available=detail == "ready",
                    detail=detail,
                )
            )
        return result

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        speech_rate: float = 1.0,
    ) -> bytes:
        selected_id = voice_id or next(iter(self._voices))
        voice = self._voices.get(selected_id)
        if voice is None:
            raise ValueError(f"Unknown Piper voice: {selected_id}")
        if not 0.6 <= speech_rate <= 1.3:
            raise ValueError("speech_rate must be between 0.6 and 1.3")
        self._assert_ready(voice)

        file_descriptor, output_name = tempfile.mkstemp(
            prefix="ugassistant-tts-",
            suffix=".wav",
        )
        os.close(file_descriptor)
        output_path = Path(output_name)
        command = [
            str(self._executable_path),
            "--model",
            str(voice.model_path),
            "--config",
            str(voice.config_path),
            "--output_file",
            str(output_path),
            "--length_scale",
            str(voice.length_scale / speech_rate),
            "--noise_scale",
            str(voice.noise_scale),
            "--noise_w",
            str(voice.noise_w),
        ]
        creation_flags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0
        )
        logger.info("piper_synthesis_started voice=%s", selected_id)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._executable_path.parent,
                creationflags=creation_flags,
            )
            try:
                _stdout, stderr = await asyncio.wait_for(
                    process.communicate((text + "\n").encode("utf-8")),
                    timeout=self._process_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.communicate()
                raise PiperSynthesisError(
                    f"Piper exceeded {self._process_timeout_seconds:.0f} seconds"
                ) from exc
            if process.returncode != 0:
                message = stderr.decode("utf-8", errors="replace").strip()
                raise PiperSynthesisError(
                    f"Piper failed with code {process.returncode}: {message[-500:]}"
                )
            wav_bytes = await asyncio.to_thread(output_path.read_bytes)
            if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
                raise PiperSynthesisError("Piper did not produce a valid WAV file")
            logger.info(
                "piper_synthesis_completed voice=%s bytes=%d",
                selected_id,
                len(wav_bytes),
            )
            return wav_bytes
        finally:
            output_path.unlink(missing_ok=True)

    def _assert_ready(self, voice: PiperVoiceConfig) -> None:
        if not self._executable_path.is_file():
            raise PiperUnavailableError(
                f"Piper runtime is not installed: {self._executable_path}"
            )
        if not voice.model_path.is_file():
            raise PiperUnavailableError(
                f"Piper voice model is not installed: {voice.model_path}"
            )
        if not voice.config_path.is_file():
            raise PiperUnavailableError(
                f"Piper voice config is not installed: {voice.config_path}"
            )
