from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import tempfile

from ugassistant.domain.audio_pcm import wav_duration_ms
from ugassistant.domain.ports import (
    STTEngineStatus,
    TranscriptionResult,
)


logger = logging.getLogger("ugassistant.stt.whisper_cpp")


class WhisperCppUnavailableError(RuntimeError):
    pass


class WhisperCppTranscriptionError(RuntimeError):
    pass


class WhisperCppSTTAdapter:
    LANGUAGE_HINT_SCORE_MARGIN = 0.05

    def __init__(
        self,
        executable_path: Path,
        model_path: Path,
        *,
        threads: int = 4,
        process_timeout_seconds: float = 120.0,
        candidate_languages: tuple[str, ...] = ("es", "fr"),
    ) -> None:
        self._executable_path = executable_path
        self._model_path = model_path
        self._threads = min(max(int(threads), 1), 8)
        self._process_timeout_seconds = max(process_timeout_seconds, 1.0)
        self._candidate_languages = tuple(
            dict.fromkeys(language.casefold() for language in candidate_languages)
        )
        if not self._candidate_languages:
            raise ValueError("At least one STT candidate language is required")

    async def status(self) -> STTEngineStatus:
        if not self._executable_path.is_file():
            return STTEngineStatus(False, "whisper_runtime_missing")
        if not self._model_path.is_file():
            return STTEngineStatus(False, "whisper_model_missing")
        return STTEngineStatus(True, "ready")

    async def transcribe(
        self,
        wav_bytes: bytes,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        engine_status = await self.status()
        if not engine_status.available:
            raise WhisperCppUnavailableError(engine_status.detail)
        if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
            raise ValueError("whisper.cpp requires a valid PCM WAV payload")

        with tempfile.TemporaryDirectory(prefix="ugassistant-stt-") as directory:
            temporary_root = Path(directory)
            audio_path = temporary_root / "utterance.wav"
            audio_path.write_bytes(wav_bytes)
            logger.info("whisper_transcription_started bytes=%d", len(wav_bytes))
            language, text, _score = await self._run_transcription(
                audio_path,
                temporary_root / "transcription-auto",
                language="auto",
                include_confidence=False,
            )
            if language not in self._candidate_languages:
                logger.info(
                    "whisper_language_fallback detected=%s candidates=%s",
                    language,
                    ",".join(self._candidate_languages),
                )
                language, text = await self._select_restricted_language(
                    audio_path,
                    temporary_root,
                    language_hint=self._normalize_language_hint(language_hint),
                )

            result = TranscriptionResult(
                text=text,
                language=language,
                duration_ms=wav_duration_ms(wav_bytes),
            )
            logger.info(
                "whisper_transcription_completed language=%s characters=%d",
                result.language,
                len(result.text),
            )
            return result

    async def _select_restricted_language(
        self,
        audio_path: Path,
        temporary_root: Path,
        *,
        language_hint: str | None,
    ) -> tuple[str, str]:
        candidates: list[tuple[float, str, str]] = []
        errors: list[str] = []
        for language in self._candidate_languages:
            try:
                _detected, text, score = await self._run_transcription(
                    audio_path,
                    temporary_root / f"transcription-{language}",
                    language=language,
                    include_confidence=True,
                )
            except WhisperCppTranscriptionError as exc:
                errors.append(f"{language}: {exc}")
                continue
            if not math.isfinite(score):
                errors.append(f"{language}: token confidence unavailable")
                continue
            candidates.append((score, language, text))
        if not candidates:
            raise WhisperCppTranscriptionError(
                "Restricted language fallback failed: " + "; ".join(errors)
            )
        score, language, text = max(candidates, key=lambda candidate: candidate[0])
        selection_reason = "confidence"
        hinted_candidate = next(
            (
                candidate
                for candidate in candidates
                if candidate[1] == language_hint
            ),
            None,
        )
        if (
            hinted_candidate is not None
            and score - hinted_candidate[0] <= self.LANGUAGE_HINT_SCORE_MARGIN
        ):
            score, language, text = hinted_candidate
            selection_reason = "selected_voice_tie_break"
        logger.info(
            "whisper_language_selected language=%s confidence_score=%.4f reason=%s",
            language,
            score,
            selection_reason,
        )
        return language, text

    def _normalize_language_hint(self, language_hint: str | None) -> str | None:
        if language_hint is None:
            return None
        normalized = language_hint.casefold().replace("-", "_").split("_", 1)[0]
        return normalized if normalized in self._candidate_languages else None

    async def _run_transcription(
        self,
        audio_path: Path,
        output_base: Path,
        *,
        language: str,
        include_confidence: bool,
    ) -> tuple[str, str, float]:
        output_path = output_base.with_suffix(".json")
        command = [
            str(self._executable_path),
            "--model",
            str(self._model_path),
            "--file",
            str(audio_path),
            "--language",
            language,
            "--threads",
            str(self._threads),
            "--no-gpu",
            "--no-prints",
            (
                "--output-json-full"
                if include_confidence
                else "--output-json"
            ),
            "--output-file",
            str(output_base),
        ]
        creation_flags = (
            getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if os.name == "nt"
            else 0
        )
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._executable_path.parent,
            creationflags=creation_flags,
        )
        try:
            _stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._process_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise WhisperCppTranscriptionError(
                "whisper.cpp exceeded "
                f"{self._process_timeout_seconds:.0f} seconds"
            ) from exc
        if process.returncode != 0:
            message = stderr.decode("utf-8", errors="replace").strip()
            raise WhisperCppTranscriptionError(
                f"whisper.cpp failed with code {process.returncode}: "
                f"{message[-800:]}"
            )
        if not output_path.is_file():
            raise WhisperCppTranscriptionError(
                "whisper.cpp did not produce JSON output"
            )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        detected_language = str(
            payload.get("result", {}).get("language", "")
        ).casefold()
        segments = payload.get("transcription", [])
        text = " ".join(
            str(segment.get("text", "")).strip()
            for segment in segments
            if isinstance(segment, dict) and str(segment.get("text", "")).strip()
        )
        text = " ".join(text.split())
        if not detected_language:
            raise WhisperCppTranscriptionError(
                "whisper.cpp did not detect a language"
            )
        if not text:
            raise WhisperCppTranscriptionError(
                "whisper.cpp did not recognize any text"
            )
        return detected_language, text, self._confidence_score(segments)

    @staticmethod
    def _confidence_score(segments: object) -> float:
        probabilities: list[float] = []
        if not isinstance(segments, list):
            return float("-inf")
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            tokens = segment.get("tokens", [])
            if not isinstance(tokens, list):
                continue
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                token_text = str(token.get("text", ""))
                if token_text.startswith(("<|", "[_")):
                    continue
                try:
                    probability = float(token.get("p", 0.0))
                except (TypeError, ValueError):
                    continue
                if probability > 0.0:
                    probabilities.append(min(probability, 1.0))
        if not probabilities:
            return float("-inf")
        return sum(math.log(probability) for probability in probabilities) / len(
            probabilities
        )
