from __future__ import annotations

import argparse
import asyncio
import io
import json
import platform
import sys
import wave
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ugassistant.adapters.piper_tts import (  # noqa: E402
    PiperTTSAdapter,
    PiperVoiceConfig,
)
from ugassistant.adapters.whisper_cpp_stt import WhisperCppSTTAdapter  # noqa: E402
from ugassistant.config import load_app_settings  # noqa: E402
from ugassistant.domain.audio_pcm import pcm16_mono_to_wav  # noqa: E402


SAMPLE_TEXTS = {
    "es": "Hola, soy UGAssistant y reconozco la voz en español.",
    "fr": "Bonjour, je suis UGAssistant et je reconnais la voix en français.",
}


async def run_check(language: str, text: str) -> dict[str, object]:
    settings = load_app_settings()
    piper_executable = settings.tts_executable_path(
        platform.system(),
        platform.machine(),
    )
    whisper_executable = settings.stt_executable_path(
        platform.system(),
        platform.machine(),
    )
    if piper_executable is None or whisper_executable is None:
        raise RuntimeError("Current platform has no locked speech runtimes")

    voice = (
        PiperVoiceConfig(
            voice_id=settings.tts_voice_id,
            display_name=settings.tts_voice_name,
            language=settings.tts_language,
            model_path=settings.tts_model_path,
            config_path=settings.tts_config_path,
        )
        if language == "es"
        else PiperVoiceConfig(
            voice_id=settings.tts_french_voice_id,
            display_name=settings.tts_french_voice_name,
            language=settings.tts_french_language,
            model_path=settings.tts_french_model_path,
            config_path=settings.tts_french_config_path,
        )
    )
    piper = PiperTTSAdapter(piper_executable, (voice,))
    source_wav = await piper.synthesize(text, voice.voice_id)
    with wave.open(io.BytesIO(source_wav), "rb") as wav_file:
        if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
            raise RuntimeError("Piper diagnostic WAV is not mono PCM16")
        whisper_wav = pcm16_mono_to_wav(
            wav_file.readframes(wav_file.getnframes()),
            wav_file.getframerate(),
            settings.stt_sample_rate,
        )

    whisper = WhisperCppSTTAdapter(
        whisper_executable,
        settings.stt_model_path,
        threads=settings.stt_threads,
        process_timeout_seconds=settings.stt_process_timeout_seconds,
    )
    result = await whisper.transcribe(whisper_wav, language_hint=language)
    return {
        "ok": result.language == language,
        "expected_language": language,
        "detected_language": result.language,
        "transcript": result.text,
        "audio_duration_ms": result.duration_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check local Piper-to-whisper.cpp language recognition."
    )
    parser.add_argument("--language", choices=("es", "fr"), default="es")
    parser.add_argument("--text", help="Override the local diagnostic phrase.")
    args = parser.parse_args()
    payload = asyncio.run(
        run_check(args.language, args.text or SAMPLE_TEXTS[args.language])
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
