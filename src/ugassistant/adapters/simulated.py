from __future__ import annotations

from array import array
from collections.abc import Callable
import io
import wave

from ugassistant.domain.ports import (
    AudioDevice,
    CameraDevice,
    CameraFrame,
    CameraPresence,
    FaceLandmarks,
    LLMEngineStatus,
    LLMMessage,
    STTEngineStatus,
    TTSVoice,
    TranscriptionResult,
)


class SimulatedLLMAdapter:
    def __init__(self, response: str = "Respuesta simulada local.") -> None:
        self.response = response
        self.messages: list[tuple[LLMMessage, ...]] = []
        self.thinking_modes: list[bool] = []
        self.context_windows: list[int] = []

    async def status(self) -> LLMEngineStatus:
        return LLMEngineStatus(True, True, "ready")

    async def chat(
        self,
        messages: tuple[LLMMessage, ...],
        *,
        max_tokens: int,
        temperature: float,
        think: bool,
        context_tokens: int,
    ) -> str:
        del max_tokens, temperature
        self.messages.append(messages)
        self.thinking_modes.append(think)
        self.context_windows.append(context_tokens)
        return self.response

    async def generate(self, prompt: str) -> str:
        return f"Respuesta simulada a: {prompt[:80]}"


class SimulatedSTTAdapter:
    def __init__(
        self,
        text: str = "Hola desde la transcripcion simulada",
        language: str = "es",
        responses: list[tuple[str, str]] | None = None,
    ) -> None:
        self.text = text
        self.language = language
        self.transcribed_audio: list[bytes] = []
        self._responses = list(responses or [])

    async def status(self) -> STTEngineStatus:
        return STTEngineStatus(True, "ready")

    async def transcribe(
        self,
        wav_bytes: bytes,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        del language_hint
        self.transcribed_audio.append(wav_bytes)
        text, language = (
            self._responses.pop(0) if self._responses else (self.text, self.language)
        )
        return TranscriptionResult(
            text=text,
            language=language,
            duration_ms=1000,
        )


class SimulatedTTSAdapter:
    def __init__(self) -> None:
        self.synthesized: list[tuple[str, str]] = []
        self.speech_rates: list[float] = []

    async def list_voices(self) -> list[TTSVoice]:
        return [
            TTSVoice(
                voice_id="es_ES-davefx-medium",
                display_name="DaveFX",
                language="es_ES",
                available=True,
            ),
            TTSVoice(
                voice_id="fr_FR-tom-medium",
                display_name="Tom",
                language="fr_FR",
                available=True,
            ),
        ]

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        speech_rate: float = 1.0,
    ) -> bytes:
        selected_voice = voice_id or "es_ES-davefx-medium"
        self.synthesized.append((text, selected_voice))
        self.speech_rates.append(speech_rate)
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(array("h", [0] * 800 + [6000, -6000] * 800).tobytes())
        return output.getvalue()


class SimulatedCameraAdapter:
    def __init__(self, person_detected: bool = False) -> None:
        self._person_detected = person_detected
        self._opened = False
        self._device_index: int | None = 0
        self.hand_detection_enabled = True
        self.face_detection_enabled = True
        self.preview_enabled = False

    async def list_devices(self) -> list[CameraDevice]:
        return [CameraDevice(device_index=0, name="Simulated camera")]

    async def select_device(self, device_index: int | None) -> None:
        if self._opened:
            raise RuntimeError("Close the simulated camera before selecting a device")
        if device_index not in {None, 0}:
            raise ValueError(f"Unknown simulated camera: {device_index}")
        self._device_index = device_index

    async def open(self) -> None:
        if self._device_index is None:
            raise RuntimeError("No simulated camera selected")
        self._opened = True

    async def close(self) -> None:
        self._opened = False

    async def read_frame(self) -> CameraFrame:
        if not self._opened:
            raise RuntimeError("Simulated camera is closed")
        presence = await self.read_presence()
        return CameraFrame(
            jpeg_bytes=b"SIMULATED_JPEG",
            width=640,
            height=480,
            presence=presence,
        )

    async def read_presence(self) -> CameraPresence:
        face_bbox = (0.34, 0.2, 0.66, 0.7) if self._person_detected else None
        face_landmarks = (
            FaceLandmarks(
                right_eye=(0.43, 0.37),
                left_eye=(0.57, 0.37),
                nose_tip=(0.5, 0.47),
                right_mouth=(0.45, 0.58),
                left_mouth=(0.55, 0.58),
            )
            if self._person_detected
            else None
        )
        return CameraPresence(
            available=self._opened,
            person_detected=self._person_detected,
            face_center_x=0.5 if self._person_detected else None,
            face_center_y=0.45 if self._person_detected else None,
            face_bbox=face_bbox,
            face_landmarks=face_landmarks,
            detail="simulated",
        )

    async def set_hand_detection_enabled(self, enabled: bool) -> None:
        self.hand_detection_enabled = enabled

    async def set_face_detection_enabled(self, enabled: bool) -> None:
        self.face_detection_enabled = enabled

    async def set_preview_enabled(self, enabled: bool) -> None:
        self.preview_enabled = enabled


class SimulatedAudioAdapter:
    def __init__(self) -> None:
        self.monitoring = False
        self._on_level: Callable[[float], None] | None = None
        self._on_monitor_audio_chunk: Callable[[float, bytes], None] | None = None
        self._on_audio_chunk: Callable[[float, bytes], None] | None = None
        self.played_audio: list[tuple[int, bytes, float, float]] = []
        self.output_stop_count = 0

    async def list_devices(self) -> list[AudioDevice]:
        return [
            AudioDevice(
                device_index=0,
                name="Simulated microphone",
                kind="input",
                available=True,
                channels=1,
                default_sample_rate=16000.0,
                host_api="Simulated",
                is_default=True,
            ),
            AudioDevice(
                device_index=1,
                name="Simulated speakers",
                kind="output",
                available=True,
                channels=2,
                default_sample_rate=48000.0,
                host_api="Simulated",
                is_default=True,
            ),
        ]

    async def start_input_monitor(
        self,
        device_index: int,
        sample_rate: int,
        block_duration_ms: int,
        on_level: Callable[[float], None],
        on_audio_chunk: Callable[[float, bytes], None] | None = None,
    ) -> None:
        if device_index != 0:
            raise ValueError(f"Unknown simulated audio input: {device_index}")
        if sample_rate <= 0 or block_duration_ms <= 0:
            raise ValueError("Invalid simulated audio monitor settings")
        self.monitoring = True
        self._on_level = on_level
        self._on_monitor_audio_chunk = on_audio_chunk

    async def stop_input_monitor(self) -> None:
        self.monitoring = False
        self._on_level = None
        self._on_monitor_audio_chunk = None

    async def start_input_capture(
        self,
        device_index: int,
        sample_rate: int,
        block_duration_ms: int,
        on_audio_chunk: Callable[[float, bytes], None],
    ) -> None:
        if device_index != 0:
            raise ValueError(f"Unknown simulated audio input: {device_index}")
        if sample_rate <= 0 or block_duration_ms <= 0:
            raise ValueError("Invalid simulated audio capture settings")
        self.monitoring = True
        self._on_audio_chunk = on_audio_chunk

    async def stop_input_capture(self) -> None:
        self.monitoring = False
        self._on_audio_chunk = None

    async def play_wav(
        self,
        device_index: int,
        wav_bytes: bytes,
        volume: float,
        balance: float = 0.0,
    ) -> None:
        if device_index != 1:
            raise ValueError(f"Unknown simulated audio output: {device_index}")
        self.played_audio.append((device_index, wav_bytes, volume, balance))

    async def stop_output(self) -> None:
        self.output_stop_count += 1

    def emit_input_level(self, level: float) -> None:
        if self._on_level is not None:
            self._on_level(level)

    def emit_input_audio(self, level: float, pcm_bytes: bytes) -> None:
        if self._on_level is not None:
            self._on_level(level)
        if self._on_monitor_audio_chunk is not None:
            self._on_monitor_audio_chunk(level, pcm_bytes)
        if self._on_audio_chunk is not None:
            self._on_audio_chunk(level, pcm_bytes)
