from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


@dataclass(frozen=True)
class FaceLandmarks:
    right_eye: tuple[float, float]
    left_eye: tuple[float, float]
    nose_tip: tuple[float, float]
    right_mouth: tuple[float, float]
    left_mouth: tuple[float, float]

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "right_eye": list(self.right_eye),
            "left_eye": list(self.left_eye),
            "nose_tip": list(self.nose_tip),
            "right_mouth": list(self.right_mouth),
            "left_mouth": list(self.left_mouth),
        }


@dataclass(frozen=True)
class CameraPresence:
    available: bool
    person_detected: bool
    face_center_x: float | None = None
    face_center_y: float | None = None
    face_bbox: tuple[float, float, float, float] | None = None
    face_landmarks: FaceLandmarks | None = None
    detail: str | None = None


class HandGesture(str, Enum):
    NONE = "NONE"
    CLOSED_FIST = "CLOSED_FIST"
    OPEN_PALM = "OPEN_PALM"
    POINTING = "POINTING"
    THUMB_UP = "THUMB_UP"
    THUMB_DOWN = "THUMB_DOWN"
    VICTORY = "VICTORY"
    UNKNOWN = "UNKNOWN"


class CombinedGesture(str, Enum):
    BOTH_HANDS_OVER_EYES = "BOTH_HANDS_OVER_EYES"
    POINTING_AT_NOSE = "POINTING_AT_NOSE"
    POINTING_AT_MOUTH = "POINTING_AT_MOUTH"
    HAND_OVER_MOUTH = "HAND_OVER_MOUTH"
    OPEN_PALM_NEAR_FACE = "OPEN_PALM_NEAR_FACE"
    VICTORY_NEAR_FACE = "VICTORY_NEAR_FACE"
    THUMB_UP_NEAR_FACE = "THUMB_UP_NEAR_FACE"
    THUMB_DOWN_NEAR_FACE = "THUMB_DOWN_NEAR_FACE"


@dataclass(frozen=True)
class CombinedGestureDetection:
    gesture: CombinedGesture
    handedness: str
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return {
            "gesture": self.gesture.value,
            "handedness": self.handedness,
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class HandDetection:
    handedness: str
    gesture: HandGesture
    confidence: float
    landmarks: tuple[tuple[float, float, float], ...]
    bbox: tuple[float, float, float, float]
    finger_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "handedness": self.handedness,
            "gesture": self.gesture.value,
            "finger_count": self.finger_count,
            "confidence": round(self.confidence, 4),
            "landmarks": [list(point) for point in self.landmarks],
            "bbox": list(self.bbox),
        }


@dataclass(frozen=True)
class CameraFrame:
    jpeg_bytes: bytes
    width: int
    height: int
    presence: CameraPresence
    hands: tuple[HandDetection, ...] = ()
    combined_gestures: tuple[CombinedGestureDetection, ...] = ()
    finger_count: int | None = None


@dataclass(frozen=True)
class CameraDevice:
    device_index: int
    name: str

    def to_dict(self) -> dict[str, object]:
        return {"device_index": self.device_index, "name": self.name}


@dataclass(frozen=True)
class AudioDevice:
    name: str
    kind: str
    available: bool
    device_index: int = -1
    channels: int = 0
    default_sample_rate: float | None = None
    host_api: str | None = None
    is_default: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "device_index": self.device_index,
            "name": self.name,
            "kind": self.kind,
            "available": self.available,
            "channels": self.channels,
            "default_sample_rate": self.default_sample_rate,
            "host_api": self.host_api,
            "is_default": self.is_default,
        }


@dataclass(frozen=True)
class TTSVoice:
    voice_id: str
    display_name: str
    language: str
    available: bool
    detail: str = "ready"

    def to_dict(self) -> dict[str, object]:
        return {
            "voice_id": self.voice_id,
            "display_name": self.display_name,
            "language": self.language,
            "available": self.available,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class STTEngineStatus:
    available: bool
    detail: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str
    duration_ms: int


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMEngineStatus:
    available: bool
    model_available: bool
    detail: str


class LLMAdapter(Protocol):
    async def status(self) -> LLMEngineStatus:
        ...

    async def chat(
        self,
        messages: tuple[LLMMessage, ...],
        *,
        max_tokens: int,
        temperature: float,
        think: bool,
        context_tokens: int,
    ) -> str:
        ...


class STTAdapter(Protocol):
    async def status(self) -> STTEngineStatus:
        ...

    async def transcribe(
        self,
        wav_bytes: bytes,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        ...


class TTSAdapter(Protocol):
    async def list_voices(self) -> list[TTSVoice]:
        ...

    async def synthesize(
        self,
        text: str,
        voice_id: str | None = None,
        speech_rate: float = 1.0,
    ) -> bytes:
        ...


class CameraAdapter(Protocol):
    async def list_devices(self) -> list[CameraDevice]:
        ...

    async def select_device(self, device_index: int | None) -> None:
        ...

    async def open(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def read_frame(self) -> CameraFrame:
        ...

    async def read_presence(self) -> CameraPresence:
        ...

    async def set_hand_detection_enabled(self, enabled: bool) -> None:
        ...

    async def set_face_detection_enabled(self, enabled: bool) -> None:
        ...

    async def set_preview_enabled(self, enabled: bool) -> None:
        ...


class AudioAdapter(Protocol):
    async def list_devices(self) -> list[AudioDevice]:
        ...

    async def start_input_monitor(
        self,
        device_index: int,
        sample_rate: int,
        block_duration_ms: int,
        on_level: Callable[[float], None],
        on_audio_chunk: Callable[[float, bytes], None] | None = None,
    ) -> None:
        ...

    async def stop_input_monitor(self) -> None:
        ...

    async def start_input_capture(
        self,
        device_index: int,
        sample_rate: int,
        block_duration_ms: int,
        on_audio_chunk: Callable[[float, bytes], None],
    ) -> None:
        ...

    async def stop_input_capture(self) -> None:
        ...

    async def play_wav(
        self,
        device_index: int,
        wav_bytes: bytes,
        volume: float,
        balance: float = 0.0,
    ) -> None:
        ...

    async def stop_output(self) -> None:
        ...
