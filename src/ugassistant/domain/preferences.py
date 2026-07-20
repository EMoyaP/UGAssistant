from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


class DeviceDescriptor(Protocol):
    device_index: int
    name: str


@dataclass(frozen=True)
class DevicePreference:
    name: str
    device_index: int | None = None
    backend: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "device_index": self.device_index,
            "backend": self.backend,
        }

    @classmethod
    def from_dict(cls, value: object) -> DevicePreference | None:
        if not isinstance(value, dict):
            return None
        name = str(value.get("name", "")).strip()
        if not name:
            return None
        raw_index = value.get("device_index")
        device_index = int(raw_index) if raw_index is not None else None
        raw_backend = value.get("backend")
        backend = str(raw_backend).strip() if raw_backend else None
        return cls(name=name, device_index=device_index, backend=backend)


@dataclass(frozen=True)
class UserPreferences:
    camera_device: DevicePreference | None = None
    camera_enabled: bool = False
    microphone_device: DevicePreference | None = None
    microphone_enabled: bool = False
    speaker_device: DevicePreference | None = None
    speaker_enabled: bool = True
    output_volume: float = 1.0
    voice_id: str = "es_ES-davefx-medium"
    language: str = "es_ES"
    speech_rate: float = 0.85
    spanish_wake_word: str = "hola"
    french_wake_word: str = "salut"
    home_assistant_url: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "camera": {
                "enabled": self.camera_enabled,
                "device": (
                    self.camera_device.to_dict()
                    if self.camera_device is not None
                    else None
                ),
            },
            "audio": {
                "microphone_enabled": self.microphone_enabled,
                "microphone_device": (
                    self.microphone_device.to_dict()
                    if self.microphone_device is not None
                    else None
                ),
                "speaker_enabled": self.speaker_enabled,
                "speaker_device": (
                    self.speaker_device.to_dict()
                    if self.speaker_device is not None
                    else None
                ),
                "output_volume": round(self.output_volume, 3),
            },
            "tts": {
                "voice_id": self.voice_id,
                "language": self.language,
                "speech_rate": round(self.speech_rate, 3),
            },
            "assistant": {
                "spanish_wake_word": self.spanish_wake_word,
                "french_wake_word": self.french_wake_word,
            },
            "iot": {
                "home_assistant_url": self.home_assistant_url,
            },
        }

    @classmethod
    def from_dict(cls, value: object) -> UserPreferences:
        if not isinstance(value, dict):
            raise ValueError("Preferences must be a YAML mapping")
        camera = _mapping(value.get("camera"))
        audio = _mapping(value.get("audio"))
        tts = _mapping(value.get("tts"))
        assistant = _mapping(value.get("assistant"))
        iot = _mapping(value.get("iot"))
        volume = min(max(float(audio.get("output_volume", 1.0)), 0.0), 1.0)
        voice_id = str(tts.get("voice_id", "es_ES-davefx-medium")).strip()
        language = str(tts.get("language", "es_ES")).strip()
        speech_rate = min(max(float(tts.get("speech_rate", 0.85)), 0.6), 1.3)
        return cls(
            camera_device=DevicePreference.from_dict(camera.get("device")),
            camera_enabled=bool(camera.get("enabled", False)),
            microphone_device=DevicePreference.from_dict(
                audio.get("microphone_device")
            ),
            microphone_enabled=bool(audio.get("microphone_enabled", False)),
            speaker_device=DevicePreference.from_dict(audio.get("speaker_device")),
            speaker_enabled=bool(audio.get("speaker_enabled", True)),
            output_volume=volume,
            voice_id=voice_id or "es_ES-davefx-medium",
            language=language or "es_ES",
            speech_rate=speech_rate,
            spanish_wake_word=_clean_text(
                assistant.get("spanish_wake_word", "hola"), "hola"
            ),
            french_wake_word=_clean_text(
                assistant.get("french_wake_word", "salut"), "salut"
            ),
            home_assistant_url=_clean_optional_text(
                iot.get("home_assistant_url"), maximum=256
            ),
        )


def preference_for_device(device: DeviceDescriptor) -> DevicePreference:
    backend = getattr(device, "host_api", None)
    return DevicePreference(
        name=device.name,
        device_index=device.device_index,
        backend=str(backend) if backend else None,
    )


def match_device_preference(
    preference: DevicePreference,
    devices: Sequence[DeviceDescriptor],
) -> DeviceDescriptor | None:
    normalized_name = _normalize(preference.name)
    name_matches = [
        device for device in devices if _normalize(device.name) == normalized_name
    ]
    if preference.backend:
        normalized_backend = _normalize(preference.backend)
        backend_matches = [
            device
            for device in name_matches
            if _normalize(str(getattr(device, "host_api", ""))) == normalized_backend
        ]
        if backend_matches:
            return backend_matches[0]
    if name_matches:
        return name_matches[0]
    if not preference.name and preference.device_index is not None:
        return next(
            (
                device
                for device in devices
                if device.device_index == preference.device_index
            ),
            None,
        )
    return None


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _clean_text(value: object, fallback: str, *, maximum: int = 40) -> str:
    cleaned = " ".join(str(value).split())[:maximum]
    return cleaned or fallback


def _clean_optional_text(value: object, *, maximum: int) -> str:
    if value is None:
        return ""
    return "".join(str(value).split())[:maximum]
