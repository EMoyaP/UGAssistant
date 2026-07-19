from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class SoundActivitySnapshot:
    active: bool
    level: float
    changed: bool


class SoundActivityDetector:
    def __init__(
        self,
        *,
        activation_threshold: float = 0.015,
        release_threshold: float = 0.008,
        activation_samples: int = 2,
        silence_seconds: float = 0.8,
    ) -> None:
        if not 0.0 <= release_threshold <= activation_threshold <= 1.0:
            raise ValueError(
                "audio thresholds must satisfy 0 <= release <= activation <= 1"
            )
        if activation_samples < 1:
            raise ValueError("activation_samples must be at least 1")
        if silence_seconds < 0:
            raise ValueError("silence_seconds must not be negative")
        self._activation_threshold = activation_threshold
        self._release_threshold = release_threshold
        self._activation_samples = activation_samples
        self._silence_seconds = silence_seconds
        self._active = False
        self._candidate_samples = 0
        self._last_sound_at: float | None = None

    @property
    def active(self) -> bool:
        return self._active

    def update(
        self,
        level: float,
        *,
        now: float | None = None,
    ) -> SoundActivitySnapshot:
        timestamp = time.monotonic() if now is None else now
        normalized_level = min(max(float(level), 0.0), 1.0)
        changed = False

        if self._active:
            if normalized_level >= self._release_threshold:
                self._last_sound_at = timestamp
            elif (
                self._last_sound_at is not None
                and timestamp - self._last_sound_at >= self._silence_seconds
            ):
                self._active = False
                self._candidate_samples = 0
                self._last_sound_at = None
                changed = True
        else:
            if normalized_level >= self._activation_threshold:
                self._candidate_samples += 1
            else:
                self._candidate_samples = 0
            if self._candidate_samples >= self._activation_samples:
                self._active = True
                self._last_sound_at = timestamp
                changed = True

        return SoundActivitySnapshot(
            active=self._active,
            level=normalized_level,
            changed=changed,
        )

    def reset(self) -> None:
        self._active = False
        self._candidate_samples = 0
        self._last_sound_at = None
