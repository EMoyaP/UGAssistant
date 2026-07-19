from __future__ import annotations

from dataclasses import dataclass
from time import time


@dataclass(frozen=True)
class TimerSnapshot:
    """A timer exposed to clients without retaining voice or audio data."""

    label: int
    duration_seconds: int
    ends_at_epoch_ms: int
    language: str = "es"

    @property
    def remaining_seconds(self) -> int:
        return max(0, (self.ends_at_epoch_ms - round(time() * 1000) + 999) // 1000)

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "duration_seconds": self.duration_seconds,
            "ends_at_epoch_ms": self.ends_at_epoch_ms,
            "remaining_seconds": self.remaining_seconds,
        }
