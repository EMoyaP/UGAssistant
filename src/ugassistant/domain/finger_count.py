from __future__ import annotations


class FingerCountStabilizer:
    def __init__(self, required_samples: int = 2) -> None:
        if required_samples < 1:
            raise ValueError("required_samples must be at least 1")
        self._required_samples = required_samples
        self._candidate: int | None = None
        self._candidate_is_set = False
        self._candidate_samples = 0
        self._stable: int | None = None

    @property
    def stable(self) -> int | None:
        return self._stable

    def update(self, value: int | None) -> int | None:
        if value is not None and not 0 <= value <= 10:
            raise ValueError("finger count must be between 0 and 10")

        if not self._candidate_is_set or value != self._candidate:
            self._candidate = value
            self._candidate_is_set = True
            self._candidate_samples = 1
        else:
            self._candidate_samples += 1

        if self._candidate_samples >= self._required_samples:
            self._stable = value
        return self._stable

    def reset(self) -> None:
        self._candidate = None
        self._candidate_is_set = False
        self._candidate_samples = 0
        self._stable = None
