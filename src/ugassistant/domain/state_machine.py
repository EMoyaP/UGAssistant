from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class AssistantState(str, Enum):
    SLEEPING = "SLEEPING"
    IDLE = "IDLE"
    PERSON_DETECTED = "PERSON_DETECTED"
    LISTENING = "LISTENING"
    TRANSCRIBING = "TRANSCRIBING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"
    ERROR = "ERROR"


ALLOWED_TRANSITIONS: dict[AssistantState, set[AssistantState]] = {
    AssistantState.SLEEPING: {AssistantState.IDLE, AssistantState.ERROR},
    AssistantState.IDLE: {
        AssistantState.SLEEPING,
        AssistantState.PERSON_DETECTED,
        AssistantState.LISTENING,
        AssistantState.ERROR,
    },
    AssistantState.PERSON_DETECTED: {
        AssistantState.IDLE,
        AssistantState.LISTENING,
        AssistantState.ERROR,
    },
    AssistantState.LISTENING: {
        AssistantState.IDLE,
        AssistantState.PERSON_DETECTED,
        AssistantState.TRANSCRIBING,
        AssistantState.INTERRUPTED,
        AssistantState.ERROR,
    },
    AssistantState.TRANSCRIBING: {
        AssistantState.THINKING,
        AssistantState.LISTENING,
        AssistantState.ERROR,
    },
    AssistantState.THINKING: {
        AssistantState.SPEAKING,
        AssistantState.LISTENING,
        AssistantState.ERROR,
    },
    AssistantState.SPEAKING: {
        AssistantState.IDLE,
        AssistantState.INTERRUPTED,
        AssistantState.ERROR,
    },
    AssistantState.INTERRUPTED: {
        AssistantState.IDLE,
        AssistantState.LISTENING,
        AssistantState.ERROR,
    },
    AssistantState.ERROR: {
        AssistantState.IDLE,
        AssistantState.SLEEPING,
    },
}


@dataclass(frozen=True)
class StateSnapshot:
    state: AssistantState
    previous_state: AssistantState | None
    updated_at: datetime
    detail: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "state": self.state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "updated_at": self.updated_at.isoformat(),
            "detail": self.detail,
        }


class InvalidStateTransition(ValueError):
    pass


class AssistantStateMachine:
    def __init__(self, initial_state: AssistantState = AssistantState.IDLE) -> None:
        self._state = initial_state
        self._previous_state: AssistantState | None = None
        self._updated_at = self._now()
        self._detail: str | None = None

    @property
    def state(self) -> AssistantState:
        return self._state

    def snapshot(self) -> StateSnapshot:
        return StateSnapshot(
            state=self._state,
            previous_state=self._previous_state,
            updated_at=self._updated_at,
            detail=self._detail,
        )

    def transition_to(
        self,
        next_state: AssistantState,
        detail: str | None = None,
        *,
        force: bool = False,
    ) -> StateSnapshot:
        if next_state == self._state:
            self._detail = detail
            self._updated_at = self._now()
            return self.snapshot()

        allowed = ALLOWED_TRANSITIONS[self._state]
        if not force and next_state not in allowed:
            raise InvalidStateTransition(
                f"Cannot transition from {self._state.value} to {next_state.value}"
            )

        self._previous_state = self._state
        self._state = next_state
        self._detail = detail
        self._updated_at = self._now()
        return self.snapshot()

    def fail(self, detail: str) -> StateSnapshot:
        return self.transition_to(AssistantState.ERROR, detail=detail, force=True)

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)
