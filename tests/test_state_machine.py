from __future__ import annotations

import unittest

from ugassistant.domain.state_machine import (
    AssistantState,
    AssistantStateMachine,
    InvalidStateTransition,
)


class AssistantStateMachineTests(unittest.TestCase):
    def test_initial_state_is_idle(self) -> None:
        machine = AssistantStateMachine()

        self.assertEqual(machine.state, AssistantState.IDLE)
        self.assertEqual(machine.snapshot().to_dict()["state"], "IDLE")

    def test_valid_conversation_path(self) -> None:
        machine = AssistantStateMachine()

        machine.transition_to(AssistantState.PERSON_DETECTED)
        machine.transition_to(AssistantState.LISTENING)
        machine.transition_to(AssistantState.TRANSCRIBING)
        machine.transition_to(AssistantState.THINKING)
        machine.transition_to(AssistantState.SPEAKING)
        snapshot = machine.transition_to(AssistantState.IDLE)

        self.assertEqual(snapshot.state, AssistantState.IDLE)
        self.assertEqual(snapshot.previous_state, AssistantState.SPEAKING)

    def test_invalid_transition_raises(self) -> None:
        machine = AssistantStateMachine()

        with self.assertRaises(InvalidStateTransition):
            machine.transition_to(AssistantState.SPEAKING)

    def test_error_can_be_forced_from_any_state(self) -> None:
        machine = AssistantStateMachine()

        snapshot = machine.fail("device disconnected")

        self.assertEqual(snapshot.state, AssistantState.ERROR)
        self.assertEqual(snapshot.detail, "device disconnected")

    def test_idle_sleep_and_wake_cycle(self) -> None:
        machine = AssistantStateMachine()

        machine.transition_to(AssistantState.SLEEPING, detail="idle-timeout")
        snapshot = machine.transition_to(AssistantState.IDLE, detail="automatic-wake")

        self.assertEqual(snapshot.state, AssistantState.IDLE)
        self.assertEqual(snapshot.previous_state, AssistantState.SLEEPING)

    def test_listening_can_return_to_a_presence_state_after_silence(self) -> None:
        machine = AssistantStateMachine()
        machine.transition_to(AssistantState.LISTENING, detail="audio-activity")

        snapshot = machine.transition_to(
            AssistantState.PERSON_DETECTED,
            detail="audio-silence",
        )

        self.assertEqual(snapshot.state, AssistantState.PERSON_DETECTED)


if __name__ == "__main__":
    unittest.main()
