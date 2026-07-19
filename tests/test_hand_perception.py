from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ugassistant.adapters.hand_perception import (
    HandPerceptionUnavailableError,
    OpenCVHandPerception,
    classify_hand_gesture,
    count_extended_fingers,
    finger_states,
)
from ugassistant.domain.ports import HandGesture


FINGER_POINTS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def make_hand(
    extended: set[str],
    *,
    thumb_direction: str = "up",
) -> np.ndarray:
    hand = np.zeros((21, 3), dtype=np.float32)
    x_positions = {"index": -1.2, "middle": -0.4, "ring": 0.4, "pinky": 1.2}

    for name, indices in FINGER_POINTS.items():
        x = x_positions[name]
        mcp, pip, dip, tip = indices
        hand[mcp, :2] = (x, -1.0)
        hand[pip, :2] = (x, -2.0)
        if name in extended:
            hand[dip, :2] = (x, -3.0)
            hand[tip, :2] = (x, -4.0)
        else:
            hand[dip, :2] = (x, -1.2)
            hand[tip, :2] = (x, -0.5)

    if "thumb" in extended:
        direction = -1.0 if thumb_direction == "up" else 1.0
        for index in range(1, 5):
            hand[index, :2] = (0.0, direction * index)
    else:
        hand[1, :2] = (0.5, -0.5)
        hand[2, :2] = (1.0, -0.8)
        hand[3, :2] = (0.8, -0.4)
        hand[4, :2] = (0.3, -0.2)
    return hand


class HandGestureTests(unittest.TestCase):
    def test_counts_from_zero_to_five_extended_fingers(self) -> None:
        ordered_fingers = ("thumb", "index", "middle", "ring", "pinky")

        for expected in range(6):
            with self.subTest(expected=expected):
                landmarks = make_hand(set(ordered_fingers[:expected]))
                self.assertEqual(count_extended_fingers(landmarks), expected)

    def test_classifies_supported_gestures(self) -> None:
        cases = (
            (make_hand(set()), HandGesture.CLOSED_FIST),
            (
                make_hand({"thumb", "index", "middle", "ring", "pinky"}),
                HandGesture.OPEN_PALM,
            ),
            (make_hand({"index"}), HandGesture.POINTING),
            (make_hand({"index", "middle"}), HandGesture.VICTORY),
            (make_hand({"thumb"}, thumb_direction="up"), HandGesture.THUMB_UP),
            (
                make_hand({"thumb"}, thumb_direction="down"),
                HandGesture.THUMB_DOWN,
            ),
        )

        for landmarks, expected in cases:
            with self.subTest(expected=expected.value):
                self.assertEqual(classify_hand_gesture(landmarks), expected)

    def test_finger_states_requires_21_landmarks(self) -> None:
        with self.assertRaises(ValueError):
            finger_states(np.zeros((20, 3), dtype=np.float32))

    def test_generates_the_locked_model_anchor_layout(self) -> None:
        anchors = OpenCVHandPerception._generate_anchors()

        self.assertEqual(anchors.shape, (2016, 2))
        np.testing.assert_allclose(anchors[0], (1 / 48, 1 / 48))
        np.testing.assert_allclose(anchors[-1], (23 / 24, 23 / 24))

    def test_missing_models_are_reported(self) -> None:
        perception = OpenCVHandPerception(
            Path("missing-palm.onnx"),
            Path("missing-hand.onnx"),
        )

        with self.assertRaises(HandPerceptionUnavailableError):
            perception.load()

    def test_palm_postprocessing_returns_a_selected_detection(self) -> None:
        regressors = np.zeros((1, 2016, 18), dtype=np.float32)
        classifiers = np.full((1, 2016, 1), -20.0, dtype=np.float32)
        classifiers[0, 600, 0] = 10.0
        regressors[0, 600, 2:4] = 50.0

        class FakePalmNet:
            def setInput(self, _blob: np.ndarray) -> None:
                return None

            def getUnconnectedOutLayersNames(self) -> tuple[str, str]:
                return ("regressors", "classifiers")

            def forward(self, _names: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
                return regressors, classifiers

        fake_dnn = SimpleNamespace(
            NMSBoxes=lambda *_args, **_kwargs: np.array([600], dtype=np.int32)
        )
        perception = OpenCVHandPerception(
            Path("unused-palm.onnx"),
            Path("unused-hand.onnx"),
            cv2_module=SimpleNamespace(dnn=fake_dnn),
        )
        perception._palm_net = FakePalmNet()
        perception._preprocess_palm = lambda _image: (  # type: ignore[method-assign]
            np.zeros((1, 192, 192, 3), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )

        palms = perception._infer_palms(np.zeros((480, 640, 3), dtype=np.uint8))

        self.assertEqual(palms.shape, (1, 19))
        self.assertGreater(float(palms[0, -1]), 0.99)


if __name__ == "__main__":
    unittest.main()
