from __future__ import annotations

import unittest

from ugassistant.domain.combined_gestures import detect_combined_gestures
from ugassistant.domain.ports import (
    CameraPresence,
    CombinedGesture,
    FaceLandmarks,
    HandDetection,
    HandGesture,
)


FACE = CameraPresence(
    available=True,
    person_detected=True,
    face_center_x=0.5,
    face_center_y=0.5,
    face_bbox=(0.3, 0.2, 0.7, 0.8),
    face_landmarks=FaceLandmarks(
        right_eye=(0.42, 0.38),
        left_eye=(0.58, 0.38),
        nose_tip=(0.5, 0.5),
        right_mouth=(0.44, 0.64),
        left_mouth=(0.56, 0.64),
    ),
    detail="face_detected",
)


def make_hand(
    gesture: HandGesture,
    center: tuple[float, float],
    *,
    handedness: str = "Right",
    index_tip: tuple[float, float] | None = None,
) -> HandDetection:
    points = [(center[0], center[1], 0.0) for _ in range(21)]
    if index_tip is not None:
        points[8] = (index_tip[0], index_tip[1], 0.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return HandDetection(
        handedness=handedness,
        gesture=gesture,
        confidence=0.92,
        landmarks=tuple(points),
        bbox=(min(xs), min(ys), max(xs), max(ys)),
    )


class CombinedGestureTests(unittest.TestCase):
    def test_detects_pointing_at_nose_and_mouth(self) -> None:
        nose = make_hand(HandGesture.POINTING, (0.15, 0.15), index_tip=(0.5, 0.5))
        mouth = make_hand(HandGesture.POINTING, (0.15, 0.15), index_tip=(0.5, 0.64))

        nose_result = detect_combined_gestures(FACE, (nose,))
        mouth_result = detect_combined_gestures(FACE, (mouth,))

        self.assertEqual(nose_result[0].gesture, CombinedGesture.POINTING_AT_NOSE)
        self.assertEqual(mouth_result[0].gesture, CombinedGesture.POINTING_AT_MOUTH)

    def test_detects_a_hand_covering_the_mouth(self) -> None:
        hand = make_hand(HandGesture.OPEN_PALM, (0.5, 0.64))

        result = detect_combined_gestures(FACE, (hand,))

        self.assertEqual(result[0].gesture, CombinedGesture.HAND_OVER_MOUTH)

    def test_detects_both_hands_covering_the_eyes(self) -> None:
        right = make_hand(HandGesture.OPEN_PALM, FACE.face_landmarks.right_eye)
        left = make_hand(
            HandGesture.UNKNOWN,
            FACE.face_landmarks.left_eye,
            handedness="Left",
        )

        result = detect_combined_gestures(FACE, (right, left))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].gesture, CombinedGesture.BOTH_HANDS_OVER_EYES)
        self.assertEqual(result[0].handedness, "Both")

    def test_requires_two_separated_hands_to_cover_the_eyes(self) -> None:
        one_hand = make_hand(HandGesture.OPEN_PALM, FACE.face_landmarks.right_eye)
        same_eye = make_hand(
            HandGesture.OPEN_PALM,
            (0.43, 0.38),
            handedness="Left",
        )

        one_result = detect_combined_gestures(FACE, (one_hand,))
        same_eye_result = detect_combined_gestures(FACE, (one_hand, same_eye))

        self.assertNotIn(
            CombinedGesture.BOTH_HANDS_OVER_EYES,
            [item.gesture for item in one_result],
        )
        self.assertNotIn(
            CombinedGesture.BOTH_HANDS_OVER_EYES,
            [item.gesture for item in same_eye_result],
        )

    def test_detects_supported_hand_signs_near_the_face(self) -> None:
        cases = (
            (HandGesture.OPEN_PALM, CombinedGesture.OPEN_PALM_NEAR_FACE),
            (HandGesture.VICTORY, CombinedGesture.VICTORY_NEAR_FACE),
            (HandGesture.THUMB_UP, CombinedGesture.THUMB_UP_NEAR_FACE),
            (HandGesture.THUMB_DOWN, CombinedGesture.THUMB_DOWN_NEAR_FACE),
        )

        for hand_gesture, expected in cases:
            with self.subTest(gesture=hand_gesture.value):
                hand = make_hand(hand_gesture, (0.75, 0.45))
                result = detect_combined_gestures(FACE, (hand,))
                self.assertEqual(result[0].gesture, expected)

    def test_uses_either_hand_and_preserves_handedness(self) -> None:
        right = make_hand(HandGesture.THUMB_UP, (0.75, 0.45), handedness="Right")
        left = make_hand(HandGesture.VICTORY, (0.25, 0.45), handedness="Left")

        result = detect_combined_gestures(FACE, (right, left))

        self.assertEqual([item.handedness for item in result], ["Right", "Left"])

    def test_requires_a_face_and_a_nearby_hand(self) -> None:
        no_face = CameraPresence(available=True, person_detected=False)
        far_hand = make_hand(HandGesture.VICTORY, (0.98, 0.5))

        self.assertEqual(detect_combined_gestures(no_face, (far_hand,)), ())
        self.assertEqual(detect_combined_gestures(FACE, (far_hand,)), ())


if __name__ == "__main__":
    unittest.main()
