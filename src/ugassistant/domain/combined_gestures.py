from __future__ import annotations

import math

from ugassistant.domain.ports import (
    CameraPresence,
    CombinedGesture,
    CombinedGestureDetection,
    HandDetection,
    HandGesture,
)


Point = tuple[float, float]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _mean(points: tuple[Point, ...]) -> Point:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _hand_is_near_face(
    hand: HandDetection,
    face_bbox: tuple[float, float, float, float],
) -> bool:
    face_x1, face_y1, face_x2, face_y2 = face_bbox
    face_width = face_x2 - face_x1
    face_height = face_y2 - face_y1
    margin_x = face_width * 0.55
    margin_y = face_height * 0.42
    expanded = (
        face_x1 - margin_x,
        face_y1 - margin_y,
        face_x2 + margin_x,
        face_y2 + margin_y,
    )
    return any(
        expanded[0] <= point[0] <= expanded[2]
        and expanded[1] <= point[1] <= expanded[3]
        for point in hand.landmarks
    )


def _detection(
    gesture: CombinedGesture,
    hand: HandDetection,
) -> CombinedGestureDetection:
    return CombinedGestureDetection(
        gesture=gesture,
        handedness=hand.handedness,
        confidence=hand.confidence,
    )


def _palm_center(hand: HandDetection) -> Point:
    points = tuple((point[0], point[1]) for point in hand.landmarks)
    return _mean(tuple(points[index] for index in (0, 5, 9, 13, 17)))


def _detect_both_hands_over_eyes(
    hands: tuple[HandDetection, ...],
    right_eye: Point,
    left_eye: Point,
    face_width: float,
) -> CombinedGestureDetection | None:
    cover_poses = {
        HandGesture.OPEN_PALM,
        HandGesture.CLOSED_FIST,
        HandGesture.UNKNOWN,
    }
    candidates = [
        (hand, _palm_center(hand))
        for hand in hands
        if len(hand.landmarks) >= 21 and hand.gesture in cover_poses
    ]
    eye_threshold = face_width * 0.3
    minimum_separation = face_width * 0.22

    for first_index, (first_hand, first_center) in enumerate(candidates):
        for second_hand, second_center in candidates[first_index + 1 :]:
            if _distance(first_center, second_center) < minimum_separation:
                continue
            direct = max(
                _distance(first_center, right_eye),
                _distance(second_center, left_eye),
            )
            crossed = max(
                _distance(first_center, left_eye),
                _distance(second_center, right_eye),
            )
            if min(direct, crossed) <= eye_threshold:
                return CombinedGestureDetection(
                    gesture=CombinedGesture.BOTH_HANDS_OVER_EYES,
                    handedness="Both",
                    confidence=min(first_hand.confidence, second_hand.confidence),
                )
    return None


def detect_combined_gestures(
    presence: CameraPresence,
    hands: tuple[HandDetection, ...],
) -> tuple[CombinedGestureDetection, ...]:
    if (
        not presence.person_detected
        or presence.face_bbox is None
        or presence.face_landmarks is None
    ):
        return ()

    face_x1, face_y1, face_x2, face_y2 = presence.face_bbox
    face_width = face_x2 - face_x1
    face_height = face_y2 - face_y1
    if face_width <= 0 or face_height <= 0:
        return ()

    eyes_covered = _detect_both_hands_over_eyes(
        hands,
        presence.face_landmarks.right_eye,
        presence.face_landmarks.left_eye,
        face_width,
    )
    if eyes_covered is not None:
        return (eyes_covered,)

    nose = presence.face_landmarks.nose_tip
    mouth = _mean(
        (
            presence.face_landmarks.right_mouth,
            presence.face_landmarks.left_mouth,
        )
    )
    pointing_threshold = face_width * 0.22
    mouth_cover_threshold = face_width * 0.3
    detections: list[CombinedGestureDetection] = []

    for hand in hands:
        if len(hand.landmarks) < 21:
            continue
        points = tuple((point[0], point[1]) for point in hand.landmarks)
        index_tip = points[8]
        palm_center = _palm_center(hand)

        if hand.gesture == HandGesture.POINTING:
            nose_distance = _distance(index_tip, nose)
            mouth_distance = _distance(index_tip, mouth)
            if nose_distance <= pointing_threshold and nose_distance < mouth_distance:
                detections.append(_detection(CombinedGesture.POINTING_AT_NOSE, hand))
                continue
            if mouth_distance <= pointing_threshold:
                detections.append(_detection(CombinedGesture.POINTING_AT_MOUTH, hand))
                continue

        if (
            hand.gesture
            in {HandGesture.OPEN_PALM, HandGesture.CLOSED_FIST, HandGesture.UNKNOWN}
            and _distance(palm_center, mouth) <= mouth_cover_threshold
        ):
            detections.append(_detection(CombinedGesture.HAND_OVER_MOUTH, hand))
            continue

        if not _hand_is_near_face(hand, presence.face_bbox):
            continue

        near_face_gestures = {
            HandGesture.OPEN_PALM: CombinedGesture.OPEN_PALM_NEAR_FACE,
            HandGesture.VICTORY: CombinedGesture.VICTORY_NEAR_FACE,
            HandGesture.THUMB_UP: CombinedGesture.THUMB_UP_NEAR_FACE,
            HandGesture.THUMB_DOWN: CombinedGesture.THUMB_DOWN_NEAR_FACE,
        }
        combined = near_face_gestures.get(hand.gesture)
        if combined is not None:
            detections.append(_detection(combined, hand))

    return tuple(detections)
