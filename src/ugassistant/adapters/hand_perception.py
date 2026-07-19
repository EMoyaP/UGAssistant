from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ugassistant.domain.ports import HandDetection, HandGesture


logger = logging.getLogger("ugassistant.hands")

# Pre/postprocessing follows the Apache-2.0 OpenCV Zoo MP-PalmDet and
# MP-HandPose reference implementations.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
)


class HandPerceptionUnavailableError(RuntimeError):
    pass


def _vector_angle(first: np.ndarray, second: np.ndarray) -> float:
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-6 or second_norm < 1e-6:
        return 180.0
    cosine = float(np.dot(first / first_norm, second / second_norm))
    return math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))


def finger_states(landmarks: Sequence[Sequence[float]]) -> tuple[bool, ...]:
    hand = np.asarray(landmarks, dtype=np.float32)
    if hand.shape[0] < 21 or hand.shape[1] < 2:
        raise ValueError("Expected 21 hand landmarks with x/y coordinates")
    hand = hand[:21, :2]

    angle_segments = (
        (0, 2, 3, 4),
        (0, 6, 7, 8),
        (0, 10, 11, 12),
        (0, 14, 15, 16),
        (0, 18, 19, 20),
    )
    distance_pairs = ((5, 4), (6, 8), (10, 12), (14, 16), (18, 20))
    wrist = hand[0]
    states: list[bool] = []

    for index, ((base_a, base_b, tip_a, tip_b), (near, tip)) in enumerate(
        zip(angle_segments, distance_pairs)
    ):
        angle = _vector_angle(
            hand[base_a] - hand[base_b],
            hand[tip_a] - hand[tip_b],
        )
        extends_from_wrist = (
            float(np.linalg.norm(hand[tip] - wrist))
            > float(np.linalg.norm(hand[near] - wrist))
        )
        threshold = 55.0 if index == 0 else 65.0
        states.append(angle < threshold and extends_from_wrist)

    return tuple(states)


def count_extended_fingers(landmarks: Sequence[Sequence[float]]) -> int:
    return sum(finger_states(landmarks))


def classify_hand_gesture(landmarks: Sequence[Sequence[float]]) -> HandGesture:
    states = finger_states(landmarks)
    thumb, index, middle, ring, pinky = states

    if not any(states):
        return HandGesture.CLOSED_FIST
    if all(states):
        return HandGesture.OPEN_PALM
    if not thumb and index and not middle and not ring and not pinky:
        return HandGesture.POINTING
    if not thumb and index and middle and not ring and not pinky:
        return HandGesture.VICTORY
    if thumb and not any((index, middle, ring, pinky)):
        hand = np.asarray(landmarks, dtype=np.float32)
        direction = hand[4, :2] - hand[0, :2]
        if abs(float(direction[1])) >= abs(float(direction[0])) * 0.8:
            return (
                HandGesture.THUMB_UP
                if float(direction[1]) < 0
                else HandGesture.THUMB_DOWN
            )
    return HandGesture.UNKNOWN


class OpenCVHandPerception:
    PALM_INPUT_SIZE = 192
    HAND_INPUT_SIZE = 224
    PALM_LANDMARK_IDS = (0, 5, 9, 13, 17, 1, 2)
    PALM_BOX_PRE_SHIFT = np.array((0.0, 0.0), dtype=np.float32)
    PALM_BOX_PRE_ENLARGE = 4.0
    PALM_BOX_SHIFT = np.array((0.0, -0.4), dtype=np.float32)
    PALM_BOX_ENLARGE = 3.0
    HAND_BOX_SHIFT = np.array((0.0, -0.1), dtype=np.float32)
    HAND_BOX_ENLARGE = 1.65

    def __init__(
        self,
        palm_model_path: Path,
        hand_pose_model_path: Path,
        *,
        cv2_module: Any | None = None,
        max_hands: int = 2,
        palm_score_threshold: float = 0.6,
        palm_nms_threshold: float = 0.3,
        hand_score_threshold: float = 0.8,
    ) -> None:
        self._cv2 = cv2_module
        self._palm_model_path = palm_model_path
        self._hand_pose_model_path = hand_pose_model_path
        self._max_hands = min(max(max_hands, 1), 2)
        self._palm_score_threshold = palm_score_threshold
        self._palm_nms_threshold = palm_nms_threshold
        self._hand_score_threshold = hand_score_threshold
        self._palm_net: Any | None = None
        self._hand_net: Any | None = None
        self._anchors = self._generate_anchors()

    @property
    def model_ready(self) -> bool:
        return self._palm_model_path.is_file() and self._hand_pose_model_path.is_file()

    def load(self) -> None:
        if self._palm_net is not None and self._hand_net is not None:
            return
        if not self.model_ready:
            raise HandPerceptionUnavailableError("Hand perception models are missing")

        if self._cv2 is None:
            try:
                self._cv2 = importlib.import_module("cv2")
            except ImportError as exc:
                raise HandPerceptionUnavailableError(
                    "opencv-python-headless is not installed"
                ) from exc

        self._palm_net = self._cv2.dnn.readNet(str(self._palm_model_path))
        self._hand_net = self._cv2.dnn.readNet(str(self._hand_pose_model_path))
        backend = getattr(self._cv2.dnn, "DNN_BACKEND_OPENCV", 3)
        target = getattr(self._cv2.dnn, "DNN_TARGET_CPU", 0)
        self._palm_net.setPreferableBackend(backend)
        self._palm_net.setPreferableTarget(target)
        self._hand_net.setPreferableBackend(backend)
        self._hand_net.setPreferableTarget(target)

    def detect(self, image: np.ndarray) -> tuple[HandDetection, ...]:
        self.load()
        palms = self._infer_palms(image)
        if len(palms) == 0:
            return ()

        palms = palms[np.argsort(-palms[:, -1])][0 : self._max_hands]
        detections: list[HandDetection] = []
        for palm in palms:
            try:
                raw_hand = self._infer_hand_pose(image, palm)
            except (ValueError, self._cv2.error):
                logger.exception("hand_pose_inference_failed")
                continue
            if raw_hand is not None:
                detections.append(self._to_detection(raw_hand, image.shape[1], image.shape[0]))
        return tuple(detections)

    def draw(self, image: np.ndarray, hands: tuple[HandDetection, ...]) -> None:
        height, width = image.shape[:2]
        for hand in hands:
            points = [
                (int(point[0] * width), int(point[1] * height))
                for point in hand.landmarks
            ]
            for start, end in HAND_CONNECTIONS:
                self._cv2.line(image, points[start], points[end], (234, 198, 91), 2)
            for point in points:
                self._cv2.circle(image, point, 3, (113, 216, 161), -1)

            x1, y1, x2, y2 = hand.bbox
            top_left = (int(x1 * width), int(y1 * height))
            bottom_right = (int(x2 * width), int(y2 * height))
            self._cv2.rectangle(image, top_left, bottom_right, (234, 198, 91), 2)
            label_y = max(top_left[1] - 8, 18)
            label = f"{hand.handedness} {hand.finger_count} {hand.gesture.value}"
            self._cv2.putText(
                image,
                label,
                (max(top_left[0], 0), label_y),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (234, 198, 91),
                1,
                self._cv2.LINE_AA,
            )

    @staticmethod
    def _generate_anchors() -> np.ndarray:
        anchors: list[tuple[float, float]] = []
        for y in range(24):
            for x in range(24):
                center = ((x + 0.5) / 24, (y + 0.5) / 24)
                anchors.extend((center, center))
        for y in range(12):
            for x in range(12):
                center = ((x + 0.5) / 12, (y + 0.5) / 12)
                anchors.extend((center,) * 6)
        return np.asarray(anchors, dtype=np.float32)

    def _infer_palms(self, image: np.ndarray) -> np.ndarray:
        assert self._palm_net is not None
        input_blob, pad_bias = self._preprocess_palm(image)
        self._palm_net.setInput(input_blob)
        output_names = self._palm_net.getUnconnectedOutLayersNames()
        regressors, classifiers = self._palm_net.forward(output_names)

        scores = classifiers[0, :, 0].astype(np.float64)
        scores = 1.0 / (1.0 + np.exp(-scores))
        box_delta = regressors[0, :, 0:4]
        landmark_delta = regressors[0, :, 4:]
        original_shape = np.array((image.shape[1], image.shape[0]), dtype=np.float32)
        scale = float(max(original_shape))

        center_delta = box_delta[:, :2] / self.PALM_INPUT_SIZE
        size_delta = box_delta[:, 2:] / self.PALM_INPUT_SIZE
        xy1 = (center_delta - size_delta / 2 + self._anchors) * scale
        xy2 = (center_delta + size_delta / 2 + self._anchors) * scale
        boxes = np.concatenate((xy1, xy2), axis=1)
        boxes -= np.array((pad_bias[0], pad_bias[1], pad_bias[0], pad_bias[1]))
        boxes_xywh = np.concatenate(
            (boxes[:, :2], boxes[:, 2:] - boxes[:, :2]),
            axis=1,
        )

        keep = self._cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            scores.tolist(),
            self._palm_score_threshold,
            self._palm_nms_threshold,
            top_k=5000,
        )
        if len(keep) == 0:
            return np.empty((0, 19), dtype=np.float32)
        indices = np.asarray(keep).reshape(-1)

        landmarks = landmark_delta[indices].reshape(-1, 7, 2)
        landmarks = landmarks / self.PALM_INPUT_SIZE
        landmarks += self._anchors[indices, np.newaxis, :]
        landmarks *= scale
        landmarks -= pad_bias
        return np.c_[
            boxes[indices],
            landmarks.reshape(-1, 14),
            scores[indices].reshape(-1, 1),
        ]

    def _preprocess_palm(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width = image.shape[:2]
        ratio = min(self.PALM_INPUT_SIZE / width, self.PALM_INPUT_SIZE / height)
        resized_width = max(1, int(width * ratio))
        resized_height = max(1, int(height * ratio))
        resized = self._cv2.resize(image, (resized_width, resized_height))
        pad_width = self.PALM_INPUT_SIZE - resized_width
        pad_height = self.PALM_INPUT_SIZE - resized_height
        left = pad_width // 2
        top = pad_height // 2
        padded = self._cv2.copyMakeBorder(
            resized,
            top,
            pad_height - top,
            left,
            pad_width - left,
            self._cv2.BORDER_CONSTANT,
            None,
            (0, 0, 0),
        )
        rgb = self._cv2.cvtColor(padded, self._cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        pad_bias = np.array((left / ratio, top / ratio), dtype=np.float32)
        return blob[np.newaxis, :, :, :], pad_bias

    def _infer_hand_pose(
        self,
        image: np.ndarray,
        palm: np.ndarray,
    ) -> np.ndarray | None:
        assert self._hand_net is not None
        blob, rotated_palm_bbox, angle, rotation_matrix, pad_bias = (
            self._preprocess_hand(image, palm)
        )
        self._hand_net.setInput(blob)
        output_names = self._hand_net.getUnconnectedOutLayersNames()
        landmarks, confidence, handedness, world_landmarks = self._hand_net.forward(
            output_names
        )
        confidence_value = float(confidence[0][0])
        if confidence_value < self._hand_score_threshold:
            return None

        landmarks = landmarks[0].reshape(-1, 3)
        world_landmarks = world_landmarks[0].reshape(-1, 3)
        box_size = rotated_palm_bbox[1] - rotated_palm_bbox[0]
        scale_factor = box_size / self.HAND_INPUT_SIZE
        landmarks[:, :2] = (
            landmarks[:, :2] - self.HAND_INPUT_SIZE / 2
        ) * max(scale_factor)
        landmarks[:, 2] *= max(scale_factor)

        coordinate_rotation = self._cv2.getRotationMatrix2D((0, 0), angle, 1.0)
        rotated_landmarks = np.dot(landmarks[:, :2], coordinate_rotation[:, :2])
        rotated_world = np.dot(world_landmarks[:, :2], coordinate_rotation[:, :2])
        rotated_world = np.c_[rotated_world, world_landmarks[:, 2]]

        rotation_component = np.array(
            (
                (rotation_matrix[0][0], rotation_matrix[1][0]),
                (rotation_matrix[0][1], rotation_matrix[1][1]),
            )
        )
        translation = np.array((rotation_matrix[0][2], rotation_matrix[1][2]))
        inverse_translation = np.array(
            (
                -np.dot(rotation_component[0], translation),
                -np.dot(rotation_component[1], translation),
            )
        )
        inverse_rotation = np.c_[rotation_component, inverse_translation]
        center = np.append(np.sum(rotated_palm_bbox, axis=0) / 2, 1)
        original_center = np.array(
            (np.dot(center, inverse_rotation[0]), np.dot(center, inverse_rotation[1]))
        )
        landmarks[:, :2] = rotated_landmarks + original_center + pad_bias

        bbox = np.array(
            (np.amin(landmarks[:, :2], axis=0), np.amax(landmarks[:, :2], axis=0))
        )
        box_size = bbox[1] - bbox[0]
        bbox += self.HAND_BOX_SHIFT * box_size
        center = np.sum(bbox, axis=0) / 2
        half_size = (bbox[1] - bbox[0]) * self.HAND_BOX_ENLARGE / 2
        bbox = np.array((center - half_size, center + half_size))

        return np.r_[
            bbox.reshape(-1),
            landmarks.reshape(-1),
            rotated_world.reshape(-1),
            handedness[0][0],
            confidence_value,
        ]

    def _preprocess_hand(
        self,
        image: np.ndarray,
        palm: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        palm_bbox = palm[0:4].reshape(2, 2)
        crop, palm_bbox, pad_bias = self._crop_and_pad(
            image, palm_bbox, for_rotation=True
        )
        crop = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2RGB)
        adjusted_bbox = palm_bbox - pad_bias
        palm_landmarks = palm[4:18].reshape(7, 2) - pad_bias

        palm_base = palm_landmarks[0]
        middle_base = palm_landmarks[2]
        radians = math.pi / 2 - math.atan2(
            -(middle_base[1] - palm_base[1]),
            middle_base[0] - palm_base[0],
        )
        radians -= 2 * math.pi * math.floor((radians + math.pi) / (2 * math.pi))
        angle = math.degrees(radians)
        center = np.sum(adjusted_bbox, axis=0) / 2
        rotation_matrix = self._cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = self._cv2.warpAffine(
            crop, rotation_matrix, (crop.shape[1], crop.shape[0])
        )

        homogeneous = np.c_[palm_landmarks, np.ones(palm_landmarks.shape[0])]
        rotated_landmarks = np.array(
            (
                np.dot(homogeneous, rotation_matrix[0]),
                np.dot(homogeneous, rotation_matrix[1]),
            )
        )
        rotated_bbox = np.array(
            (
                np.amin(rotated_landmarks, axis=1),
                np.amax(rotated_landmarks, axis=1),
            )
        )
        hand_crop, rotated_bbox, _ = self._crop_and_pad(
            rotated_image, rotated_bbox, for_rotation=False
        )
        blob = self._cv2.resize(
            hand_crop,
            (self.HAND_INPUT_SIZE, self.HAND_INPUT_SIZE),
            interpolation=self._cv2.INTER_AREA,
        ).astype(np.float32)
        blob /= 255.0
        return blob[np.newaxis, :, :, :], rotated_bbox, angle, rotation_matrix, pad_bias

    def _crop_and_pad(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        *,
        for_rotation: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = bbox[1] - bbox[0]
        shift = self.PALM_BOX_PRE_SHIFT if for_rotation else self.PALM_BOX_SHIFT
        enlarge = (
            self.PALM_BOX_PRE_ENLARGE if for_rotation else self.PALM_BOX_ENLARGE
        )
        bbox = bbox + shift * size
        center = np.sum(bbox, axis=0) / 2
        half_size = (bbox[1] - bbox[0]) * enlarge / 2
        bbox = np.array((center - half_size, center + half_size)).astype(np.int32)
        bbox[:, 0] = np.clip(bbox[:, 0], 0, image.shape[1])
        bbox[:, 1] = np.clip(bbox[:, 1], 0, image.shape[0])
        crop = image[bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0], :]
        if crop.size == 0:
            raise ValueError("Palm crop is empty")

        side = int(np.linalg.norm(crop.shape[:2])) if for_rotation else max(crop.shape[:2])
        pad_height = side - crop.shape[0]
        pad_width = side - crop.shape[1]
        left = pad_width // 2
        top = pad_height // 2
        padded = self._cv2.copyMakeBorder(
            crop,
            top,
            pad_height - top,
            left,
            pad_width - left,
            self._cv2.BORDER_CONSTANT,
            None,
            (0, 0, 0),
        )
        bias = bbox[0] - np.array((left, top))
        return padded, bbox, bias

    @staticmethod
    def _to_detection(raw_hand: np.ndarray, width: int, height: int) -> HandDetection:
        landmarks = raw_hand[4:67].reshape(21, 3)
        normalized_landmarks = tuple(
            (
                min(max(float(point[0]) / width, 0.0), 1.0),
                min(max(float(point[1]) / height, 0.0), 1.0),
                float(point[2]) / max(width, height),
            )
            for point in landmarks
        )
        bbox = raw_hand[:4]
        normalized_bbox = (
            min(max(float(bbox[0]) / width, 0.0), 1.0),
            min(max(float(bbox[1]) / height, 0.0), 1.0),
            min(max(float(bbox[2]) / width, 0.0), 1.0),
            min(max(float(bbox[3]) / height, 0.0), 1.0),
        )
        handedness = "Left" if float(raw_hand[130]) <= 0.5 else "Right"
        return HandDetection(
            handedness=handedness,
            gesture=classify_hand_gesture(landmarks),
            confidence=float(raw_hand[131]),
            landmarks=normalized_landmarks,
            bbox=normalized_bbox,
            finger_count=count_extended_fingers(landmarks),
        )
