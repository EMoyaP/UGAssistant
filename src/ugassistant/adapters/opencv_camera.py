from __future__ import annotations

import asyncio
import importlib
import logging
import platform
import threading
from pathlib import Path
from typing import Any

from ugassistant.adapters.hand_perception import OpenCVHandPerception
from ugassistant.domain.combined_gestures import detect_combined_gestures
from ugassistant.domain.finger_count import FingerCountStabilizer
from ugassistant.domain.ports import (
    CameraDevice,
    CameraFrame,
    CameraPresence,
    FaceLandmarks,
    HandDetection,
)


logger = logging.getLogger("ugassistant.camera")


class CameraUnavailableError(RuntimeError):
    pass


class CameraReadError(RuntimeError):
    pass


class OpenCVCameraAdapter:
    def __init__(
        self,
        model_path: Path,
        *,
        device_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 8.0,
        mirror: bool = True,
        score_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        cv2_module: Any | None = None,
        device_candidates: list[CameraDevice] | None = None,
        hand_perception: OpenCVHandPerception | None = None,
        hand_inference_interval_frames: int = 3,
        finger_count_stable_samples: int = 2,
    ) -> None:
        self._model_path = model_path
        self._device_index: int | None = device_index
        self._width = min(width, 640)
        self._height = min(height, 480)
        self._fps = min(max(fps, 1.0), 10.0)
        self._mirror = mirror
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._cv2 = cv2_module
        self._device_candidates = device_candidates
        self._hand_perception = hand_perception
        self._hand_inference_interval_frames = max(1, hand_inference_interval_frames)
        self._finger_count_stabilizer = FingerCountStabilizer(
            finger_count_stable_samples
        )
        self._capture: Any | None = None
        self._detector: Any | None = None
        self._frame_count = 0
        self._last_hands: tuple[HandDetection, ...] = ()
        self._stable_finger_count: int | None = None
        self._lock = threading.Lock()
        self._last_presence = CameraPresence(
            available=False,
            person_detected=False,
            detail="camera_closed",
        )

    async def open(self) -> None:
        await asyncio.to_thread(self._open_sync)

    async def list_devices(self) -> list[CameraDevice]:
        return await asyncio.to_thread(self._list_devices_sync)

    async def select_device(self, device_index: int | None) -> None:
        await asyncio.to_thread(self._select_device_sync, device_index)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    async def read_frame(self) -> CameraFrame:
        return await asyncio.to_thread(self._read_frame_sync)

    async def read_presence(self) -> CameraPresence:
        return self._last_presence

    def _load_cv2(self) -> Any:
        if self._cv2 is None:
            try:
                self._cv2 = importlib.import_module("cv2")
            except ImportError as exc:
                raise CameraUnavailableError(
                    "opencv-python-headless is not installed"
                ) from exc
        return self._cv2

    def _open_sync(self) -> None:
        with self._lock:
            if self._capture is not None and self._capture.isOpened():
                return
            if not self._model_path.is_file():
                raise CameraUnavailableError(
                    f"YuNet model not found: {self._model_path}"
                )
            if self._device_index is None:
                raise CameraUnavailableError("No camera device selected")

            cv2 = self._load_cv2()
            if self._hand_perception is not None:
                self._hand_perception.load()
            capture = self._create_capture(cv2, self._device_index)
            if not capture.isOpened():
                capture.release()
                raise CameraUnavailableError(
                    f"Camera device {self._device_index} could not be opened"
                )

            self._set_capture_property(cv2, capture, "CAP_PROP_FRAME_WIDTH", self._width)
            self._set_capture_property(cv2, capture, "CAP_PROP_FRAME_HEIGHT", self._height)
            self._set_capture_property(cv2, capture, "CAP_PROP_FPS", self._fps)
            self._set_capture_property(cv2, capture, "CAP_PROP_BUFFERSIZE", 1)

            detector = cv2.FaceDetectorYN.create(
                str(self._model_path),
                "",
                (320, 320),
                self._score_threshold,
                self._nms_threshold,
                5000,
            )
            self._capture = capture
            self._detector = detector
            self._last_presence = CameraPresence(
                available=True,
                person_detected=False,
                detail="camera_ready",
            )

    def _list_devices_sync(self) -> list[CameraDevice]:
        with self._lock:
            cv2 = self._load_cv2()
            if self._device_candidates is None:
                enumerated_devices = self._enumerated_camera_devices()
                if enumerated_devices:
                    return enumerated_devices

            devices: list[CameraDevice] = []
            current_is_open = (
                self._capture is not None and self._capture.isOpened()
            )
            for candidate in self._fallback_candidate_devices():
                device_index = candidate.device_index
                if current_is_open and device_index == self._device_index:
                    devices.append(candidate)
                    continue
                capture = self._create_capture(cv2, device_index)
                try:
                    if capture.isOpened():
                        devices.append(candidate)
                finally:
                    capture.release()
            return devices

    def _select_device_sync(self, device_index: int | None) -> None:
        with self._lock:
            if self._capture is not None and self._capture.isOpened():
                raise RuntimeError("Close the camera before selecting another device")
            self._device_index = device_index

    def _close_sync(self) -> None:
        with self._lock:
            if self._capture is not None:
                self._capture.release()
            self._capture = None
            self._detector = None
            self._frame_count = 0
            self._last_hands = ()
            self._finger_count_stabilizer.reset()
            self._stable_finger_count = None
            self._last_presence = CameraPresence(
                available=False,
                person_detected=False,
                detail="camera_closed",
            )

    def _read_frame_sync(self) -> CameraFrame:
        with self._lock:
            if self._capture is None or self._detector is None:
                raise CameraReadError("Camera is not open")

            ok, frame = self._capture.read()
            if not ok or frame is None:
                raise CameraReadError("Camera frame could not be read")

            cv2 = self._load_cv2()
            if self._mirror:
                frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            self._detector.setInputSize((width, height))
            _result, faces = self._detector.detect(frame)
            presence = self._presence_from_faces(faces, width, height)

            self._frame_count += 1
            if (
                self._hand_perception is not None
                and (self._frame_count - 1) % self._hand_inference_interval_frames == 0
            ):
                self._last_hands = self._hand_perception.detect(frame)
                raw_finger_count = (
                    sum(hand.finger_count for hand in self._last_hands)
                    if self._last_hands
                    else None
                )
                self._stable_finger_count = self._finger_count_stabilizer.update(
                    raw_finger_count
                )
            combined_gestures = detect_combined_gestures(
                presence,
                self._last_hands,
            )

            if presence.person_detected and faces is not None:
                face = max(faces, key=lambda row: float(row[2]) * float(row[3]))
                x, y, face_width, face_height = (int(value) for value in face[:4])
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + face_width, y + face_height),
                    (113, 216, 161),
                    2,
                )
                if presence.face_landmarks is not None:
                    for point in presence.face_landmarks.to_dict().values():
                        cv2.circle(
                            frame,
                            (int(point[0] * width), int(point[1] * height)),
                            4,
                            (91, 202, 240),
                            -1,
                        )
            if self._hand_perception is not None:
                self._hand_perception.draw(frame, self._last_hands)

            encode_options = [getattr(cv2, "IMWRITE_JPEG_QUALITY", 1), 78]
            encoded_ok, encoded = cv2.imencode(".jpg", frame, encode_options)
            if not encoded_ok:
                raise CameraReadError("Camera frame could not be encoded")

            self._last_presence = presence
            return CameraFrame(
                jpeg_bytes=encoded.tobytes(),
                width=width,
                height=height,
                presence=presence,
                hands=self._last_hands,
                combined_gestures=combined_gestures,
                finger_count=self._stable_finger_count,
            )

    @staticmethod
    def _preferred_backend(cv2: Any) -> int:
        system = platform.system()
        if system == "Windows":
            return int(getattr(cv2, "CAP_DSHOW", getattr(cv2, "CAP_ANY", 0)))
        if system == "Linux":
            return int(getattr(cv2, "CAP_V4L2", getattr(cv2, "CAP_ANY", 0)))
        return int(getattr(cv2, "CAP_ANY", 0))

    def _create_capture(self, cv2: Any, device_index: int) -> Any:
        backend = self._preferred_backend(cv2)
        return cv2.VideoCapture(device_index, backend)

    def _fallback_candidate_devices(self) -> list[CameraDevice]:
        if self._device_candidates is not None:
            return self._device_candidates
        if platform.system() == "Linux":
            devices: list[CameraDevice] = []
            for path in sorted(Path("/dev").glob("video*")):
                suffix = path.name.removeprefix("video")
                if suffix.isdigit():
                    devices.append(
                        CameraDevice(device_index=int(suffix), name=str(path))
                    )
            if devices:
                return devices
        return [
            CameraDevice(device_index=index, name=f"Camara {index}")
            for index in range(5)
        ]

    def _enumerated_camera_devices(self) -> list[CameraDevice]:
        try:
            module = importlib.import_module("cv2_enumerate_cameras")
            cv2 = self._load_cv2()
            backend = self._preferred_backend(cv2)
            camera_infos = module.enumerate_cameras(backend)
        except (ImportError, OSError, RuntimeError):
            logger.exception("camera_enumeration_failed")
            return []
        return [
            CameraDevice(device_index=int(info.index), name=str(info.name))
            for info in camera_infos
        ]

    @staticmethod
    def _set_capture_property(
        cv2: Any,
        capture: Any,
        property_name: str,
        value: float,
    ) -> None:
        property_id = getattr(cv2, property_name, None)
        if property_id is not None:
            capture.set(property_id, value)

    @staticmethod
    def _presence_from_faces(
        faces: Any | None,
        frame_width: int,
        frame_height: int,
    ) -> CameraPresence:
        if faces is None or len(faces) == 0:
            return CameraPresence(
                available=True,
                person_detected=False,
                detail="no_face",
            )

        face = max(faces, key=lambda row: float(row[2]) * float(row[3]))
        x = float(face[0])
        y = float(face[1])
        width = float(face[2])
        height = float(face[3])
        center_x = (x + width / 2) / frame_width
        center_y = (y + height / 2) / frame_height

        def normalized_point(offset: int) -> tuple[float, float]:
            return (
                min(max(float(face[offset]) / frame_width, 0.0), 1.0),
                min(max(float(face[offset + 1]) / frame_height, 0.0), 1.0),
            )

        return CameraPresence(
            available=True,
            person_detected=True,
            face_center_x=min(max(center_x, 0.0), 1.0),
            face_center_y=min(max(center_y, 0.0), 1.0),
            face_bbox=(
                min(max(x / frame_width, 0.0), 1.0),
                min(max(y / frame_height, 0.0), 1.0),
                min(max((x + width) / frame_width, 0.0), 1.0),
                min(max((y + height) / frame_height, 0.0), 1.0),
            ),
            face_landmarks=FaceLandmarks(
                right_eye=normalized_point(4),
                left_eye=normalized_point(6),
                nose_tip=normalized_point(8),
                right_mouth=normalized_point(10),
                left_mouth=normalized_point(12),
            ),
            detail="face_detected",
        )
