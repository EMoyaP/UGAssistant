from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ugassistant.adapters.opencv_camera import (
    CameraUnavailableError,
    OpenCVCameraAdapter,
)
from ugassistant.domain.ports import CameraDevice, HandDetection, HandGesture


class FakeFrame:
    shape = (480, 640, 3)


class FakeEncodedFrame:
    def tobytes(self) -> bytes:
        return b"JPEG"


class FakeCapture:
    def __init__(self, _index: int, _backend: int | None = None) -> None:
        self.opened = True
        self.released = False
        self.properties: list[tuple[int, float]] = []

    def isOpened(self) -> bool:
        return self.opened

    def set(self, property_id: int, value: float) -> bool:
        self.properties.append((property_id, value))
        return True

    def read(self) -> tuple[bool, FakeFrame]:
        return True, FakeFrame()

    def release(self) -> None:
        self.released = True
        self.opened = False


class FakeDetector:
    def __init__(self) -> None:
        self.input_size = (320, 320)

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_size = size

    def detect(self, _frame: FakeFrame) -> tuple[int, list[list[float]]]:
        face = [
            100.0,
            80.0,
            200.0,
            240.0,
            160.0,
            160.0,
            240.0,
            160.0,
            200.0,
            200.0,
            170.0,
            250.0,
            230.0,
            250.0,
            0.95,
        ]
        return 1, [face]


class FakeHandPerception:
    def __init__(self, hands: tuple[HandDetection, ...] = ()) -> None:
        self.load_count = 0
        self.detect_count = 0
        self.draw_count = 0
        self.hands = hands

    def load(self) -> None:
        self.load_count += 1

    def detect(self, _frame: FakeFrame) -> tuple[HandDetection, ...]:
        self.detect_count += 1
        return self.hands

    def draw(self, _frame: FakeFrame, _hands: tuple[object, ...]) -> None:
        self.draw_count += 1


class FakeFaceDetectorFactory:
    @staticmethod
    def create(*_args: object) -> FakeDetector:
        return FakeDetector()


class FakeCV2:
    CAP_ANY = 0
    CAP_DSHOW = 700
    CAP_V4L2 = 200
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38
    IMWRITE_JPEG_QUALITY = 1
    FaceDetectorYN = FakeFaceDetectorFactory
    VideoCapture = FakeCapture

    @staticmethod
    def flip(frame: FakeFrame, _axis: int) -> FakeFrame:
        return frame

    @staticmethod
    def rectangle(*_args: object) -> None:
        return None

    @staticmethod
    def circle(*_args: object) -> None:
        return None

    @staticmethod
    def imencode(
        _extension: str,
        _frame: FakeFrame,
        _options: list[int],
    ) -> tuple[bool, FakeEncodedFrame]:
        return True, FakeEncodedFrame()


class OpenCVCameraAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_detects_largest_face_and_normalizes_center(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "yunet.onnx"
            model_path.write_bytes(b"model")
            adapter = OpenCVCameraAdapter(
                model_path,
                cv2_module=FakeCV2(),
                device_candidates=[CameraDevice(0, "Fake camera")],
            )

            await adapter.open()
            frame = await adapter.read_frame()
            await adapter.close()

        self.assertEqual(frame.jpeg_bytes, b"JPEG")
        self.assertTrue(frame.presence.person_detected)
        self.assertAlmostEqual(frame.presence.face_center_x or 0, 0.3125)
        self.assertAlmostEqual(frame.presence.face_center_y or 0, 5 / 12)
        self.assertEqual(frame.presence.face_bbox, (0.15625, 1 / 6, 0.46875, 2 / 3))
        self.assertIsNotNone(frame.presence.face_landmarks)
        self.assertEqual(
            frame.presence.face_landmarks.nose_tip,  # type: ignore[union-attr]
            (0.3125, 5 / 12),
        )

    async def test_missing_model_is_reported(self) -> None:
        adapter = OpenCVCameraAdapter(
            Path("missing-yunet.onnx"),
            cv2_module=FakeCV2(),
            device_candidates=[CameraDevice(0, "Fake camera")],
        )

        with self.assertRaises(CameraUnavailableError):
            await adapter.open()

    async def test_lists_and_clears_camera_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "yunet.onnx"
            model_path.write_bytes(b"model")
            adapter = OpenCVCameraAdapter(
                model_path,
                cv2_module=FakeCV2(),
                device_candidates=[CameraDevice(0, "Fake camera")],
            )

            devices = await adapter.list_devices()
            await adapter.select_device(None)

            with self.assertRaises(CameraUnavailableError):
                await adapter.open()

        self.assertEqual(devices[0].device_index, 0)
        self.assertGreaterEqual(len(devices), 1)

    async def test_preserves_backend_camera_name_and_index_mapping(self) -> None:
        camera_module = SimpleNamespace(
            enumerate_cameras=lambda _backend: [
                SimpleNamespace(index=0, name="Integrated Camera"),
                SimpleNamespace(index=1, name="USB Live camera"),
            ]
        )
        adapter = OpenCVCameraAdapter(
            Path("unused-yunet.onnx"),
            cv2_module=FakeCV2(),
        )

        with (
            patch(
                "ugassistant.adapters.opencv_camera.importlib.import_module",
                return_value=camera_module,
            ),
            patch(
                "ugassistant.adapters.opencv_camera.platform.system",
                return_value="Windows",
            ),
        ):
            devices = await adapter.list_devices()

        self.assertEqual(
            [(device.device_index, device.name) for device in devices],
            [(0, "Integrated Camera"), (1, "USB Live camera")],
        )

    async def test_runs_hand_inference_at_a_lower_frame_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "yunet.onnx"
            model_path.write_bytes(b"model")
            hand_perception = FakeHandPerception()
            adapter = OpenCVCameraAdapter(
                model_path,
                cv2_module=FakeCV2(),
                device_candidates=[CameraDevice(0, "Fake camera")],
                hand_perception=hand_perception,  # type: ignore[arg-type]
                hand_inference_interval_frames=3,
            )

            await adapter.open()
            for _ in range(4):
                await adapter.read_frame()
            await adapter.close()

        self.assertEqual(hand_perception.load_count, 1)
        self.assertEqual(hand_perception.detect_count, 2)
        self.assertEqual(hand_perception.draw_count, 4)

    async def test_stabilizes_total_only_on_fresh_hand_inferences(self) -> None:
        landmarks = tuple((0.9, 0.9, 0.0) for _ in range(21))
        hands = (
            HandDetection(
                handedness="Left",
                gesture=HandGesture.OPEN_PALM,
                confidence=0.9,
                landmarks=landmarks,
                bbox=(0.7, 0.7, 1.0, 1.0),
                finger_count=5,
            ),
            HandDetection(
                handedness="Right",
                gesture=HandGesture.VICTORY,
                confidence=0.9,
                landmarks=landmarks,
                bbox=(0.7, 0.7, 1.0, 1.0),
                finger_count=2,
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "yunet.onnx"
            model_path.write_bytes(b"model")
            adapter = OpenCVCameraAdapter(
                model_path,
                cv2_module=FakeCV2(),
                device_candidates=[CameraDevice(0, "Fake camera")],
                hand_perception=FakeHandPerception(hands),  # type: ignore[arg-type]
                hand_inference_interval_frames=3,
                finger_count_stable_samples=2,
            )

            await adapter.open()
            counts = [(await adapter.read_frame()).finger_count for _ in range(4)]
            await adapter.close()

        self.assertEqual(counts, [None, None, None, 7])


if __name__ == "__main__":
    unittest.main()
