from __future__ import annotations

import asyncio
import unittest

from ugassistant.adapters.simulated import SimulatedCameraAdapter
from ugassistant.services.camera import CameraService, CameraStatus


class CameraServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_one_shared_camera_frame(self) -> None:
        statuses: list[dict[str, object]] = []

        async def collect_status(status: CameraStatus) -> None:
            statuses.append(status.to_dict())

        service = CameraService(
            SimulatedCameraAdapter(person_detected=True),
            target_fps=10,
            model_ready=True,
            on_status=collect_status,
        )

        await service.enable()
        stream = service.mjpeg_stream()
        chunk = await asyncio.wait_for(anext(stream), timeout=1)
        await stream.aclose()
        status = await service.disable()

        self.assertIn(b"Content-Type: image/jpeg", chunk)
        self.assertIn(b"SIMULATED_JPEG", chunk)
        self.assertTrue(any(item["person_detected"] for item in statuses))
        self.assertFalse(status.enabled)

    async def test_enable_is_idempotent(self) -> None:
        service = CameraService(
            SimulatedCameraAdapter(),
            model_ready=True,
        )

        first = await service.enable()
        second = await service.enable()
        await service.disable()

        self.assertTrue(first.enabled)
        self.assertTrue(second.enabled)

    async def test_can_select_a_camera_or_none(self) -> None:
        service = CameraService(
            SimulatedCameraAdapter(),
            model_ready=True,
        )

        devices = await service.list_devices()
        none_status = await service.select_device(None)
        selected_status = await service.select_device(0)
        await service.disable()

        self.assertEqual(devices[0].name, "Simulated camera")
        self.assertIsNone(none_status.selected_device_index)
        self.assertEqual(selected_status.selected_device_index, 0)
        self.assertTrue(selected_status.enabled)

    def test_serializes_stable_finger_count(self) -> None:
        payload = CameraStatus(
            enabled=True,
            available=True,
            model_ready=True,
            hand_model_ready=True,
            person_detected=False,
            finger_count=0,
        ).to_dict()

        self.assertEqual(payload["finger_count"], 0)
