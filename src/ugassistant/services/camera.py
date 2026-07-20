from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from enum import Enum

from ugassistant.domain.ports import (
    CameraAdapter,
    CameraDevice,
    CameraFrame,
    CombinedGestureDetection,
    FaceLandmarks,
    HandDetection,
)
from ugassistant.domain.preferences import (
    DevicePreference,
    match_device_preference,
    preference_for_device,
)
from ugassistant.domain.state_machine import AssistantState


logger = logging.getLogger("ugassistant.camera")


@dataclass(frozen=True)
class CameraStatus:
    enabled: bool
    available: bool
    model_ready: bool
    hand_model_ready: bool
    person_detected: bool
    selected_device_index: int | None = 0
    face_center_x: float | None = None
    face_center_y: float | None = None
    face_bbox: tuple[float, float, float, float] | None = None
    face_landmarks: FaceLandmarks | None = None
    width: int | None = None
    height: int | None = None
    sequence: int = 0
    detail: str | None = None
    hands: tuple[HandDetection, ...] = ()
    combined_gestures: tuple[CombinedGestureDetection, ...] = ()
    finger_count: int | None = None
    perception_profile: str = "idle"

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "model_ready": self.model_ready,
            "hand_model_ready": self.hand_model_ready,
            "person_detected": self.person_detected,
            "selected_device_index": self.selected_device_index,
            "face_center_x": self.face_center_x,
            "face_center_y": self.face_center_y,
            "face_bbox": list(self.face_bbox) if self.face_bbox is not None else None,
            "face_landmarks": (
                self.face_landmarks.to_dict()
                if self.face_landmarks is not None
                else None
            ),
            "width": self.width,
            "height": self.height,
            "sequence": self.sequence,
            "detail": self.detail,
            "hands": [hand.to_dict() for hand in self.hands],
            "combined_gestures": [
                gesture.to_dict() for gesture in self.combined_gestures
            ],
            "finger_count": self.finger_count,
            "perception_profile": self.perception_profile,
        }


CameraStatusCallback = Callable[[CameraStatus], Awaitable[None]]


class CameraActivityProfile(str, Enum):
    IDLE = "idle"
    PERSON_DETECTED = "person_detected"
    PROCESSING = "processing"
    GESTURE = "gesture"
    DEBUG = "debug"


class CameraService:
    def __init__(
        self,
        adapter: CameraAdapter,
        *,
        target_fps: float = 8.0,
        idle_fps: float = 1.0,
        person_detected_fps: float = 2.0,
        processing_fps: float = 1.0,
        gesture_fps: float | None = None,
        model_ready: bool = False,
        hand_model_ready: bool = False,
        selected_device_index: int | None = 0,
        on_status: CameraStatusCallback | None = None,
    ) -> None:
        self._adapter = adapter
        self._profile_fps = {
            CameraActivityProfile.IDLE: self._bounded_fps(idle_fps),
            CameraActivityProfile.PERSON_DETECTED: self._bounded_fps(person_detected_fps),
            CameraActivityProfile.PROCESSING: self._bounded_fps(processing_fps),
            CameraActivityProfile.GESTURE: self._bounded_fps(
                target_fps if gesture_fps is None else gesture_fps
            ),
            CameraActivityProfile.DEBUG: self._bounded_fps(
                target_fps if gesture_fps is None else gesture_fps
            ),
        }
        self._activity_profile = CameraActivityProfile.IDLE
        self._effective_profile = CameraActivityProfile.IDLE
        self._profile_applied = False
        self._target_fps = self._profile_fps[self._effective_profile]
        self._preview_subscribers = 0
        self._profile_changed = asyncio.Event()
        self._model_ready = model_ready
        self._hand_model_ready = hand_model_ready
        self._selected_device_index = selected_device_index
        self._device_preference: DevicePreference | None = None
        self._on_status = on_status
        self._enabled = False
        self._task: asyncio.Task[None] | None = None
        self._latest_frame: CameraFrame | None = None
        self._condition = asyncio.Condition()
        self._sequence = 0
        self._status = CameraStatus(
            enabled=False,
            available=False,
            model_ready=model_ready,
            hand_model_ready=hand_model_ready,
            person_detected=False,
            selected_device_index=selected_device_index,
            detail="camera_disabled",
        )

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def perception_profile(self) -> CameraActivityProfile:
        return self._effective_profile

    @property
    def target_fps(self) -> float:
        return self._target_fps

    async def set_activity(
        self,
        state: AssistantState,
    ) -> None:
        if state in {
            AssistantState.LISTENING,
            AssistantState.SPEAKING,
        }:
            activity_profile = CameraActivityProfile.GESTURE
        elif state == AssistantState.PERSON_DETECTED:
            activity_profile = CameraActivityProfile.PERSON_DETECTED
        elif state in {AssistantState.TRANSCRIBING, AssistantState.THINKING}:
            activity_profile = CameraActivityProfile.PROCESSING
        else:
            activity_profile = CameraActivityProfile.IDLE
        self._activity_profile = activity_profile
        await self._apply_effective_profile()

    async def list_devices(self) -> list[CameraDevice]:
        return await self._adapter.list_devices()

    async def restore_device_preference(
        self,
        preference: DevicePreference | None,
    ) -> CameraStatus:
        if preference is None:
            return await self.select_device(None, enable=False)
        devices = await self.list_devices()
        selected = match_device_preference(preference, devices)
        if selected is None:
            status = await self.select_device(None, enable=False)
            self._device_preference = preference
            return status
        return await self.select_device(selected.device_index, enable=False)

    async def device_preference(self) -> DevicePreference | None:
        selected_index = self._selected_device_index
        if selected_index is None:
            return self._device_preference
        devices = await self.list_devices()
        selected = next(
            (
                device
                for device in devices
                if device.device_index == selected_index
            ),
            None,
        )
        if selected is not None:
            self._device_preference = preference_for_device(selected)
        return self._device_preference

    async def select_device(
        self,
        device_index: int | None,
        *,
        enable: bool = True,
    ) -> CameraStatus:
        selected_device: CameraDevice | None = None
        if device_index is not None:
            devices = await self.list_devices()
            selected_device = next(
                (
                    device
                    for device in devices
                    if device.device_index == device_index
                ),
                None,
            )
            if selected_device is None:
                raise ValueError(f"Camera device {device_index} is not available")
        if self._enabled or self._task is not None:
            await self.disable()
        await self._adapter.select_device(device_index)
        self._selected_device_index = device_index
        self._device_preference = (
            preference_for_device(selected_device)
            if selected_device is not None
            else None
        )
        self._status = CameraStatus(
            enabled=False,
            available=False,
            model_ready=self._model_ready,
            hand_model_ready=self._hand_model_ready,
            person_detected=False,
            selected_device_index=device_index,
            sequence=self._sequence,
            detail="camera_selected" if device_index is not None else "no_camera_selected",
        )
        await self._publish()
        if device_index is not None and enable:
            return await self.enable()
        return self._status

    async def enable(self) -> CameraStatus:
        if self._enabled:
            return self._status
        try:
            await self._apply_effective_profile()
            await self._adapter.open()
        except Exception as exc:
            self._status = CameraStatus(
                enabled=False,
                available=False,
                model_ready=self._model_ready,
                hand_model_ready=self._hand_model_ready,
                person_detected=False,
                selected_device_index=self._selected_device_index,
                sequence=self._sequence,
                detail=str(exc),
            )
            await self._publish()
            raise

        self._enabled = True
        self._status = CameraStatus(
            enabled=True,
            available=True,
            model_ready=self._model_ready,
            hand_model_ready=self._hand_model_ready,
            person_detected=False,
            selected_device_index=self._selected_device_index,
            sequence=self._sequence,
            detail="camera_started",
        )
        await self._publish()
        self._task = asyncio.create_task(self._capture_loop(), name="camera-capture")
        return self._status

    async def disable(self) -> CameraStatus:
        self._enabled = False
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        await self._adapter.close()
        self._latest_frame = None
        self._status = CameraStatus(
            enabled=False,
            available=False,
            model_ready=self._model_ready,
            hand_model_ready=self._hand_model_ready,
            person_detected=False,
            selected_device_index=self._selected_device_index,
            sequence=self._sequence,
            detail=(
                "camera_disabled"
                if self._selected_device_index is not None
                else "no_camera_selected"
            ),
        )
        async with self._condition:
            self._condition.notify_all()
        await self._publish()
        return self._status

    async def shutdown(self) -> None:
        if self._enabled or self._task is not None:
            await self.disable()

    async def mjpeg_stream(self) -> AsyncIterator[bytes]:
        self._preview_subscribers += 1
        await self._apply_effective_profile()
        sequence = self._sequence
        try:
            while self._enabled:
                frame, sequence = await self._wait_for_frame(sequence)
                if frame is None:
                    return
                if not frame.jpeg_bytes:
                    continue
                header = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(frame.jpeg_bytes)}\r\n\r\n".encode("ascii")
                )
                yield header + frame.jpeg_bytes + b"\r\n"
        finally:
            self._preview_subscribers = max(0, self._preview_subscribers - 1)
            await self._apply_effective_profile()

    async def _capture_loop(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            while self._enabled:
                started_at = loop.time()
                frame = await self._adapter.read_frame()
                self._latest_frame = frame
                self._sequence += 1
                presence = frame.presence
                self._status = CameraStatus(
                    enabled=True,
                    available=presence.available,
                    model_ready=self._model_ready,
                    hand_model_ready=self._hand_model_ready,
                    person_detected=presence.person_detected,
                    selected_device_index=self._selected_device_index,
                    face_center_x=presence.face_center_x,
                    face_center_y=presence.face_center_y,
                    face_bbox=presence.face_bbox,
                    face_landmarks=presence.face_landmarks,
                    width=frame.width,
                    height=frame.height,
                    sequence=self._sequence,
                    detail=presence.detail,
                    hands=frame.hands,
                    combined_gestures=frame.combined_gestures,
                    finger_count=frame.finger_count,
                    perception_profile=self._effective_profile.value,
                )
                async with self._condition:
                    self._condition.notify_all()
                await self._publish()
                interval = 1.0 / self._target_fps
                delay = interval - (loop.time() - started_at)
                if delay > 0:
                    try:
                        await asyncio.wait_for(self._profile_changed.wait(), delay)
                    except asyncio.TimeoutError:
                        pass
                    self._profile_changed.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._enabled = False
            logger.exception(
                json.dumps(
                    {"event": "camera_capture_failed", "detail": str(exc)},
                    ensure_ascii=True,
                )
            )
            self._status = CameraStatus(
                enabled=False,
                available=False,
                model_ready=self._model_ready,
                hand_model_ready=self._hand_model_ready,
                person_detected=False,
                selected_device_index=self._selected_device_index,
                sequence=self._sequence,
                detail=str(exc),
            )
            async with self._condition:
                self._condition.notify_all()
            await self._adapter.close()
            await self._publish()

    async def _wait_for_frame(
        self,
        after_sequence: int,
    ) -> tuple[CameraFrame | None, int]:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._sequence > after_sequence or not self._enabled
            )
            return self._latest_frame, self._sequence

    async def _publish(self) -> None:
        if self._on_status is None:
            return
        try:
            await self._on_status(self._status)
        except Exception:
            logger.exception(json.dumps({"event": "camera_status_callback_failed"}))

    async def _apply_effective_profile(self) -> None:
        effective_profile = (
            CameraActivityProfile.DEBUG
            if self._preview_subscribers
            else self._activity_profile
        )
        if effective_profile == self._effective_profile and self._profile_applied:
            return
        self._effective_profile = effective_profile
        self._target_fps = self._profile_fps[effective_profile]
        hands_enabled = effective_profile in {
            CameraActivityProfile.PERSON_DETECTED,
            CameraActivityProfile.GESTURE,
            CameraActivityProfile.DEBUG,
        }
        face_enabled = effective_profile != CameraActivityProfile.PROCESSING
        await self._adapter.set_hand_detection_enabled(hands_enabled)
        await self._adapter.set_face_detection_enabled(face_enabled)
        await self._adapter.set_preview_enabled(
            effective_profile == CameraActivityProfile.DEBUG
        )
        self._profile_applied = True
        self._status = replace(
            self._status,
            perception_profile=effective_profile.value,
        )
        self._profile_changed.set()
        await self._publish()

    @staticmethod
    def _bounded_fps(value: float) -> float:
        return min(max(value, 1.0), 10.0)
