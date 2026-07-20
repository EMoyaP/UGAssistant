from __future__ import annotations

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ugassistant.domain.audio_pcm import pcm16_mono_to_wav
from ugassistant.domain.ports import AudioAdapter, AudioDevice
from ugassistant.domain.preferences import (
    DevicePreference,
    match_device_preference,
    preference_for_device,
)
from ugassistant.domain.sound_activity import SoundActivityDetector


logger = logging.getLogger("ugassistant.audio")


class AudioCaptureBusyError(RuntimeError):
    pass


class AudioCaptureCancelledError(RuntimeError):
    pass


class NoSpeechDetectedError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioStatus:
    inputs: tuple[AudioDevice, ...] = ()
    outputs: tuple[AudioDevice, ...] = ()
    selected_input_index: int | None = None
    selected_output_index: int | None = None
    available: bool = False
    ready: bool = False
    monitoring: bool = False
    recording: bool = False
    output_enabled: bool = False
    output_playing: bool = False
    output_volume: float = 1.0
    output_balance: float = 0.0
    sound_detected: bool = False
    input_level: float = 0.0
    activation_threshold: float = 0.015
    detail: str = "not_scanned"

    def to_dict(self) -> dict[str, object]:
        return {
            "inputs": [device.to_dict() for device in self.inputs],
            "outputs": [device.to_dict() for device in self.outputs],
            "selected_input_index": self.selected_input_index,
            "selected_output_index": self.selected_output_index,
            "available": self.available,
            "ready": self.ready,
            "monitoring": self.monitoring,
            "recording": self.recording,
            "output_enabled": self.output_enabled,
            "output_playing": self.output_playing,
            "output_volume": round(self.output_volume, 2),
            "output_balance": round(self.output_balance, 3),
            "sound_detected": self.sound_detected,
            "input_level": round(self.input_level, 5),
            "activation_threshold": self.activation_threshold,
            "detail": self.detail,
        }


AudioStatusCallback = Callable[[AudioStatus], Awaitable[None]]


class AudioDeviceService:
    def __init__(
        self,
        adapter: AudioAdapter,
        *,
        activation_threshold: float = 0.015,
        release_threshold: float = 0.008,
        activation_samples: int = 2,
        silence_seconds: float = 0.8,
        block_duration_ms: int = 50,
        output_volume: float = 1.0,
        monitor_buffer_seconds: float = 1.0,
        on_status: AudioStatusCallback | None = None,
    ) -> None:
        self._adapter = adapter
        self._activation_threshold = activation_threshold
        self._release_threshold = release_threshold
        self._activation_samples = max(1, activation_samples)
        self._block_duration_ms = max(20, min(block_duration_ms, 200))
        self._monitor_pcm_chunks: deque[tuple[float, bytes]] = deque(
            maxlen=max(1, round(monitor_buffer_seconds * 1000 / self._block_duration_ms))
        )
        self._on_status = on_status
        self._activity_detector = SoundActivityDetector(
            activation_threshold=activation_threshold,
            release_threshold=release_threshold,
            activation_samples=activation_samples,
            silence_seconds=silence_seconds,
        )
        self._selected_input_index: int | None = None
        self._selected_output_index: int | None = None
        self._input_device_preference: DevicePreference | None = None
        self._output_device_preference: DevicePreference | None = None
        self._input_selection_initialized = False
        self._output_selection_initialized = False
        self._monitoring = False
        self._recording = False
        self._capture_sound_detected = False
        self._capture_cancel_event: asyncio.Event | None = None
        self._output_enabled = False
        self._output_playing = False
        self._output_volume = min(max(output_volume, 0.0), 1.0)
        self._output_balance = 0.0
        self._output_lock = asyncio.Lock()
        self._input_level = 0.0
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._level_task: asyncio.Task[None] | None = None
        self._pending_input_level: float | None = None
        self._last_level_publish_at = 0.0
        self._level_publish_interval_seconds = 0.1
        self._status = AudioStatus()

    @property
    def status(self) -> AudioStatus:
        return self._status

    def restore_device_preference(
        self,
        kind: str,
        preference: DevicePreference | None,
    ) -> None:
        if kind == "input":
            self._input_device_preference = preference
            self._selected_input_index = (
                preference.device_index if preference is not None else None
            )
            self._input_selection_initialized = True
            return
        if kind == "output":
            self._output_device_preference = preference
            self._selected_output_index = (
                preference.device_index if preference is not None else None
            )
            self._output_selection_initialized = True
            return
        raise ValueError(f"Unknown audio device kind: {kind}")

    def device_preference(self, kind: str) -> DevicePreference | None:
        if kind == "input":
            return self._input_device_preference
        if kind == "output":
            return self._output_device_preference
        raise ValueError(f"Unknown audio device kind: {kind}")

    async def refresh(self) -> AudioStatus:
        devices = await self._adapter.list_devices()
        inputs = tuple(device for device in devices if device.kind == "input")
        outputs = tuple(device for device in devices if device.kind == "output")

        (
            self._selected_input_index,
            self._input_device_preference,
            self._input_selection_initialized,
        ) = (
            self._resolve_selection(
                inputs,
                self._input_device_preference,
                self._input_selection_initialized,
            )
        )
        output_was_initialized = self._output_selection_initialized
        (
            self._selected_output_index,
            self._output_device_preference,
            self._output_selection_initialized,
        ) = (
            self._resolve_selection(
                outputs,
                self._output_device_preference,
                self._output_selection_initialized,
            )
        )
        if not output_was_initialized:
            self._output_enabled = self._selected_output_index is not None
        elif self._selected_output_index is None:
            self._output_enabled = False
        self._status = self._build_status(inputs, outputs)
        await self._publish()
        return self._status

    async def select_device(
        self,
        kind: str,
        device_index: int | None,
    ) -> AudioStatus:
        if kind not in {"input", "output"}:
            raise ValueError(f"Unknown audio device kind: {kind}")
        if self._recording:
            raise AudioCaptureBusyError("Cannot change devices while recording")

        resume_monitoring = kind == "input" and self._monitoring
        if resume_monitoring:
            await self.disable_monitoring()

        status = await self.refresh()
        candidates = status.inputs if kind == "input" else status.outputs
        if (
            device_index is not None
            and device_index not in {device.device_index for device in candidates}
        ):
            raise ValueError(f"Audio {kind} device {device_index} is not available")

        if kind == "input":
            self._selected_input_index = device_index
            selected_device = next(
                (
                    device
                    for device in candidates
                    if device.device_index == device_index
                ),
                None,
            )
            self._input_device_preference = (
                preference_for_device(selected_device)
                if selected_device is not None
                else None
            )
            self._input_selection_initialized = True
        else:
            self._selected_output_index = device_index
            selected_device = next(
                (
                    device
                    for device in candidates
                    if device.device_index == device_index
                ),
                None,
            )
            self._output_device_preference = (
                preference_for_device(selected_device)
                if selected_device is not None
                else None
            )
            self._output_selection_initialized = True
            if device_index is None:
                self._output_enabled = False
        self._status = self._build_status(status.inputs, status.outputs)
        await self._publish()
        if resume_monitoring and device_index is not None:
            return await self.enable_monitoring()
        return self._status

    async def enable_output(self) -> AudioStatus:
        status = await self._current_device_status()
        if status.selected_output_index is None:
            raise RuntimeError("No audio output device selected")
        self._output_enabled = True
        self._status = self._build_status(status.inputs, status.outputs)
        await self._publish()
        return self._status

    async def disable_output(self) -> AudioStatus:
        if self._output_playing:
            raise RuntimeError("Audio output is currently playing")
        self._output_enabled = False
        self._status = self._build_status(self._status.inputs, self._status.outputs)
        await self._publish()
        return self._status

    async def set_output_volume(self, volume: float) -> AudioStatus:
        if not 0.0 <= volume <= 1.0:
            raise ValueError("Output volume must be between 0 and 1")
        self._output_volume = volume
        self._status = self._build_status(self._status.inputs, self._status.outputs)
        await self._publish()
        return self._status

    async def play_wav(
        self,
        wav_bytes: bytes,
        *,
        balance: float = 0.0,
    ) -> AudioStatus:
        if not self._output_enabled:
            raise RuntimeError("Audio output is disabled")
        if not -1.0 <= balance <= 1.0:
            raise ValueError("Output balance must be between -1 and 1")
        device_index = self._selected_output_index
        if device_index is None:
            raise RuntimeError("No audio output device selected")
        async with self._output_lock:
            self._output_playing = True
            self._output_balance = balance
            self._status = self._build_status(
                self._status.inputs,
                self._status.outputs,
            )
            await self._publish()
            try:
                await self._adapter.play_wav(
                    device_index,
                    wav_bytes,
                    self._output_volume,
                    balance,
                )
            finally:
                self._output_playing = False
                self._status = self._build_status(
                    self._status.inputs,
                    self._status.outputs,
                )
                await self._publish()
        return self._status

    async def stop_output(self) -> AudioStatus:
        if self._output_playing:
            await self._adapter.stop_output()
        return self._status

    async def enable_monitoring(self) -> AudioStatus:
        if self._monitoring:
            return self._status
        if self._recording:
            raise AudioCaptureBusyError("A voice recording is active")
        status = await self._current_device_status()
        if status.selected_input_index is None:
            raise RuntimeError("No audio input device selected")
        selected_input = next(
            (
                device
                for device in status.inputs
                if device.device_index == status.selected_input_index
            ),
            None,
        )
        if selected_input is None:
            raise RuntimeError("Selected audio input is no longer available")

        sample_rate = round(selected_input.default_sample_rate or 48000.0)
        self._activity_detector.reset()
        self._input_level = 0.0
        self._event_loop = asyncio.get_running_loop()
        self._monitoring = True
        try:
            await self._adapter.start_input_monitor(
                selected_input.device_index,
                sample_rate,
                self._block_duration_ms,
                self._on_input_level,
                self._on_monitor_audio,
            )
        except Exception:
            self._monitoring = False
            self._event_loop = None
            raise
        self._status = self._build_status(status.inputs, status.outputs)
        await self._publish()
        return self._status

    async def disable_monitoring(self) -> AudioStatus:
        was_monitoring = self._monitoring
        self._monitoring = False
        self._event_loop = None
        self._activity_detector.reset()
        self._monitor_pcm_chunks.clear()
        self._pending_input_level = None
        level_task = self._level_task
        self._level_task = None
        if level_task is not None and not level_task.done():
            level_task.cancel()
            try:
                await level_task
            except asyncio.CancelledError:
                pass
        self._input_level = 0.0
        if was_monitoring:
            await self._adapter.stop_input_monitor()
        self._status = self._build_status(self._status.inputs, self._status.outputs)
        await self._publish()
        return self._status

    async def capture_utterance(
        self,
        *,
        target_sample_rate: int = 16000,
        wait_for_speech_seconds: float = 8.0,
        silence_seconds: float = 1.5,
        max_recording_seconds: float = 20.0,
        pre_roll_seconds: float = 0.3,
        initial_chunks: tuple[tuple[float, bytes], ...] = (),
        cancel_event: asyncio.Event | None = None,
    ) -> bytes:
        if self._recording:
            raise AudioCaptureBusyError("A voice recording is already active")
        if self._monitoring:
            raise AudioCaptureBusyError("Disable input monitoring before recording")
        if target_sample_rate <= 0:
            raise ValueError("Target sample rate must be positive")

        status = await self._current_device_status()
        selected_input = next(
            (
                device
                for device in status.inputs
                if device.device_index == status.selected_input_index
            ),
            None,
        )
        if selected_input is None:
            raise RuntimeError("No audio input device selected")

        source_sample_rate = round(selected_input.default_sample_rate or 48000.0)
        queue: asyncio.Queue[tuple[float, bytes]] = asyncio.Queue(maxsize=20)
        loop = asyncio.get_running_loop()
        cancel_event = cancel_event or asyncio.Event()
        self._capture_cancel_event = cancel_event
        self._recording = True
        self._capture_sound_detected = False
        self._input_level = 0.0
        stream_started = False

        def enqueue(level: float, pcm_bytes: bytes) -> None:
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait((level, pcm_bytes))

        def on_audio_chunk(level: float, pcm_bytes: bytes) -> None:
            if not loop.is_closed():
                loop.call_soon_threadsafe(enqueue, level, pcm_bytes)

        pre_roll_chunks: deque[bytes] = deque(
            maxlen=max(
                1,
                round(pre_roll_seconds * 1000 / self._block_duration_ms),
            )
        )
        captured_chunks: list[bytes] = []
        activation_count = 0
        silence_duration = 0.0
        speech_started_at: float | None = None
        waiting_started_at = loop.time()
        for level, pcm_bytes in initial_chunks:
            if pcm_bytes:
                enqueue(level, pcm_bytes)
        try:
            await self._adapter.start_input_capture(
                selected_input.device_index,
                source_sample_rate,
                self._block_duration_ms,
                on_audio_chunk,
            )
            stream_started = True
            self._status = self._build_status(status.inputs, status.outputs)
            await self._publish()

            while True:
                if cancel_event.is_set():
                    raise AudioCaptureCancelledError("Voice recognition cancelled")
                now = loop.time()
                if speech_started_at is None:
                    if now - waiting_started_at >= wait_for_speech_seconds:
                        raise NoSpeechDetectedError("No speech detected before timeout")
                elif now - speech_started_at >= max_recording_seconds:
                    break

                try:
                    level, pcm_bytes = await asyncio.wait_for(
                        queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    continue
                if not pcm_bytes:
                    continue

                self._input_level = min(max(float(level), 0.0), 1.0)
                self._capture_sound_detected = (
                    self._input_level >= self._activation_threshold
                )
                pre_roll_chunks.append(pcm_bytes)

                if speech_started_at is None:
                    if self._input_level >= self._activation_threshold:
                        activation_count += 1
                    else:
                        activation_count = 0
                    if activation_count >= self._activation_samples:
                        speech_started_at = loop.time()
                        captured_chunks.extend(pre_roll_chunks)
                else:
                    captured_chunks.append(pcm_bytes)
                    chunk_seconds = len(pcm_bytes) / (2 * source_sample_rate)
                    if self._input_level <= self._release_threshold:
                        silence_duration += chunk_seconds
                    else:
                        silence_duration = 0.0
                    if silence_duration >= silence_seconds:
                        break

                self._status = self._build_status(
                    self._status.inputs,
                    self._status.outputs,
                )
                await self._publish()
        finally:
            if stream_started:
                await self._adapter.stop_input_capture()
            self._recording = False
            self._capture_sound_detected = False
            self._capture_cancel_event = None
            self._input_level = 0.0
            self._status = self._build_status(
                self._status.inputs,
                self._status.outputs,
            )
            await self._publish()

        if not captured_chunks:
            raise NoSpeechDetectedError("No speech was captured")
        return pcm16_mono_to_wav(
            b"".join(captured_chunks),
            source_sample_rate,
            target_sample_rate,
        )

    def cancel_capture(self) -> None:
        if self._capture_cancel_event is not None:
            self._capture_cancel_event.set()

    def take_monitor_audio_buffer(self) -> tuple[tuple[float, bytes], ...]:
        """Return the recent in-memory monitor audio and immediately discard it."""
        buffered = tuple(self._monitor_pcm_chunks)
        self._monitor_pcm_chunks.clear()
        return buffered

    @property
    def has_monitor_audio_buffer(self) -> bool:
        return bool(self._monitor_pcm_chunks)

    async def shutdown(self) -> None:
        self.cancel_capture()
        await self.disable_monitoring()

    def _on_input_level(self, level: float) -> None:
        event_loop = self._event_loop
        if event_loop is None or event_loop.is_closed():
            return
        event_loop.call_soon_threadsafe(self._schedule_input_level, level)

    def _on_monitor_audio(self, level: float, pcm_bytes: bytes) -> None:
        event_loop = self._event_loop
        if event_loop is None or event_loop.is_closed() or not pcm_bytes:
            return
        event_loop.call_soon_threadsafe(
            self._append_monitor_audio,
            min(max(float(level), 0.0), 1.0),
            pcm_bytes,
        )

    def _append_monitor_audio(self, level: float, pcm_bytes: bytes) -> None:
        if self._monitoring:
            self._monitor_pcm_chunks.append((level, pcm_bytes))

    def _schedule_input_level(self, level: float) -> None:
        if not self._monitoring:
            return
        self._pending_input_level = min(max(float(level), 0.0), 1.0)
        if self._level_task is not None and not self._level_task.done():
            return
        self._level_task = asyncio.create_task(self._drain_input_levels())
        self._level_task.add_done_callback(self._log_level_task_error)

    async def _drain_input_levels(self) -> None:
        loop = asyncio.get_running_loop()
        while self._monitoring and self._pending_input_level is not None:
            level = self._pending_input_level
            self._pending_input_level = None
            was_active = self._activity_detector.active
            snapshot = self._activity_detector.update(level)
            self._input_level = snapshot.level
            self._status = self._build_status(self._status.inputs, self._status.outputs)
            now = loop.time()
            if (
                was_active != snapshot.active
                or now - self._last_level_publish_at >= self._level_publish_interval_seconds
            ):
                self._last_level_publish_at = now
                await self._publish()

    @staticmethod
    def _log_level_task_error(task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            logger.error(
                "audio_level_processing_failed",
                exc_info=(type(exception), exception, exception.__traceback__),
            )

    async def _publish(self) -> None:
        if self._on_status is not None:
            await self._on_status(self._status)

    async def _current_device_status(self) -> AudioStatus:
        """Use the last device scan until an explicit refresh or a stream failure."""
        if self._status.inputs or self._status.outputs:
            return self._status
        return await self.refresh()

    def _build_status(
        self,
        inputs: tuple[AudioDevice, ...],
        outputs: tuple[AudioDevice, ...],
    ) -> AudioStatus:
        if inputs and outputs:
            detail = "audio_inputs_and_outputs_detected"
        elif inputs:
            detail = "audio_outputs_missing"
        elif outputs:
            detail = "audio_inputs_missing"
        else:
            detail = "no_audio_devices_detected"
        if self._monitoring:
            detail = (
                "sound_detected"
                if self._activity_detector.active
                else "monitoring_input"
            )
        elif self._recording:
            detail = (
                "recording_speech"
                if self._capture_sound_detected
                else "recording_waiting_for_speech"
            )
        sound_detected = (
            self._capture_sound_detected
            if self._recording
            else self._activity_detector.active
        )
        return AudioStatus(
            inputs=inputs,
            outputs=outputs,
            selected_input_index=self._selected_input_index,
            selected_output_index=self._selected_output_index,
            available=bool(inputs or outputs),
            ready=bool(inputs and outputs),
            monitoring=self._monitoring,
            recording=self._recording,
            output_enabled=self._output_enabled,
            output_playing=self._output_playing,
            output_volume=self._output_volume,
            output_balance=self._output_balance,
            sound_detected=sound_detected,
            input_level=self._input_level,
            activation_threshold=self._activation_threshold,
            detail=detail,
        )

    @staticmethod
    def _resolve_selection(
        devices: tuple[AudioDevice, ...],
        preference: DevicePreference | None,
        initialized: bool,
    ) -> tuple[int | None, DevicePreference | None, bool]:
        if initialized:
            if preference is None:
                return None, None, True
            selected = match_device_preference(preference, devices)
            if selected is None:
                return None, preference, True
            return selected.device_index, preference_for_device(selected), True
        default = next((device for device in devices if device.is_default), None)
        selected = default or (devices[0] if devices else None)
        return (
            selected.device_index if selected is not None else None,
            preference_for_device(selected) if selected is not None else None,
            True,
        )
