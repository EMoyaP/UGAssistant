from __future__ import annotations

import asyncio
from array import array
import io
import importlib
import logging
import math
import platform
import sys
import threading
import wave
from collections.abc import Mapping, Sequence
from collections.abc import Callable
from typing import Any

from ugassistant.domain.ports import AudioDevice


logger = logging.getLogger("ugassistant.audio.portaudio")


class AudioEnumerationError(RuntimeError):
    pass


class PortAudioAdapter:
    def __init__(
        self,
        *,
        sounddevice_module: Any | None = None,
        system_name: str | None = None,
    ) -> None:
        self._sounddevice = sounddevice_module
        self._system_name = system_name
        self._input_stream: Any | None = None
        self._stream_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._output_stream: Any | None = None
        self._output_cancel_event: threading.Event | None = None

    async def list_devices(self) -> list[AudioDevice]:
        return await asyncio.to_thread(self._list_devices_sync)

    async def start_input_monitor(
        self,
        device_index: int,
        sample_rate: int,
        block_duration_ms: int,
        on_level: Callable[[float], None],
        on_audio_chunk: Callable[[float, bytes], None] | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._start_input_monitor_sync,
            device_index,
            sample_rate,
            block_duration_ms,
            on_level,
            on_audio_chunk,
        )

    async def stop_input_monitor(self) -> None:
        await asyncio.to_thread(self._stop_input_monitor_sync)

    async def start_input_capture(
        self,
        device_index: int,
        sample_rate: int,
        block_duration_ms: int,
        on_audio_chunk: Callable[[float, bytes], None],
    ) -> None:
        await asyncio.to_thread(
            self._start_input_capture_sync,
            device_index,
            sample_rate,
            block_duration_ms,
            on_audio_chunk,
        )

    async def stop_input_capture(self) -> None:
        await asyncio.to_thread(self._stop_input_monitor_sync)

    async def play_wav(
        self,
        device_index: int,
        wav_bytes: bytes,
        volume: float,
        balance: float = 0.0,
    ) -> None:
        await asyncio.to_thread(
            self._play_wav_sync,
            device_index,
            wav_bytes,
            volume,
            balance,
        )

    async def stop_output(self) -> None:
        await asyncio.to_thread(self._stop_output_sync)

    def _load_sounddevice(self) -> Any:
        if self._sounddevice is None:
            try:
                self._sounddevice = importlib.import_module("sounddevice")
            except (ImportError, OSError) as exc:
                raise AudioEnumerationError(
                    "sounddevice or the PortAudio runtime is not available"
                ) from exc
        return self._sounddevice

    def _list_devices_sync(self) -> list[AudioDevice]:
        sounddevice = self._load_sounddevice()
        try:
            raw_devices = list(sounddevice.query_devices())
            host_apis = list(sounddevice.query_hostapis())
        except Exception as exc:
            raise AudioEnumerationError(
                f"PortAudio device enumeration failed: {exc}"
            ) from exc

        devices: list[AudioDevice] = []
        for kind, channel_key, default_key in (
            ("input", "max_input_channels", "default_input_device"),
            ("output", "max_output_channels", "default_output_device"),
        ):
            host_api_index = self._preferred_host_api_index(
                raw_devices,
                host_apis,
                channel_key,
            )
            if host_api_index is None:
                continue

            host_api = host_apis[host_api_index]
            host_api_name = str(self._value(host_api, "name", "PortAudio"))
            default_index = int(self._value(host_api, default_key, -1))
            for device_index, raw_device in enumerate(raw_devices):
                channels = int(self._value(raw_device, channel_key, 0))
                device_host_api = int(self._value(raw_device, "hostapi", -1))
                if channels <= 0 or device_host_api != host_api_index:
                    continue

                sample_rate_value = self._value(
                    raw_device,
                    "default_samplerate",
                    None,
                )
                sample_rate = (
                    float(sample_rate_value)
                    if sample_rate_value is not None
                    else None
                )
                name = " ".join(str(self._value(raw_device, "name", "Audio")).split())
                devices.append(
                    AudioDevice(
                        device_index=device_index,
                        name=name,
                        kind=kind,
                        available=True,
                        channels=channels,
                        default_sample_rate=sample_rate,
                        host_api=host_api_name,
                        is_default=device_index == default_index,
                    )
                )

        return sorted(
            devices,
            key=lambda device: (
                device.kind,
                not device.is_default,
                device.name.casefold(),
                device.device_index,
            ),
        )

    def _start_input_monitor_sync(
        self,
        device_index: int,
        sample_rate: int,
        block_duration_ms: int,
        on_level: Callable[[float], None],
        on_audio_chunk: Callable[[float, bytes], None] | None,
    ) -> None:
        self._start_input_stream_sync(
            device_index,
            sample_rate,
            block_duration_ms,
            lambda level, pcm_bytes: self._notify_monitor_callbacks(
                on_level,
                on_audio_chunk,
                level,
                pcm_bytes,
            ),
        )

    @staticmethod
    def _notify_monitor_callbacks(
        on_level: Callable[[float], None],
        on_audio_chunk: Callable[[float, bytes], None] | None,
        level: float,
        pcm_bytes: bytes,
    ) -> None:
        on_level(level)
        if on_audio_chunk is not None:
            on_audio_chunk(level, pcm_bytes)

    def _start_input_capture_sync(
        self,
        device_index: int,
        sample_rate: int,
        block_duration_ms: int,
        on_audio_chunk: Callable[[float, bytes], None],
    ) -> None:
        self._start_input_stream_sync(
            device_index,
            sample_rate,
            block_duration_ms,
            on_audio_chunk,
        )

    def _start_input_stream_sync(
        self,
        device_index: int,
        sample_rate: int,
        block_duration_ms: int,
        on_audio_chunk: Callable[[float, bytes], None],
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if block_duration_ms <= 0:
            raise ValueError("block_duration_ms must be positive")

        sounddevice = self._load_sounddevice()
        block_size = max(128, round(sample_rate * block_duration_ms / 1000))

        def audio_callback(
            input_data: Any,
            _frames: int,
            _time_info: Any,
            status: Any,
        ) -> None:
            if status:
                logger.warning("portaudio_input_status: %s", status)
            try:
                pcm_bytes = bytes(input_data)
                samples = memoryview(pcm_bytes).cast("h")
                if not samples:
                    on_audio_chunk(0.0, b"")
                    return
                mean_square = sum(sample * sample for sample in samples) / len(samples)
                level = min(math.sqrt(mean_square) / 32768.0, 1.0)
                on_audio_chunk(level, pcm_bytes)
            except Exception:
                logger.exception("audio_input_processing_failed")

        with self._stream_lock:
            self._close_input_stream_locked()
            stream = sounddevice.RawInputStream(
                device=device_index,
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                blocksize=block_size,
                callback=audio_callback,
            )
            try:
                stream.start()
            except Exception:
                stream.close()
                raise
            self._input_stream = stream

    def _stop_input_monitor_sync(self) -> None:
        with self._stream_lock:
            self._close_input_stream_locked()

    def _play_wav_sync(
        self,
        device_index: int,
        wav_bytes: bytes,
        volume: float,
        balance: float,
    ) -> None:
        if not 0.0 <= volume <= 1.0:
            raise ValueError("volume must be between 0 and 1")
        if not -1.0 <= balance <= 1.0:
            raise ValueError("balance must be between -1 and 1")
        if not wav_bytes:
            raise ValueError("WAV payload is empty")

        sounddevice = self._load_sounddevice()
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            compression = wav_file.getcomptype()
            frames = wav_file.readframes(wav_file.getnframes())

        if channels <= 0 or sample_rate <= 0:
            raise ValueError("WAV stream has invalid audio metadata")
        if sample_width != 2 or compression != "NONE":
            raise ValueError("Only uncompressed 16-bit PCM WAV is supported")
        extra_settings: Any | None = None
        use_wasapi_conversion = (
            (self._system_name or platform.system()) == "Windows"
            and hasattr(sounddevice, "WasapiSettings")
        )
        if use_wasapi_conversion:
            # Shared-mode WASAPI uses the same system mixer path as Windows WAV
            # playback and preserves Piper's source PCM for high-quality conversion.
            extra_settings = sounddevice.WasapiSettings(auto_convert=True)
            if balance and channels in {1, 2}:
                output_device = sounddevice.query_devices(device_index)
                if int(self._value(output_device, "max_output_channels", 0)) >= 2:
                    frames, channels = self._apply_balance_pcm16(
                        frames,
                        channels,
                        balance,
                    )
        else:
            try:
                output_device = sounddevice.query_devices(device_index)
                target_sample_rate = round(
                    float(
                        self._value(
                            output_device,
                            "default_samplerate",
                            sample_rate,
                        )
                    )
                )
                maximum_output_channels = int(
                    self._value(output_device, "max_output_channels", channels)
                )
                target_channels = min(max(maximum_output_channels, 1), 2)
            except Exception:
                logger.exception("audio_output_sample_rate_query_failed")
                target_sample_rate = sample_rate
                target_channels = channels
            if target_sample_rate <= 0:
                target_sample_rate = sample_rate
            if target_sample_rate != sample_rate:
                frames = self._resample_pcm16(
                    frames,
                    channels,
                    sample_rate,
                    target_sample_rate,
                )
                sample_rate = target_sample_rate
            if target_channels != channels:
                frames = self._convert_channels_pcm16(
                    frames,
                    channels,
                    target_channels,
                )
                channels = target_channels
            if balance and channels == 2:
                frames, channels = self._apply_balance_pcm16(
                    frames,
                    channels,
                    balance,
                )
        if volume < 1.0:
            frames = self._scale_pcm16(frames, volume)
        if volume == 0.0:
            return

        stream_options: dict[str, Any] = {
            "device": device_index,
            "samplerate": sample_rate,
            "channels": channels,
            "dtype": "int16",
        }
        if extra_settings is not None:
            stream_options["extra_settings"] = extra_settings
        stream = sounddevice.RawOutputStream(**stream_options)
        cancel_event = threading.Event()
        with self._output_lock:
            self._output_stream = stream
            self._output_cancel_event = cancel_event
        try:
            stream.start()
            block_size = max(channels * 2 * 2048, 1)
            for offset in range(0, len(frames), block_size):
                if cancel_event.is_set():
                    break
                stream.write(frames[offset : offset + block_size])
            if not cancel_event.is_set():
                stream.stop()
        except Exception:
            if not cancel_event.is_set():
                raise
        finally:
            try:
                stream.close()
            finally:
                with self._output_lock:
                    if self._output_stream is stream:
                        self._output_stream = None
                        self._output_cancel_event = None

    def _stop_output_sync(self) -> None:
        with self._output_lock:
            cancel_event = self._output_cancel_event
            stream = self._output_stream
        if cancel_event is not None:
            cancel_event.set()
        if stream is not None:
            try:
                stream.abort()
            except Exception:
                logger.exception("audio_output_abort_failed")

    @staticmethod
    def _scale_pcm16(frames: bytes, volume: float) -> bytes:
        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        for index, sample in enumerate(samples):
            samples[index] = max(-32768, min(32767, round(sample * volume)))
        if sys.byteorder != "little":
            samples.byteswap()
        return samples.tobytes()

    @staticmethod
    def _resample_pcm16(
        frames: bytes,
        channels: int,
        source_rate: int,
        target_rate: int,
    ) -> bytes:
        if channels <= 0 or source_rate <= 0 or target_rate <= 0:
            raise ValueError("Invalid PCM resampling metadata")
        if source_rate == target_rate:
            return frames
        try:
            numpy = importlib.import_module("numpy")
        except ImportError:
            return PortAudioAdapter._resample_pcm16_linear(
                frames,
                channels,
                source_rate,
                target_rate,
            )

        samples = numpy.frombuffer(frames, dtype="<i2")
        if samples.size % channels:
            raise ValueError("PCM payload does not contain complete frames")
        source = samples.reshape(-1, channels).astype(numpy.float64)
        source_frame_count = source.shape[0]
        if source_frame_count <= 1:
            return frames

        target_frame_count = max(
            1,
            round(source_frame_count * target_rate / source_rate),
        )
        result = numpy.empty((target_frame_count, channels), dtype=numpy.float64)
        radius = 8
        offsets = numpy.arange(-radius + 1, radius + 1)
        block_size = 16384
        for start in range(0, target_frame_count, block_size):
            stop = min(start + block_size, target_frame_count)
            positions = numpy.minimum(
                numpy.arange(start, stop, dtype=numpy.float64)
                * source_rate
                / target_rate,
                source_frame_count - 1,
            )
            centers = numpy.floor(positions).astype(numpy.int64)
            indices = centers[:, None] + offsets[None, :]
            distances = positions[:, None] - indices
            weights = numpy.sinc(distances) * numpy.sinc(distances / radius)
            weights[numpy.abs(distances) >= radius] = 0.0
            clipped_indices = numpy.clip(indices, 0, source_frame_count - 1)
            weight_totals = weights.sum(axis=1, keepdims=True)
            result[start:stop] = numpy.einsum(
                "bt,btc->bc",
                weights / weight_totals,
                source[clipped_indices],
            )

        converted = numpy.clip(numpy.rint(result), -32768, 32767).astype("<i2")
        return converted.tobytes()

    @staticmethod
    def _resample_pcm16_linear(
        frames: bytes,
        channels: int,
        source_rate: int,
        target_rate: int,
    ) -> bytes:
        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        source_frame_count = len(samples) // channels
        if source_frame_count <= 1:
            return frames

        target_frame_count = max(
            1,
            round(source_frame_count * target_rate / source_rate),
        )
        result = array("h")
        for target_frame in range(target_frame_count):
            source_position = min(
                target_frame * source_rate / target_rate,
                source_frame_count - 1,
            )
            left_frame = int(source_position)
            right_frame = min(left_frame + 1, source_frame_count - 1)
            fraction = source_position - left_frame
            for channel in range(channels):
                left_sample = samples[left_frame * channels + channel]
                right_sample = samples[right_frame * channels + channel]
                interpolated = round(
                    left_sample + (right_sample - left_sample) * fraction
                )
                result.append(max(-32768, min(32767, interpolated)))
        if sys.byteorder != "little":
            result.byteswap()
        return result.tobytes()

    @staticmethod
    def _convert_channels_pcm16(
        frames: bytes,
        source_channels: int,
        target_channels: int,
    ) -> bytes:
        if source_channels == target_channels:
            return frames
        if {source_channels, target_channels} - {1, 2}:
            raise ValueError("Only mono/stereo PCM channel conversion is supported")
        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        converted = array("h")
        if source_channels == 1 and target_channels == 2:
            for sample in samples:
                converted.extend((sample, sample))
        else:
            for index in range(0, len(samples), 2):
                converted.append(round((samples[index] + samples[index + 1]) / 2))
        if sys.byteorder != "little":
            converted.byteswap()
        return converted.tobytes()

    @staticmethod
    def _apply_balance_pcm16(
        frames: bytes,
        channels: int,
        balance: float,
    ) -> tuple[bytes, int]:
        if not -1.0 <= balance <= 1.0:
            raise ValueError("balance must be between -1 and 1")
        if channels not in {1, 2}:
            raise ValueError("Spatial balance only supports mono or stereo PCM")
        if balance == 0.0:
            return frames, channels

        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        left_gain = 1.0 - max(balance, 0.0)
        right_gain = 1.0 + min(balance, 0.0)
        balanced = array("h")
        if channels == 1:
            for sample in samples:
                balanced.extend(
                    (
                        round(sample * left_gain),
                        round(sample * right_gain),
                    )
                )
        else:
            for index in range(0, len(samples), 2):
                balanced.extend(
                    (
                        round(samples[index] * left_gain),
                        round(samples[index + 1] * right_gain),
                    )
                )
        if sys.byteorder != "little":
            balanced.byteswap()
        return balanced.tobytes(), 2

    def _close_input_stream_locked(self) -> None:
        stream = self._input_stream
        self._input_stream = None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()

    def _preferred_host_api_index(
        self,
        devices: Sequence[Any],
        host_apis: Sequence[Any],
        channel_key: str,
    ) -> int | None:
        usable = {
            int(self._value(device, "hostapi", -1))
            for device in devices
            if int(self._value(device, channel_key, 0)) > 0
        }
        usable = {index for index in usable if 0 <= index < len(host_apis)}
        if not usable:
            return None

        system_name = self._system_name or platform.system()
        priorities = {
            "Windows": ("wasapi", "directsound", "mme", "wdm-ks"),
            "Linux": ("alsa", "pulse", "jack"),
        }.get(system_name, ())
        for preferred_name in priorities:
            for host_api_index in sorted(usable):
                host_api_name = str(
                    self._value(host_apis[host_api_index], "name", "")
                ).casefold()
                if preferred_name in host_api_name:
                    return host_api_index
        return min(usable)

    @staticmethod
    def _value(record: Any, key: str, default: Any) -> Any:
        if isinstance(record, Mapping):
            return record.get(key, default)
        try:
            return record[key]
        except (KeyError, IndexError, TypeError):
            return getattr(record, key, default)
