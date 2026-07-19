from __future__ import annotations

from array import array
import io
import unittest
import wave

from ugassistant.adapters.portaudio import PortAudioAdapter


class FakeSoundDevice:
    streams: list[object] = []
    output_streams: list[object] = []

    class WasapiSettings:
        def __init__(self, *, auto_convert: bool = False) -> None:
            self.auto_convert = auto_convert

    class RawInputStream:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            self.closed = False
            FakeSoundDevice.streams.append(self)

        def start(self) -> None:
            self.started = True

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    class RawOutputStream:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.started = False
            self.stopped = False
            self.closed = False
            self.written = b""
            FakeSoundDevice.output_streams.append(self)

        def start(self) -> None:
            self.started = True

        def write(self, data: bytes) -> None:
            self.written += data

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            self.closed = True

    @staticmethod
    def query_hostapis() -> list[dict[str, object]]:
        return [
            {
                "name": "MME",
                "default_input_device": 0,
                "default_output_device": 1,
            },
            {
                "name": "Windows WASAPI",
                "default_input_device": 2,
                "default_output_device": 3,
            },
            {
                "name": "ALSA",
                "default_input_device": 4,
                "default_output_device": 5,
            },
        ]

    @staticmethod
    def query_devices(device: int | None = None) -> object:
        devices = [
            {
                "name": "MME microphone",
                "hostapi": 0,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 44100.0,
            },
            {
                "name": "MME speakers",
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            },
            {
                "name": "USB microphone",
                "hostapi": 1,
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
            },
            {
                "name": "Display speakers",
                "hostapi": 1,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
            {
                "name": "USB capture",
                "hostapi": 2,
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000.0,
            },
            {
                "name": "Headphones",
                "hostapi": 2,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            },
        ]
        return devices if device is None else devices[device]


class PortAudioAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_windows_prefers_wasapi_and_preserves_real_indices(self) -> None:
        adapter = PortAudioAdapter(
            sounddevice_module=FakeSoundDevice(),
            system_name="Windows",
        )

        devices = await adapter.list_devices()

        self.assertEqual(
            [(device.kind, device.device_index) for device in devices],
            [("input", 2), ("output", 3)],
        )
        self.assertTrue(all(device.host_api == "Windows WASAPI" for device in devices))
        self.assertTrue(all(device.is_default for device in devices))

    async def test_linux_prefers_alsa_for_raspberry_pi(self) -> None:
        adapter = PortAudioAdapter(
            sounddevice_module=FakeSoundDevice(),
            system_name="Linux",
        )

        devices = await adapter.list_devices()

        self.assertEqual(
            [(device.name, device.device_index) for device in devices],
            [("USB capture", 4), ("Headphones", 5)],
        )
        self.assertEqual(devices[0].channels, 1)
        self.assertEqual(devices[0].default_sample_rate, 16000.0)
        self.assertTrue(all(device.host_api == "ALSA" for device in devices))

    async def test_input_monitor_reports_normalized_rms_and_closes_stream(self) -> None:
        FakeSoundDevice.streams.clear()
        levels: list[float] = []
        adapter = PortAudioAdapter(
            sounddevice_module=FakeSoundDevice(),
            system_name="Windows",
        )

        await adapter.start_input_monitor(2, 48000, 50, levels.append)
        stream = FakeSoundDevice.streams[0]
        callback = stream.kwargs["callback"]  # type: ignore[attr-defined,index]
        callback(array("h", [3277] * 20).tobytes(), 20, None, None)  # type: ignore[operator]
        await adapter.stop_input_monitor()

        self.assertTrue(stream.started)  # type: ignore[attr-defined]
        self.assertTrue(stream.stopped)  # type: ignore[attr-defined]
        self.assertTrue(stream.closed)  # type: ignore[attr-defined]
        self.assertAlmostEqual(levels[0], 0.1, places=3)

    async def test_input_capture_reports_level_and_pcm_bytes(self) -> None:
        FakeSoundDevice.streams.clear()
        chunks: list[tuple[float, bytes]] = []
        adapter = PortAudioAdapter(
            sounddevice_module=FakeSoundDevice(),
            system_name="Windows",
        )
        pcm_bytes = array("h", [6554] * 20).tobytes()

        await adapter.start_input_capture(2, 48000, 50, lambda *item: chunks.append(item))
        stream = FakeSoundDevice.streams[0]
        callback = stream.kwargs["callback"]  # type: ignore[attr-defined,index]
        callback(pcm_bytes, 20, None, None)  # type: ignore[operator]
        await adapter.stop_input_capture()

        self.assertAlmostEqual(chunks[0][0], 0.2, places=3)
        self.assertEqual(chunks[0][1], pcm_bytes)

    async def test_plays_pcm_wav_on_selected_output_with_volume(self) -> None:
        FakeSoundDevice.output_streams.clear()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48000)
            wav_file.writeframes(array("h", [1000, -1000]).tobytes())
        adapter = PortAudioAdapter(
            sounddevice_module=FakeSoundDevice(),
            system_name="Windows",
        )

        await adapter.play_wav(3, buffer.getvalue(), 0.5)

        stream = FakeSoundDevice.output_streams[0]
        self.assertEqual(stream.kwargs["device"], 3)  # type: ignore[attr-defined,index]
        self.assertEqual(stream.kwargs["samplerate"], 48000)  # type: ignore[attr-defined,index]
        self.assertEqual(stream.kwargs["channels"], 1)  # type: ignore[attr-defined,index]
        self.assertTrue(  # type: ignore[attr-defined,index]
            stream.kwargs["extra_settings"].auto_convert
        )
        samples = array("h")
        samples.frombytes(stream.written)  # type: ignore[attr-defined]
        self.assertEqual(samples.tolist(), [500, -500])
        self.assertTrue(stream.started)  # type: ignore[attr-defined]
        self.assertTrue(stream.stopped)  # type: ignore[attr-defined]
        self.assertTrue(stream.closed)  # type: ignore[attr-defined]

    async def test_windows_wasapi_preserves_piper_source_format(self) -> None:
        FakeSoundDevice.output_streams.clear()
        buffer = io.BytesIO()
        source_samples = array("h", [0, 1200, -1200, 0]).tobytes()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(source_samples)
        adapter = PortAudioAdapter(
            sounddevice_module=FakeSoundDevice(),
            system_name="Windows",
        )

        await adapter.play_wav(3, buffer.getvalue(), 1.0)

        stream = FakeSoundDevice.output_streams[0]
        self.assertEqual(stream.kwargs["samplerate"], 22050)  # type: ignore[attr-defined,index]
        self.assertEqual(stream.kwargs["channels"], 1)  # type: ignore[attr-defined,index]
        self.assertEqual(stream.written, source_samples)  # type: ignore[attr-defined]

    def test_resamples_pcm16_for_fixed_rate_outputs(self) -> None:
        source = array("h", [0, 1000]).tobytes()

        converted = PortAudioAdapter._resample_pcm16(source, 1, 2, 4)

        samples = array("h")
        samples.frombytes(converted)
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0], 0)
        self.assertAlmostEqual(samples[1], 500, delta=30)
        self.assertEqual(samples[2:], array("h", [1000, 1000]))

    def test_resampler_preserves_constant_pcm_without_noise(self) -> None:
        source = array("h", [1234] * 2205).tobytes()

        converted = PortAudioAdapter._resample_pcm16(source, 1, 22050, 48000)

        samples = array("h")
        samples.frombytes(converted)
        self.assertEqual(len(samples), 4800)
        self.assertEqual(set(samples), {1234})

    def test_duplicates_mono_samples_for_stereo_output(self) -> None:
        source = array("h", [120, -240]).tobytes()

        converted = PortAudioAdapter._convert_channels_pcm16(source, 1, 2)

        samples = array("h")
        samples.frombytes(converted)
        self.assertEqual(samples.tolist(), [120, 120, -240, -240])

    def test_spatial_balance_pans_mono_without_muting_opposite_channel(self) -> None:
        source = array("h", [1000, -1000]).tobytes()

        converted, channels = PortAudioAdapter._apply_balance_pcm16(
            source,
            1,
            0.6,
        )

        samples = array("h")
        samples.frombytes(converted)
        self.assertEqual(channels, 2)
        self.assertEqual(samples.tolist(), [400, 1000, -400, -1000])


if __name__ == "__main__":
    unittest.main()
