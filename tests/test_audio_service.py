from __future__ import annotations

import asyncio
from array import array
import io
import unittest
import wave

from ugassistant.adapters.simulated import SimulatedAudioAdapter
from ugassistant.domain.ports import AudioDevice
from ugassistant.domain.preferences import DevicePreference
from ugassistant.services.audio import AudioDeviceService


class ReindexedAudioAdapter(SimulatedAudioAdapter):
    async def list_devices(self) -> list[AudioDevice]:
        return [
            AudioDevice(
                device_index=20,
                name="Simulated microphone",
                kind="input",
                available=True,
                channels=1,
                default_sample_rate=16000.0,
                host_api="Simulated",
                is_default=True,
            ),
            AudioDevice(
                device_index=21,
                name="Simulated speakers",
                kind="output",
                available=True,
                channels=2,
                default_sample_rate=48000.0,
                host_api="Simulated",
                is_default=True,
            ),
        ]


class CountingAudioAdapter(SimulatedAudioAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.list_devices_calls = 0

    async def list_devices(self) -> list[AudioDevice]:
        self.list_devices_calls += 1
        return await super().list_devices()


class AudioDeviceServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_selects_default_input_and_output_on_first_scan(self) -> None:
        service = AudioDeviceService(SimulatedAudioAdapter())

        status = await service.refresh()

        self.assertTrue(status.ready)
        self.assertEqual(status.selected_input_index, 0)
        self.assertEqual(status.selected_output_index, 1)
        self.assertTrue(status.output_enabled)
        self.assertEqual(status.to_dict()["inputs"][0]["channels"], 1)  # type: ignore[index]

    async def test_can_select_none_without_reselecting_a_default(self) -> None:
        service = AudioDeviceService(SimulatedAudioAdapter())
        await service.refresh()

        status = await service.select_device("input", None)
        refreshed = await service.refresh()

        self.assertIsNone(status.selected_input_index)
        self.assertIsNone(refreshed.selected_input_index)
        self.assertEqual(refreshed.selected_output_index, 1)

    async def test_restored_devices_follow_identity_instead_of_old_index(self) -> None:
        service = AudioDeviceService(ReindexedAudioAdapter())
        service.restore_device_preference(
            "input",
            DevicePreference("Simulated microphone", 0, "Simulated"),
        )
        service.restore_device_preference(
            "output",
            DevicePreference("Simulated speakers", 1, "Simulated"),
        )

        status = await service.refresh()

        self.assertEqual(status.selected_input_index, 20)
        self.assertEqual(status.selected_output_index, 21)

    async def test_missing_preferred_device_does_not_select_default(self) -> None:
        service = AudioDeviceService(SimulatedAudioAdapter())
        service.restore_device_preference(
            "input",
            DevicePreference("Disconnected microphone", 0, "Simulated"),
        )

        status = await service.refresh()

        self.assertIsNone(status.selected_input_index)

    async def test_rejects_an_unknown_device_or_kind(self) -> None:
        service = AudioDeviceService(SimulatedAudioAdapter())

        with self.assertRaises(ValueError):
            await service.select_device("input", 99)
        with self.assertRaises(ValueError):
            await service.select_device("duplex", 0)

    async def test_monitors_levels_and_releases_after_silence(self) -> None:
        adapter = SimulatedAudioAdapter()
        service = AudioDeviceService(
            adapter,
            activation_threshold=0.02,
            release_threshold=0.01,
            activation_samples=1,
            silence_seconds=0.01,
        )
        await service.refresh()

        enabled = await service.enable_monitoring()
        adapter.emit_input_level(0.08)
        await asyncio.sleep(0.01)
        active = service.status
        await asyncio.sleep(0.02)
        adapter.emit_input_level(0.0)
        await asyncio.sleep(0.01)
        silent = service.status
        disabled = await service.disable_monitoring()

        self.assertTrue(enabled.monitoring)
        self.assertTrue(active.sound_detected)
        self.assertGreater(active.input_level, 0.02)
        self.assertFalse(silent.sound_detected)
        self.assertFalse(disabled.monitoring)
        self.assertFalse(adapter.monitoring)

    async def test_coalesces_a_burst_of_monitor_levels(self) -> None:
        adapter = SimulatedAudioAdapter()
        published_levels: list[float] = []

        async def on_status(status: object) -> None:
            published_levels.append(getattr(status, "input_level"))

        service = AudioDeviceService(adapter, on_status=on_status)
        await service.refresh()
        await service.enable_monitoring()
        baseline = len(published_levels)

        for level in range(100):
            adapter.emit_input_level(level / 100)
        await asyncio.sleep(0.05)

        self.assertLessEqual(len(published_levels) - baseline, 2)
        self.assertAlmostEqual(service.status.input_level, 0.99, places=2)
        await service.disable_monitoring()

    async def test_reuses_cached_devices_when_monitoring_restarts(self) -> None:
        adapter = CountingAudioAdapter()
        service = AudioDeviceService(adapter)
        await service.refresh()
        await service.enable_monitoring()
        await service.disable_monitoring()
        await service.enable_monitoring()

        self.assertEqual(adapter.list_devices_calls, 1)
        await service.disable_monitoring()

    async def test_requires_a_selected_input_to_start_monitoring(self) -> None:
        service = AudioDeviceService(SimulatedAudioAdapter())
        await service.refresh()
        await service.select_device("input", None)

        with self.assertRaises(RuntimeError):
            await service.enable_monitoring()

    async def test_captures_until_a_long_pause_and_returns_16khz_wav(self) -> None:
        adapter = SimulatedAudioAdapter()
        service = AudioDeviceService(
            adapter,
            activation_threshold=0.02,
            release_threshold=0.01,
            activation_samples=1,
            block_duration_ms=50,
        )
        await service.refresh()

        task = asyncio.create_task(
            service.capture_utterance(
                wait_for_speech_seconds=1.0,
                silence_seconds=0.1,
                max_recording_seconds=2.0,
            )
        )
        for _attempt in range(50):
            if adapter._on_audio_chunk is not None:
                break
            await asyncio.sleep(0.01)
        adapter.emit_input_audio(0.1, array("h", [4000] * 800).tobytes())
        adapter.emit_input_audio(0.0, array("h", [0] * 800).tobytes())
        adapter.emit_input_audio(0.0, array("h", [0] * 800).tobytes())
        wav_bytes = await task

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertGreaterEqual(wav_file.getnframes(), 2400)
        self.assertFalse(service.status.recording)

    async def test_output_can_be_disabled_without_losing_its_selection(self) -> None:
        service = AudioDeviceService(SimulatedAudioAdapter())
        await service.refresh()

        disabled = await service.disable_output()
        enabled = await service.enable_output()

        self.assertFalse(disabled.output_enabled)
        self.assertEqual(disabled.selected_output_index, 1)
        self.assertTrue(enabled.output_enabled)
        self.assertEqual(enabled.selected_output_index, 1)

    async def test_output_without_a_selection_cannot_be_enabled(self) -> None:
        service = AudioDeviceService(SimulatedAudioAdapter())
        await service.refresh()
        await service.select_device("output", None)

        with self.assertRaises(RuntimeError):
            await service.enable_output()

    async def test_changing_input_resumes_an_active_monitor(self) -> None:
        adapter = SimulatedAudioAdapter()
        service = AudioDeviceService(adapter)
        await service.refresh()
        await service.enable_monitoring()

        status = await service.select_device("input", 0)

        self.assertTrue(status.monitoring)
        self.assertTrue(adapter.monitoring)

    async def test_output_volume_and_playback_use_selected_device(self) -> None:
        adapter = SimulatedAudioAdapter()
        service = AudioDeviceService(adapter)
        await service.refresh()

        volume_status = await service.set_output_volume(0.65)
        played_status = await service.play_wav(b"SIMULATED_WAV", balance=-0.4)

        self.assertEqual(volume_status.output_volume, 0.65)
        self.assertFalse(played_status.output_playing)
        self.assertEqual(played_status.output_balance, -0.4)
        self.assertEqual(
            adapter.played_audio,
            [(1, b"SIMULATED_WAV", 0.65, -0.4)],
        )

    async def test_output_volume_rejects_values_outside_range(self) -> None:
        service = AudioDeviceService(SimulatedAudioAdapter())

        with self.assertRaises(ValueError):
            await service.set_output_volume(1.1)

    async def test_output_balance_rejects_values_outside_range(self) -> None:
        service = AudioDeviceService(SimulatedAudioAdapter())
        await service.refresh()

        with self.assertRaises(ValueError):
            await service.play_wav(b"SIMULATED_WAV", balance=-1.1)


if __name__ == "__main__":
    unittest.main()
