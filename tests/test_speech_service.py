from __future__ import annotations

import asyncio
import unittest

from ugassistant.adapters.simulated import SimulatedAudioAdapter, SimulatedTTSAdapter
from ugassistant.services.audio import AudioDeviceService
from ugassistant.services.speech import SpeechService, SpeechStatus


class SpeechServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_speaks_with_selected_voice_and_restores_monitor(self) -> None:
        audio_adapter = SimulatedAudioAdapter()
        audio_service = AudioDeviceService(audio_adapter, output_volume=0.7)
        tts_adapter = SimulatedTTSAdapter()
        phases: list[str] = []

        async def on_status(status: SpeechStatus) -> None:
            phases.append(status.phase)

        service = SpeechService(
            tts_adapter,
            audio_service,
            default_voice_id="es_ES-davefx-medium",
            output_guard_seconds=0.0,
            on_status=on_status,
            balance_provider=lambda: 0.45,
        )
        await audio_service.refresh()
        await audio_service.enable_monitoring()
        await service.refresh()

        status = await service.speak("Hola desde local")

        self.assertTrue(status.ready)
        self.assertTrue(audio_service.status.monitoring)
        self.assertEqual(
            tts_adapter.synthesized,
            [("Hola desde local", "es_ES-davefx-medium")],
        )
        self.assertEqual(audio_adapter.played_audio[0][0], 1)
        self.assertEqual(audio_adapter.played_audio[0][2], 0.7)
        self.assertEqual(audio_adapter.played_audio[0][3], 0.45)
        self.assertIn("synthesizing", phases)
        self.assertIn("playing", phases)
        self.assertEqual(phases[-1], "ready")

    async def test_persists_a_slower_speech_rate_for_synthesis(self) -> None:
        audio_service = AudioDeviceService(SimulatedAudioAdapter())
        tts_adapter = SimulatedTTSAdapter()
        service = SpeechService(
            tts_adapter,
            audio_service,
            default_voice_id="es_ES-davefx-medium",
        )
        await audio_service.refresh()
        await audio_service.enable_output()
        await service.refresh()

        status = await service.set_speech_rate(0.8)
        await service.speak("Habla mas despacio")

        self.assertEqual(status.speech_rate, 0.8)
        self.assertEqual(tts_adapter.speech_rates, [0.8])

    async def test_interrupt_stops_current_audio_playback(self) -> None:
        class BlockingAudioAdapter(SimulatedAudioAdapter):
            def __init__(self) -> None:
                super().__init__()
                self.playback_started = asyncio.Event()
                self.playback_released = asyncio.Event()

            async def play_wav(
                self,
                device_index: int,
                wav_bytes: bytes,
                volume: float,
                balance: float = 0.0,
            ) -> None:
                await super().play_wav(device_index, wav_bytes, volume, balance)
                self.playback_started.set()
                await self.playback_released.wait()

            async def stop_output(self) -> None:
                await super().stop_output()
                self.playback_released.set()

        audio_adapter = BlockingAudioAdapter()
        audio_service = AudioDeviceService(audio_adapter)
        service = SpeechService(
            SimulatedTTSAdapter(),
            audio_service,
            default_voice_id="es_ES-davefx-medium",
        )
        await audio_service.refresh()
        await audio_service.enable_output()
        await service.refresh()

        speaking = asyncio.create_task(service.speak("Deten esta lectura"))
        await asyncio.wait_for(audio_adapter.playback_started.wait(), timeout=1.0)
        status = await service.interrupt()

        with self.assertRaises(asyncio.CancelledError):
            await speaking
        self.assertEqual(status.detail, "interrupted")
        self.assertEqual(audio_adapter.output_stop_count, 1)

    async def test_selects_only_supported_language_and_voice(self) -> None:
        audio_service = AudioDeviceService(SimulatedAudioAdapter())
        service = SpeechService(
            SimulatedTTSAdapter(),
            audio_service,
            default_voice_id="es_ES-davefx-medium",
        )
        await audio_service.refresh()

        selected = await service.select_language("es_ES")

        self.assertEqual(selected.selected_voice_id, "es_ES-davefx-medium")
        selected = await service.select_language("fr_FR")
        self.assertEqual(selected.selected_voice_id, "fr_FR-tom-medium")
        self.assertEqual(selected.selected_language, "fr_FR")
        with self.assertRaises(ValueError):
            await service.select_language("en_US")
        with self.assertRaises(ValueError):
            await service.select_voice("another-voice")

    async def test_rejects_empty_text_and_splits_long_text(self) -> None:
        audio_service = AudioDeviceService(SimulatedAudioAdapter())
        tts_adapter = SimulatedTTSAdapter()
        service = SpeechService(
            tts_adapter,
            audio_service,
            default_voice_id="es_ES-davefx-medium",
            max_text_length=5,
            output_guard_seconds=0.0,
        )
        await audio_service.refresh()
        await audio_service.enable_output()

        with self.assertRaises(ValueError):
            await service.speak("   ")
        await service.speak("uno dos tres")
        self.assertEqual(
            [text for text, _ in tts_adapter.synthesized],
            ["uno", "dos", "tres"],
        )


if __name__ == "__main__":
    unittest.main()
