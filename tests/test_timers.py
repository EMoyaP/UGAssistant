from __future__ import annotations

import asyncio
import io
import unittest
import wave
from time import time

from ugassistant.domain.timer_alarm import build_timer_alarm_wav
from ugassistant.domain.timers import TimerSnapshot
from ugassistant.services.timers import TimerService


class TimerServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_labels_follow_creation_and_reset_after_last_timer(self) -> None:
        service = TimerService()

        first = await service.create(3600)
        second = await service.create(300)
        third = await service.create(1800)

        self.assertEqual((first.label, second.label, third.label), (1, 2, 3))
        self.assertEqual([timer.label for timer in service.timers], [2, 3, 1])

        await service.cancel(2)
        fourth = await service.create(120)
        self.assertEqual(fourth.label, 4)

        await service.cancel(1)
        await service.cancel(3)
        await service.cancel(4)
        reset = await service.create(60)
        self.assertEqual(reset.label, 1)
        await service.shutdown()

    async def test_alarm_wave_is_two_seconds_without_external_asset(self) -> None:
        wav_bytes = build_timer_alarm_wav()

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getnframes(), 32000)

    async def test_modification_keeps_its_label_and_reorders_the_status(self) -> None:
        service = TimerService()
        first = await service.create(300)
        second = await service.create(600)

        changed = await service.modify(second.label, 60)

        self.assertEqual(changed.label, second.label)
        self.assertEqual([timer.label for timer in service.timers], [2, 1])
        await service.shutdown()

    async def test_expiration_removes_timer_before_notifying(self) -> None:
        expired: list[TimerSnapshot] = []
        delivered = asyncio.Event()

        async def on_expired(timer: TimerSnapshot) -> None:
            expired.append(timer)
            delivered.set()

        service = TimerService(on_expired=on_expired)
        timer = await service.create(60)
        service._timers[timer.label] = TimerSnapshot(  # type: ignore[attr-defined]
            label=timer.label,
            duration_seconds=timer.duration_seconds,
            ends_at_epoch_ms=round((time() - 1) * 1000),
            language="es",
        )

        await service._expire_due()  # type: ignore[attr-defined]
        await asyncio.wait_for(delivered.wait(), timeout=0.5)

        self.assertEqual([item.label for item in expired], [1])
        self.assertEqual(service.timers, ())
        reset = await service.create(60)
        self.assertEqual(reset.label, 1)
        await service.shutdown()


if __name__ == "__main__":
    unittest.main()
