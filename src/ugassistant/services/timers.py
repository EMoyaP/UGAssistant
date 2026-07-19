from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from time import time

from ugassistant.domain.timers import TimerSnapshot


logger = logging.getLogger("ugassistant.timers")

TimerStatusCallback = Callable[[tuple[TimerSnapshot, ...]], Awaitable[None]]
TimerExpiredCallback = Callable[[TimerSnapshot], Awaitable[None]]


class TimerService:
    """Schedules independent local countdowns and keeps creation labels stable."""

    def __init__(
        self,
        *,
        on_status: TimerStatusCallback | None = None,
        on_expired: TimerExpiredCallback | None = None,
    ) -> None:
        self._on_status = on_status
        self._on_expired = on_expired
        self._timers: dict[int, TimerSnapshot] = {}
        self._next_label = 1
        self._changed = asyncio.Event()
        self._runner: asyncio.Task[None] | None = None
        self._expiry_tasks: set[asyncio.Task[None]] = set()

    @property
    def timers(self) -> tuple[TimerSnapshot, ...]:
        return tuple(
            sorted(
                self._timers.values(),
                key=lambda timer: (timer.ends_at_epoch_ms, timer.label),
            )
        )

    async def start(self) -> None:
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run())

    def set_callbacks(
        self,
        *,
        on_status: TimerStatusCallback | None = None,
        on_expired: TimerExpiredCallback | None = None,
    ) -> None:
        self._on_status = on_status
        self._on_expired = on_expired

    async def shutdown(self) -> None:
        if self._runner is not None and not self._runner.done():
            self._runner.cancel()
            try:
                await self._runner
            except asyncio.CancelledError:
                pass
        self._runner = None
        for task in tuple(self._expiry_tasks):
            task.cancel()
        if self._expiry_tasks:
            await asyncio.gather(*self._expiry_tasks, return_exceptions=True)
        self._expiry_tasks.clear()

    async def create(self, duration_seconds: int, *, language: str = "es") -> TimerSnapshot:
        if duration_seconds <= 0:
            raise ValueError("Timer duration must be positive")
        await self.start()
        snapshot = TimerSnapshot(
            label=self._next_label,
            duration_seconds=duration_seconds,
            ends_at_epoch_ms=round((time() + duration_seconds) * 1000),
            language=language,
        )
        self._next_label += 1
        self._timers[snapshot.label] = snapshot
        self._changed.set()
        await self._publish()
        return snapshot

    async def modify(self, label: int, duration_seconds: int) -> TimerSnapshot:
        if duration_seconds <= 0:
            raise ValueError("Timer duration must be positive")
        current = self._timers.get(label)
        if current is None:
            raise KeyError(label)
        snapshot = TimerSnapshot(
            label=current.label,
            duration_seconds=duration_seconds,
            ends_at_epoch_ms=round((time() + duration_seconds) * 1000),
            language=current.language,
        )
        self._timers[label] = snapshot
        self._changed.set()
        await self._publish()
        return snapshot

    async def cancel(self, label: int) -> TimerSnapshot:
        snapshot = self._timers.pop(label, None)
        if snapshot is None:
            raise KeyError(label)
        if not self._timers:
            self._next_label = 1
        self._changed.set()
        await self._publish()
        return snapshot

    async def _run(self) -> None:
        try:
            while True:
                if not self._timers:
                    self._changed.clear()
                    await self._changed.wait()
                    continue
                next_expiry = min(
                    timer.ends_at_epoch_ms for timer in self._timers.values()
                )
                wait_seconds = max((next_expiry / 1000) - time(), 0.0)
                self._changed.clear()
                try:
                    await asyncio.wait_for(self._changed.wait(), timeout=wait_seconds)
                    continue
                except asyncio.TimeoutError:
                    await self._expire_due()
        except asyncio.CancelledError:
            raise

    async def _expire_due(self) -> None:
        now_ms = round(time() * 1000)
        expired = tuple(
            timer
            for timer in self.timers
            if timer.ends_at_epoch_ms <= now_ms
        )
        if not expired:
            return
        for timer in expired:
            self._timers.pop(timer.label, None)
        if not self._timers:
            self._next_label = 1
        await self._publish()
        for timer in expired:
            task = asyncio.create_task(self._notify_expired(timer))
            self._expiry_tasks.add(task)
            task.add_done_callback(self._discard_expiry_task)

    async def _publish(self) -> None:
        if self._on_status is not None:
            await self._on_status(self.timers)

    async def _notify_expired(self, timer: TimerSnapshot) -> None:
        if self._on_expired is not None:
            await self._on_expired(timer)

    def _discard_expiry_task(self, task: asyncio.Task[None]) -> None:
        self._expiry_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("timer_expiration_callback_failed", exc_info=task.exception())
