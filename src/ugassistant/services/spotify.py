from __future__ import annotations

from collections.abc import Awaitable, Callable

from ugassistant.domain.spotify import SpotifyAdapter, SpotifyStatus


SpotifyStatusCallback = Callable[[SpotifyStatus], Awaitable[None]]


class SpotifyService:
    def __init__(
        self,
        adapter: SpotifyAdapter,
        *,
        on_status: SpotifyStatusCallback | None = None,
    ) -> None:
        self._adapter = adapter
        self._on_status = on_status
        self._status = SpotifyStatus()

    @property
    def status(self) -> SpotifyStatus:
        return self._status

    def configure(self, client_id: str) -> None:
        self._adapter.configure(client_id)

    async def refresh(self) -> SpotifyStatus:
        return await self._publish(await self._adapter.status())

    async def authorization_url(self) -> str:
        return await self._adapter.authorization_url()

    async def complete_authorization(self, code: str, state: str) -> SpotifyStatus:
        return await self._publish(
            await self._adapter.complete_authorization(code, state)
        )

    async def play_query(self, query: str) -> SpotifyStatus:
        return await self._publish(await self._adapter.play_query(query))

    async def control(self, action: str) -> SpotifyStatus:
        return await self._publish(await self._adapter.control(action))

    async def stop(self) -> SpotifyStatus:
        if not self._status.connected:
            return self._status
        return await self.control("pause")

    async def disconnect(self) -> SpotifyStatus:
        return await self._publish(await self._adapter.disconnect())

    async def _publish(self, status: SpotifyStatus) -> SpotifyStatus:
        self._status = status
        if self._on_status is not None:
            await self._on_status(status)
        return status
