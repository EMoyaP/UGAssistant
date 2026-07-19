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

    async def web_player_access_token(self) -> str:
        return await self._adapter.web_player_access_token()

    def note_web_player_available(self) -> None:
        self._adapter.note_web_player_available()

    async def set_web_player_device(self, device_id: str) -> SpotifyStatus:
        return await self._publish(await self._adapter.set_web_player_device(device_id))

    async def authorization_url(self) -> str:
        return await self._adapter.authorization_url()

    async def complete_authorization(self, code: str, state: str) -> SpotifyStatus:
        return await self._publish(
            await self._adapter.complete_authorization(code, state)
        )

    async def play_query(
        self,
        query: str,
        *,
        prefer_artist: bool = False,
    ) -> SpotifyStatus:
        return await self._publish(
            await self._adapter.play_query(query, prefer_artist=prefer_artist)
        )

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
