from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SpotifyError(RuntimeError):
    """Base error exposed by the optional Spotify integration."""


class SpotifyNotConfiguredError(SpotifyError):
    """Raised when no Spotify Client ID has been configured."""


class SpotifyNotConnectedError(SpotifyError):
    """Raised when the local OAuth authorization is unavailable or expired."""


@dataclass(frozen=True)
class SpotifyPlayback:
    track_id: str = ""
    title: str = ""
    artists: str = ""
    album: str = ""
    is_playing: bool = False
    progress_ms: int = 0
    duration_ms: int = 0
    device_name: str = ""

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class SpotifyStatus:
    configured: bool = False
    connected: bool = False
    detail: str = "not_configured"
    playback: SpotifyPlayback | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "configured": self.configured,
            "connected": self.connected,
            "detail": self.detail,
            "playback": self.playback.to_dict() if self.playback else None,
        }


class SpotifyAdapter(Protocol):
    def configure(self, client_id: str) -> None:
        ...

    async def authorization_url(self) -> str:
        ...

    async def complete_authorization(self, code: str, state: str) -> SpotifyStatus:
        ...

    async def status(self) -> SpotifyStatus:
        ...

    async def play_query(
        self,
        query: str,
        *,
        prefer_artist: bool = False,
    ) -> SpotifyStatus:
        ...

    async def control(self, action: str) -> SpotifyStatus:
        ...

    async def disconnect(self) -> SpotifyStatus:
        ...
