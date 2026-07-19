from __future__ import annotations

import asyncio
import base64
import ctypes
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from secrets import token_urlsafe
from time import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ugassistant.domain.spotify import (
    SpotifyError,
    SpotifyNotConfiguredError,
    SpotifyNotConnectedError,
    SpotifyPlayback,
    SpotifyStatus,
)


@dataclass(frozen=True)
class SpotifyToken:
    access_token: str
    refresh_token: str
    expires_at: float

    def to_dict(self) -> dict[str, object]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, value: object) -> SpotifyToken | None:
        if not isinstance(value, dict):
            return None
        access_token = str(value.get("access_token", "")).strip()
        refresh_token = str(value.get("refresh_token", "")).strip()
        if not access_token or not refresh_token:
            return None
        try:
            expires_at = float(value.get("expires_at", 0))
        except (TypeError, ValueError):
            return None
        return cls(access_token, refresh_token, expires_at)


class LocalSpotifyTokenStore:
    """Stores OAuth tokens only in a user-owned local file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> SpotifyToken | None:
        if not self._path.is_file():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as source:
                payload = json.load(source)
            if isinstance(payload, dict) and payload.get("protection") == "dpapi":
                encoded = str(payload.get("payload", ""))
                decrypted = _unprotect_windows(base64.b64decode(encoded))
                payload = json.loads(decrypted.decode("utf-8"))
            return SpotifyToken.from_dict(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def save(self, token: SpotifyToken) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            payload: dict[str, object] = token.to_dict()
            if os.name == "nt":
                encrypted = _protect_windows(
                    json.dumps(payload, separators=(",", ":")).encode("utf-8")
                )
                payload = {
                    "protection": "dpapi",
                    "payload": base64.b64encode(encrypted).decode("ascii"),
                }
            json.dump(payload, target, separators=(",", ":"))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, self._path)
        try:
            os.chmod(self._path, 0o600)
        except OSError:
            pass

    def clear(self) -> None:
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_byte)),
    ]


def _protect_windows(data: bytes) -> bytes:
    if os.name != "nt":
        return data
    return _crypt_data(data, protect=True)


def _unprotect_windows(data: bytes) -> bytes:
    if os.name != "nt":
        return data
    return _crypt_data(data, protect=False)


def _crypt_data(data: bytes, *, protect: bool) -> bytes:
    source = ctypes.create_string_buffer(data)
    input_blob = _DataBlob(
        len(data),
        ctypes.cast(source, ctypes.POINTER(ctypes.c_byte)),
    )
    output_blob = _DataBlob()
    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32
    if protect:
        succeeded = crypt32.CryptProtectData(
            ctypes.byref(input_blob),
            "UGAssistant Spotify token",
            None,
            None,
            None,
            0,
            ctypes.byref(output_blob),
        )
    else:
        succeeded = crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            None,
            None,
            None,
            None,
            0,
            ctypes.byref(output_blob),
        )
    if not succeeded:
        raise SpotifyError("Windows could not protect Spotify credentials")
    try:
        return ctypes.string_at(output_blob.pbData, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)


class SpotifyWebAPIAdapter:
    AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    API_URL = "https://api.spotify.com/v1"
    PLAYBACK_CONFIRMATION_ATTEMPTS = 6
    PLAYBACK_CONFIRMATION_DELAY_SECONDS = 0.5
    SCOPES = (
        "streaming",
        "user-read-email",
        "user-read-private",
        "user-read-playback-state",
        "user-read-currently-playing",
        "user-modify-playback-state",
    )

    def __init__(
        self,
        token_store: LocalSpotifyTokenStore,
        *,
        redirect_uri: str,
        market: str = "ES",
    ) -> None:
        self._token_store = token_store
        self._redirect_uri = redirect_uri
        self._market = market
        self._client_id = ""
        self._pending_state = ""
        self._pending_verifier = ""
        self._web_player_device_id = ""

    def configure(self, client_id: str) -> None:
        normalized_client_id = client_id.strip()
        if normalized_client_id != self._client_id:
            self._web_player_device_id = ""
        self._client_id = normalized_client_id

    async def authorization_url(self) -> str:
        if not self._client_id:
            raise SpotifyNotConfiguredError("Spotify Client ID is not configured")
        self._pending_state = token_urlsafe(24)
        self._pending_verifier = token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(self._pending_verifier.encode("ascii")).digest()
        ).decode("ascii").rstrip("=")
        parameters = urlencode(
            {
                "client_id": self._client_id,
                "response_type": "code",
                "redirect_uri": self._redirect_uri,
                "scope": " ".join(self.SCOPES),
                "state": self._pending_state,
                "code_challenge_method": "S256",
                "code_challenge": challenge,
                "show_dialog": "true",
            }
        )
        return f"{self.AUTHORIZE_URL}?{parameters}"

    async def complete_authorization(self, code: str, state: str) -> SpotifyStatus:
        if not self._pending_state or state != self._pending_state:
            raise SpotifyError("Invalid Spotify authorization state")
        response = await self._token_request(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self._redirect_uri,
                "client_id": self._client_id,
                "code_verifier": self._pending_verifier,
            }
        )
        self._pending_state = ""
        self._pending_verifier = ""
        self._save_token_response(response)
        return await self.status()

    async def status(self) -> SpotifyStatus:
        if not self._client_id:
            return SpotifyStatus(detail="not_configured")
        token = self._token_store.load()
        if token is None:
            return SpotifyStatus(configured=True, detail="not_connected")
        try:
            playback = await self._current_playback()
        except SpotifyNotConnectedError:
            return SpotifyStatus(configured=True, detail="not_connected")
        return SpotifyStatus(
            configured=True,
            connected=True,
            detail="playing" if playback and playback.is_playing else "ready",
            playback=playback,
        )

    async def web_player_access_token(self) -> str:
        return await self._access_token()

    async def set_web_player_device(self, device_id: str) -> SpotifyStatus:
        normalized_device_id = device_id.strip()
        if not normalized_device_id:
            raise ValueError("Spotify Web Playback device ID cannot be empty")
        self._web_player_device_id = normalized_device_id
        return await self.status()

    async def play_query(
        self,
        query: str,
        *,
        prefer_artist: bool = False,
    ) -> SpotifyStatus:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("Spotify query cannot be empty")
        if prefer_artist:
            artist_payload = await self._api_request(
                "GET",
                "/search?" + urlencode(
                    {"q": normalized_query, "type": "artist", "limit": 1, "market": self._market}
                ),
            )
            artists = artist_payload.get("artists", {}).get("items", [])
            if artists:
                artist = artists[0]
                artist_uri = str(artist.get("uri", "")).strip()
                if artist_uri:
                    await self._api_request(
                        "PUT",
                        self._play_path(),
                        {"context_uri": artist_uri},
                        allow_empty=True,
                    )
                    return await self._wait_for_playback()
        payload = await self._api_request(
            "GET",
            "/search?" + urlencode(
                {"q": normalized_query, "type": "track", "limit": 1, "market": self._market}
            ),
        )
        tracks = payload.get("tracks", {}).get("items", [])
        if not tracks:
            raise SpotifyError("No Spotify track found for the requested music")
        track = tracks[0]
        await self._api_request(
            "PUT",
            self._play_path(),
            {"uris": [str(track["uri"])]},
            allow_empty=True,
        )
        return await self._wait_for_playback()

    async def _wait_for_playback(self) -> SpotifyStatus:
        """Wait briefly for Spotify to expose the playback started by its API."""
        status = await self.status()
        for _ in range(self.PLAYBACK_CONFIRMATION_ATTEMPTS - 1):
            if status.playback is not None and status.playback.is_playing:
                return status
            await asyncio.sleep(self.PLAYBACK_CONFIRMATION_DELAY_SECONDS)
            status = await self.status()
        return status

    async def control(self, action: str) -> SpotifyStatus:
        if action in {"volume_up", "volume_down"}:
            return await self._adjust_volume(action)
        endpoints = {
            "pause": ("PUT", "/me/player/pause"),
            "resume": ("PUT", "/me/player/play"),
            "next": ("POST", "/me/player/next"),
            "previous": ("POST", "/me/player/previous"),
        }
        if action not in endpoints:
            raise ValueError(f"Unsupported Spotify control: {action}")
        method, path = endpoints[action]
        await self._api_request(method, path, allow_empty=True)
        if action == "pause":
            return await self._wait_for_playback_state(False)
        if action == "resume":
            return await self._wait_for_playback_state(True)
        return await self.status()

    async def _adjust_volume(self, action: str) -> SpotifyStatus:
        status = await self.status()
        playback = status.playback
        if playback is None or playback.volume_percent is None:
            raise SpotifyError("Spotify has no active playback device with volume information")
        if not playback.supports_volume:
            raise SpotifyError("The active Spotify device does not support remote volume")
        delta = 10 if action == "volume_up" else -10
        volume_percent = min(max(playback.volume_percent + delta, 0), 100)
        await self._api_request(
            "PUT",
            "/me/player/volume?" + urlencode({"volume_percent": volume_percent}),
            allow_empty=True,
        )
        return await self.status()

    async def disconnect(self) -> SpotifyStatus:
        self._token_store.clear()
        self._pending_state = ""
        self._pending_verifier = ""
        self._web_player_device_id = ""
        return SpotifyStatus(
            configured=bool(self._client_id),
            detail="not_connected" if self._client_id else "not_configured",
        )

    async def _current_playback(self) -> SpotifyPlayback | None:
        payload = await self._api_request("GET", "/me/player", allow_empty=True)
        if not payload:
            return None
        item = payload.get("item") if isinstance(payload, dict) else None
        if not isinstance(item, dict):
            return None
        artists = item.get("artists", [])
        artist_names = ", ".join(
            str(artist.get("name", ""))
            for artist in artists
            if isinstance(artist, dict) and artist.get("name")
        )
        album = item.get("album") if isinstance(item.get("album"), dict) else {}
        artwork_source = album if album else item
        images = artwork_source.get("images", []) if isinstance(artwork_source, dict) else []
        album_art_url = next(
            (
                str(image.get("url", ""))
                for image in images
                if isinstance(image, dict) and str(image.get("url", "")).startswith("https://")
            ),
            "",
        )
        external_urls = item.get("external_urls", {})
        spotify_url = (
            str(external_urls.get("spotify", ""))
            if isinstance(external_urls, dict)
            else ""
        )
        device = payload.get("device") if isinstance(payload.get("device"), dict) else {}
        raw_volume = device.get("volume_percent")
        volume_percent = (
            min(max(int(raw_volume), 0), 100)
            if isinstance(raw_volume, (int, float)) and not isinstance(raw_volume, bool)
            else None
        )
        return SpotifyPlayback(
            track_id=str(item.get("id", "")),
            title=str(item.get("name", "")),
            artists=artist_names,
            album=str(album.get("name", "")),
            album_art_url=album_art_url,
            spotify_url=spotify_url,
            is_playing=bool(payload.get("is_playing", False)),
            progress_ms=int(payload.get("progress_ms") or 0),
            duration_ms=int(item.get("duration_ms") or 0),
            device_name=str(device.get("name", "")),
            volume_percent=volume_percent,
            supports_volume=bool(device.get("supports_volume", False)),
        )

    async def _wait_for_playback_state(self, is_playing: bool) -> SpotifyStatus:
        status = await self.status()
        for _ in range(self.PLAYBACK_CONFIRMATION_ATTEMPTS - 1):
            actual_state = bool(status.playback and status.playback.is_playing)
            if actual_state == is_playing:
                return status
            await asyncio.sleep(self.PLAYBACK_CONFIRMATION_DELAY_SECONDS)
            status = await self.status()
        return status

    def _play_path(self) -> str:
        if not self._web_player_device_id:
            return "/me/player/play"
        return "/me/player/play?" + urlencode(
            {"device_id": self._web_player_device_id}
        )

    async def _token_request(self, parameters: dict[str, str]) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._request_json,
            "POST",
            self.TOKEN_URL,
            parameters,
            None,
            False,
        )

    async def _api_request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        allow_empty: bool = False,
    ) -> dict[str, Any]:
        token = await self._access_token()
        return await asyncio.to_thread(
            self._request_json,
            method,
            f"{self.API_URL}{path}",
            payload,
            token,
            allow_empty,
        )

    async def _access_token(self) -> str:
        if not self._client_id:
            raise SpotifyNotConfiguredError("Spotify Client ID is not configured")
        token = self._token_store.load()
        if token is None:
            raise SpotifyNotConnectedError("Spotify is not connected")
        if token.expires_at > time() + 60:
            return token.access_token
        response = await self._token_request(
            {
                "grant_type": "refresh_token",
                "refresh_token": token.refresh_token,
                "client_id": self._client_id,
            }
        )
        self._save_token_response(response, fallback_refresh_token=token.refresh_token)
        refreshed = self._token_store.load()
        if refreshed is None:
            raise SpotifyNotConnectedError("Spotify token refresh failed")
        return refreshed.access_token

    def _save_token_response(
        self,
        response: dict[str, Any],
        *,
        fallback_refresh_token: str = "",
    ) -> None:
        access_token = str(response.get("access_token", "")).strip()
        refresh_token = str(response.get("refresh_token", fallback_refresh_token)).strip()
        expires_in = int(response.get("expires_in", 3600))
        if not access_token or not refresh_token:
            raise SpotifyError("Spotify did not return renewable credentials")
        self._token_store.save(
            SpotifyToken(access_token, refresh_token, time() + max(expires_in, 60)),
        )

    @staticmethod
    def _request_json(
        method: str,
        url: str,
        payload: dict[str, object] | None,
        access_token: str | None,
        allow_empty: bool,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        body: bytes | None = None
        if access_token is None:
            if payload is not None:
                headers["Content-Type"] = "application/x-www-form-urlencoded"
                body = urlencode(payload).encode("utf-8")
        else:
            headers["Authorization"] = f"Bearer {access_token}"
            if payload is not None:
                headers["Content-Type"] = "application/json"
                body = json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=20) as response:
                raw = response.read()
        except HTTPError as exc:
            if exc.code == 401:
                raise SpotifyNotConnectedError("Spotify authorization expired") from exc
            detail = exc.read().decode("utf-8", errors="replace")[:240]
            raise SpotifyError(f"Spotify request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise SpotifyError(f"Spotify is unreachable: {exc.reason}") from exc
        if not raw:
            return {} if allow_empty else {}
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise SpotifyError("Spotify returned invalid data") from exc
        if not isinstance(parsed, dict):
            raise SpotifyError("Spotify returned an invalid response")
        return parsed
