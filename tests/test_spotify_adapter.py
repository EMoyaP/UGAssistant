from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest
from time import time
from unittest.mock import AsyncMock, call, patch

from ugassistant.adapters.spotify import (
    LocalSpotifyTokenStore,
    SpotifyToken,
    SpotifyWebAPIAdapter,
)


class LocalSpotifyTokenStoreTests(unittest.TestCase):
    def test_round_trip_keeps_tokens_outside_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "spotify.tokens.json"
            store = LocalSpotifyTokenStore(path)
            token = SpotifyToken("access", "refresh", 1234.0)

            store.save(token)

            self.assertEqual(store.load(), token)
            self.assertTrue(path.is_file())
            if os.name == "nt":
                self.assertNotIn("access", path.read_text(encoding="utf-8"))


class SpotifyWebAPIAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_prefers_an_artist_context_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalSpotifyTokenStore(
                Path(temporary_directory) / "spotify.tokens.json"
            )
            store.save(SpotifyToken("access", "refresh", time() + 3600))
            adapter = SpotifyWebAPIAdapter(
                store,
                redirect_uri="http://127.0.0.1:8000/api/spotify/callback",
            )
            adapter.configure("spotify-client-id")
            request = AsyncMock(
                side_effect=[
                    {"artists": {"items": [{"uri": "spotify:artist:madonna"}]}},
                    {},
                    {
                        "item": {
                            "id": "track-id",
                            "name": "Like a Prayer",
                            "artists": [{"name": "Madonna"}],
                            "album": {
                                "name": "Like a Prayer",
                                "images": [{"url": "https://i.scdn.co/image/cover"}],
                            },
                            "external_urls": {"spotify": "https://open.spotify.com/track/track-id"},
                            "duration_ms": 320000,
                        },
                        "is_playing": True,
                        "progress_ms": 0,
                        "device": {
                            "name": "Spotify Desktop",
                            "volume_percent": 65,
                            "supports_volume": True,
                        },
                    },
                ]
            )
            adapter._api_request = request  # type: ignore[method-assign]

            status = await adapter.play_query("Madonna", prefer_artist=True)

            self.assertTrue(status.playback is not None and status.playback.is_playing)
            self.assertEqual(status.playback.album_art_url if status.playback else "", "https://i.scdn.co/image/cover")
            self.assertEqual(status.playback.volume_percent if status.playback else None, 65)
            self.assertEqual(
                request.await_args_list[:2],
                [
                    call(
                        "GET",
                        "/search?q=Madonna&type=artist&limit=1&market=ES",
                    ),
                    call(
                        "PUT",
                        "/me/player/play",
                        {"context_uri": "spotify:artist:madonna"},
                        allow_empty=True,
                    ),
                ],
            )

    async def test_waits_until_spotify_reports_the_started_playback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalSpotifyTokenStore(
                Path(temporary_directory) / "spotify.tokens.json"
            )
            store.save(SpotifyToken("access", "refresh", time() + 3600))
            adapter = SpotifyWebAPIAdapter(
                store,
                redirect_uri="http://127.0.0.1:8000/api/spotify/callback",
            )
            adapter.configure("spotify-client-id")
            adapter._api_request = AsyncMock(  # type: ignore[method-assign]
                side_effect=[
                    {"tracks": {"items": [{"uri": "spotify:track:like-a-prayer"}]}},
                    {},
                    {},
                    {
                        "item": {
                            "id": "track-id",
                            "name": "Like a Prayer",
                            "artists": [{"name": "Madonna"}],
                            "album": {"name": "Like a Prayer"},
                            "duration_ms": 320000,
                        },
                        "is_playing": True,
                        "progress_ms": 0,
                        "device": {"name": "Spotify Desktop"},
                    },
                ]
            )

            with patch(
                "ugassistant.adapters.spotify.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep:
                status = await adapter.play_query("Like a Prayer")

            self.assertTrue(status.playback is not None and status.playback.is_playing)
            sleep.assert_awaited_once_with(
                SpotifyWebAPIAdapter.PLAYBACK_CONFIRMATION_DELAY_SECONDS
            )

    async def test_waits_for_pause_before_publishing_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = LocalSpotifyTokenStore(
                Path(temporary_directory) / "spotify.tokens.json"
            )
            store.save(SpotifyToken("access", "refresh", time() + 3600))
            adapter = SpotifyWebAPIAdapter(
                store,
                redirect_uri="http://127.0.0.1:8000/api/spotify/callback",
            )
            adapter.configure("spotify-client-id")
            track = {
                "id": "track-id",
                "name": "Like a Prayer",
                "artists": [{"name": "Madonna"}],
                "album": {"name": "Like a Prayer"},
                "duration_ms": 320000,
            }
            adapter._api_request = AsyncMock(  # type: ignore[method-assign]
                side_effect=[
                    {},
                    {"item": track, "is_playing": True, "device": {"name": "Spotify Desktop"}},
                    {"item": track, "is_playing": False, "device": {"name": "Spotify Desktop"}},
                ]
            )

            with patch(
                "ugassistant.adapters.spotify.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep:
                status = await adapter.control("pause")

            self.assertFalse(status.playback is not None and status.playback.is_playing)
            sleep.assert_awaited_once_with(
                SpotifyWebAPIAdapter.PLAYBACK_CONFIRMATION_DELAY_SECONDS
            )
