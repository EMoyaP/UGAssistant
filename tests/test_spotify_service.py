from __future__ import annotations

import unittest

from ugassistant.adapters.simulated import SimulatedSpotifyAdapter
from ugassistant.services.spotify import SpotifyService


class SpotifyServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_connects_plays_and_pauses_with_simulated_adapter(self) -> None:
        adapter = SimulatedSpotifyAdapter()
        service = SpotifyService(adapter)

        self.assertFalse((await service.refresh()).configured)
        service.configure("spotify-client-id")
        self.assertIn("accounts.spotify.com", await service.authorization_url())
        connected = await service.complete_authorization("code", "state")
        self.assertTrue(connected.connected)
        self.assertEqual(
            await service.web_player_access_token(),
            "simulated-spotify-access-token",
        )
        await service.set_web_player_device("browser-device-id")
        self.assertEqual(adapter.web_player_device_id, "browser-device-id")

        playing = await service.play_query("Queen")
        self.assertTrue(playing.playback is not None and playing.playback.is_playing)
        self.assertEqual(adapter.played_queries, ["Queen"])
        self.assertEqual(adapter.played_query_preferences, [False])

        await service.play_query("Madonna", prefer_artist=True)
        self.assertEqual(adapter.played_query_preferences, [False, True])

        latest_album = await service.play_latest_album("Shakira")
        self.assertTrue(latest_album.playback is not None and latest_album.playback.is_playing)
        self.assertEqual(adapter.played_queries[-1], "Ultimo album de Shakira")

        stopped = await service.stop()
        self.assertFalse(stopped.playback is not None and stopped.playback.is_playing)
        self.assertEqual(adapter.controls, ["pause"])

        adjusted = await service.control("volume_up")
        self.assertEqual(adjusted.playback.volume_percent if adjusted.playback else None, 60)
        self.assertEqual(adapter.controls, ["pause", "volume_up"])

    async def test_disconnect_removes_the_active_connection(self) -> None:
        adapter = SimulatedSpotifyAdapter()
        service = SpotifyService(adapter)
        service.configure("spotify-client-id")
        await service.complete_authorization("code", "state")

        status = await service.disconnect()

        self.assertTrue(status.configured)
        self.assertFalse(status.connected)
