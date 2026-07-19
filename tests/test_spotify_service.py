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

        playing = await service.play_query("Queen")
        self.assertTrue(playing.playback is not None and playing.playback.is_playing)
        self.assertEqual(adapter.played_queries, ["Queen"])
        self.assertEqual(adapter.played_query_preferences, [False])

        await service.play_query("Madonna", prefer_artist=True)
        self.assertEqual(adapter.played_query_preferences, [False, True])

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
