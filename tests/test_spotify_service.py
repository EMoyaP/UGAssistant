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

        stopped = await service.stop()
        self.assertFalse(stopped.playback is not None and stopped.playback.is_playing)
        self.assertEqual(adapter.controls, ["pause"])

    async def test_disconnect_removes_the_active_connection(self) -> None:
        adapter = SimulatedSpotifyAdapter()
        service = SpotifyService(adapter)
        service.configure("spotify-client-id")
        await service.complete_authorization("code", "state")

        status = await service.disconnect()

        self.assertTrue(status.configured)
        self.assertFalse(status.connected)
