from __future__ import annotations

from pathlib import Path
import os
import tempfile
import unittest

from ugassistant.adapters.spotify import LocalSpotifyTokenStore, SpotifyToken


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
