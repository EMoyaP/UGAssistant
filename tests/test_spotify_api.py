from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from typing import Any, Callable

from fastapi import FastAPI

from ugassistant.adapters.preferences import YAMLPreferenceStore
from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedLLMAdapter,
    SimulatedSpotifyAdapter,
    SimulatedSTTAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import (
    SpotifyConfigurationRequest,
    create_app,
)
from ugassistant.config import AppSettings


def route_endpoint(app: FastAPI, path: str, method: str) -> Callable[..., Any]:
    return next(
        route.endpoint
        for route in app.routes
        if route.path == path and method in (route.methods or set())
    )  # type: ignore[attr-defined]


class SpotifyAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_saves_the_client_id_without_storing_tokens_in_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = YAMLPreferenceStore(root / "data" / "preferences.yaml")
            spotify = SimulatedSpotifyAdapter()
            app = create_app(
                AppSettings(project_root=root),
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
                SimulatedSTTAdapter(),
                store,
                SimulatedLLMAdapter(),
                spotify,
            )

            async with app.router.lifespan_context(app):
                update = route_endpoint(app, "/api/spotify/config", "PUT")
                status = await update(
                    SpotifyConfigurationRequest(client_id="spotify-client-id")
                )

            self.assertTrue(status["configured"])
            self.assertEqual(store.load().spotify_client_id, "spotify-client-id")  # type: ignore[union-attr]
            self.assertFalse((root / "data" / "spotify.tokens.json").exists())
