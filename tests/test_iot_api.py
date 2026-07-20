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
    SimulatedIoTAdapter,
    SimulatedSTTAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import (
    IoTConfigurationRequest,
    IoTControlRequest,
    create_app,
)
from ugassistant.config import AppSettings


def route_endpoint(app: FastAPI, path: str, method: str) -> Callable[..., Any]:
    return next(
        route.endpoint
        for route in app.routes
        if route.path == path and method in (route.methods or set())
    )  # type: ignore[attr-defined]


class IoTAPITests(unittest.IsolatedAsyncioTestCase):
    async def test_configures_discovers_and_controls_local_iot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = YAMLPreferenceStore(root / "data" / "preferences.yaml")
            adapter = SimulatedIoTAdapter()
            app = create_app(
                AppSettings(project_root=root),
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
                SimulatedSTTAdapter(),
                store,
                iot_adapter=adapter,
            )

            async with app.router.lifespan_context(app):
                configure = route_endpoint(app, "/api/iot/config", "PUT")
                control = route_endpoint(app, "/api/iot/entities/{entity_id:path}", "POST")
                status = await configure(
                    IoTConfigurationRequest(
                        home_assistant_url="http://homeassistant.local:8123",
                        token="local-test-token",
                    )
                )
                controlled = await control(
                    "light.salon",
                    IoTControlRequest(action="turn_on"),
                )

            self.assertTrue(status["connected"])
            self.assertEqual(len(status["entities"]), 1)
            self.assertEqual(adapter.controls, [("light.salon", "turn_on")])
            self.assertTrue(controlled["connected"])
            self.assertEqual(
                store.load().home_assistant_url,  # type: ignore[union-attr]
                "http://homeassistant.local:8123",
            )
