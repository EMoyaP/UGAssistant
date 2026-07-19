from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest
from typing import Any, Callable

from fastapi import FastAPI, HTTPException

from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import create_app
from ugassistant.config import AppSettings


def route_endpoint(app: FastAPI, path: str) -> Callable[..., Any]:
    return next(route.endpoint for route in app.routes if route.path == path)  # type: ignore[attr-defined]


class ShutdownApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_endpoint_calls_runtime_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(
                AppSettings(project_root=Path(temporary_directory)),
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
            )
            calls: list[str] = []
            app.state.shutdown_callback = lambda: calls.append("shutdown")

            response = await route_endpoint(app, "/api/system/shutdown")()
            await asyncio.sleep(0.2)

            self.assertEqual(response, {"status": "shutting_down"})
            self.assertEqual(calls, ["shutdown"])

    async def test_shutdown_endpoint_rejects_unmanaged_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(
                AppSettings(project_root=Path(temporary_directory)),
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
            )

            with self.assertRaises(HTTPException) as captured:
                await route_endpoint(app, "/api/system/shutdown")()

            self.assertEqual(captured.exception.status_code, 409)
