from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import threading
import unittest

from fastapi import FastAPI

from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import create_app
from ugassistant.config import AppSettings
from ugassistant.services.model_updates import ModelUpdateService


LOCKED_DIGEST = "a" * 64
NEW_DIGEST = "b" * 64


def model_lock() -> dict[str, object]:
    return {
        "models": [
            {
                "logical_name": "llm",
                "version_or_tag": "gemma3:4b",
                "sha256": LOCKED_DIGEST,
                "official_url": "https://ollama.com/library/gemma3:4b",
            },
            {
                "logical_name": "stt",
                "sha256": "c" * 64,
            },
        ]
    }


def route_endpoint(app: FastAPI, path: str):
    return next(route.endpoint for route in app.routes if route.path == path)  # type: ignore[attr-defined]


class ModelUpdateServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_locked_model_as_current_without_pulling(self) -> None:
        pulled: list[str] = []
        service = ModelUpdateService(
            model_lock=model_lock(),
            ollama_base_url="http://127.0.0.1:11434",
            llm_model="gemma3:4b",
            fixed_model_paths={},
            local_digest_loader=lambda _tag: LOCKED_DIGEST,
            remote_digest_loader=lambda _url, _tag: LOCKED_DIGEST,
            pull_model=pulled.append,
        )

        result = await service.check_and_update()

        self.assertEqual(result["status"], "up_to_date")
        self.assertFalse(result["llm"]["updated"])  # type: ignore[index]
        self.assertEqual(pulled, [])

    async def test_repairs_missing_locked_ollama_model(self) -> None:
        installed_digest: str | None = None

        def pull(_tag: str) -> None:
            nonlocal installed_digest
            installed_digest = LOCKED_DIGEST

        service = ModelUpdateService(
            model_lock=model_lock(),
            ollama_base_url="http://127.0.0.1:11434",
            llm_model="gemma3:4b",
            fixed_model_paths={},
            local_digest_loader=lambda _tag: installed_digest,
            remote_digest_loader=lambda _url, _tag: LOCKED_DIGEST,
            pull_model=pull,
        )

        result = await service.check_and_update()

        self.assertEqual(result["status"], "updated")
        self.assertTrue(result["llm"]["updated"])  # type: ignore[index]

    async def test_reports_new_remote_revision_without_replacing_the_lock(self) -> None:
        pulled: list[str] = []
        service = ModelUpdateService(
            model_lock=model_lock(),
            ollama_base_url="http://127.0.0.1:11434",
            llm_model="gemma3:4b",
            fixed_model_paths={},
            local_digest_loader=lambda _tag: LOCKED_DIGEST,
            remote_digest_loader=lambda _url, _tag: NEW_DIGEST,
            pull_model=pulled.append,
        )

        result = await service.check_and_update()

        self.assertEqual(result["status"], "update_available")
        self.assertEqual(pulled, [])

    async def test_rejects_parallel_update_requests(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def remote(_url: str, _tag: str) -> str:
            started.set()
            release.wait()
            return LOCKED_DIGEST

        service = ModelUpdateService(
            model_lock=model_lock(),
            ollama_base_url="http://127.0.0.1:11434",
            llm_model="gemma3:4b",
            fixed_model_paths={},
            local_digest_loader=lambda _tag: LOCKED_DIGEST,
            remote_digest_loader=remote,
            pull_model=lambda _tag: None,
        )
        first = asyncio.create_task(service.check_and_update())
        await asyncio.to_thread(started.wait)
        with self.assertRaisesRegex(RuntimeError, "model_update_in_progress"):
            await service.check_and_update()
        release.set()
        await first


class ModelUpdateApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_endpoint_returns_the_service_result(self) -> None:
        class StubModelUpdateService:
            async def check_and_update(self) -> dict[str, object]:
                return {"status": "up_to_date", "message": "Modelo verificado."}

        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(
                AppSettings(project_root=Path(temporary_directory)),
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
                model_update_service=StubModelUpdateService(),  # type: ignore[arg-type]
            )

            response = await route_endpoint(app, "/api/models/update")()

        self.assertEqual(response["status"], "up_to_date")


if __name__ == "__main__":
    unittest.main()
