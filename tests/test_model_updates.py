from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import threading
import unittest

from fastapi import FastAPI
import yaml

from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import create_app
from ugassistant.config import AppSettings
from ugassistant.services.model_updates import ModelUpdateService, OllamaManifest


LOCKED_DIGEST = "a" * 64
NEW_DIGEST = "b" * 64


def manifest(digest: str, size_bytes: int = 123) -> OllamaManifest:
    return OllamaManifest(digest=digest, size_bytes=size_bytes)


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
            remote_manifest_loader=lambda _url, _tag: manifest(LOCKED_DIGEST),
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
            remote_manifest_loader=lambda _url, _tag: manifest(LOCKED_DIGEST),
            pull_model=pull,
        )

        result = await service.check_and_update()

        self.assertEqual(result["status"], "updated")
        self.assertTrue(result["llm"]["updated"])  # type: ignore[index]

    async def test_updates_the_lock_before_pulling_a_new_remote_revision(self) -> None:
        installed_digest = LOCKED_DIGEST
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_path = Path(temporary_directory) / "models.lock.yaml"
            lock_path.write_text(yaml.safe_dump(model_lock(), sort_keys=False), encoding="utf-8")

            def pull(_tag: str) -> None:
                nonlocal installed_digest
                installed_digest = NEW_DIGEST

            service = ModelUpdateService(
                model_lock=model_lock(),
                ollama_base_url="http://127.0.0.1:11434",
                llm_model="gemma3:4b",
                fixed_model_paths={},
                model_lock_path=lock_path,
                local_digest_loader=lambda _tag: installed_digest,
                remote_manifest_loader=lambda _url, _tag: manifest(NEW_DIGEST, 987),
                pull_model=pull,
            )

            result = await service.check_and_update()
            persisted = yaml.safe_load(lock_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "updated")
        self.assertTrue(result["llm"]["updated"])  # type: ignore[index]
        self.assertEqual(persisted["models"][0]["sha256"], NEW_DIGEST)
        self.assertEqual(persisted["models"][0]["size"], "987 bytes managed_by_ollama")

    async def test_restores_the_previous_lock_when_the_update_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_path = Path(temporary_directory) / "models.lock.yaml"
            original = yaml.safe_dump(model_lock(), sort_keys=False)
            lock_path.write_text(original, encoding="utf-8")
            service = ModelUpdateService(
                model_lock=model_lock(),
                ollama_base_url="http://127.0.0.1:11434",
                llm_model="gemma3:4b",
                fixed_model_paths={},
                model_lock_path=lock_path,
                local_digest_loader=lambda _tag: LOCKED_DIGEST,
                remote_manifest_loader=lambda _url, _tag: manifest(NEW_DIGEST, 987),
                pull_model=lambda _tag: (_ for _ in ()).throw(RuntimeError("network")),
            )

            result = await service.check_and_update()

            self.assertEqual(lock_path.read_text(encoding="utf-8"), original)

        self.assertEqual(result["status"], "error")
        self.assertFalse(result["llm"]["updated"])  # type: ignore[index]

    async def test_rejects_parallel_update_requests(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def remote(_url: str, _tag: str) -> OllamaManifest:
            started.set()
            release.wait()
            return manifest(LOCKED_DIGEST)

        service = ModelUpdateService(
            model_lock=model_lock(),
            ollama_base_url="http://127.0.0.1:11434",
            llm_model="gemma3:4b",
            fixed_model_paths={},
            local_digest_loader=lambda _tag: LOCKED_DIGEST,
            remote_manifest_loader=remote,
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
