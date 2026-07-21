from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi import FastAPI

from ugassistant.adapters.simulated import (
    SimulatedAudioAdapter,
    SimulatedCameraAdapter,
    SimulatedTTSAdapter,
)
from ugassistant.api.app import create_app
from ugassistant.config import AppSettings
from ugassistant.services.application_updates import (
    ApplicationUpdateService,
    CommandResult,
)


INSTALLED = "a" * 40
REMOTE = "b" * 40


def route_endpoint(app: FastAPI, path: str):
    return next(route.endpoint for route in app.routes if route.path == path)  # type: ignore[attr-defined]


class ApplicationUpdateServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_reports_when_origin_main_is_current(self) -> None:
        commands: list[tuple[str, ...]] = []

        def runner(command: tuple[str, ...]) -> CommandResult:
            commands.append(command)
            values = {
                ("git", "status", "--porcelain"): CommandResult(0, "", ""),
                ("git", "rev-parse", "--verify", "HEAD"): CommandResult(0, f"{INSTALLED}\n", ""),
                ("git", "fetch", "origin", "main"): CommandResult(0, "", ""),
                ("git", "rev-parse", "--verify", "origin/main"): CommandResult(0, f"{INSTALLED}\n", ""),
            }
            return values[command]

        service = ApplicationUpdateService(
            project_root=Path.cwd(), command_runner=runner
        )

        result = await service.check_and_update()

        self.assertEqual(result["state"], "up_to_date")
        self.assertFalse(result["restart_required"])
        self.assertNotIn(("git", "pull", "--ff-only", "origin", "main"), commands)

    async def test_fast_forwards_to_the_fetched_origin_revision(self) -> None:
        head_reads = iter([INSTALLED, REMOTE])
        commands: list[tuple[str, ...]] = []

        def runner(command: tuple[str, ...]) -> CommandResult:
            commands.append(command)
            if command == ("git", "status", "--porcelain"):
                return CommandResult(0, "", "")
            if command == ("git", "rev-parse", "--verify", "HEAD"):
                return CommandResult(0, f"{next(head_reads)}\n", "")
            if command == ("git", "fetch", "origin", "main"):
                return CommandResult(0, "", "")
            if command == ("git", "rev-parse", "--verify", "origin/main"):
                return CommandResult(0, f"{REMOTE}\n", "")
            if command == ("git", "merge-base", "--is-ancestor", REMOTE, INSTALLED):
                return CommandResult(1, "", "")
            if command == ("git", "merge-base", "--is-ancestor", INSTALLED, REMOTE):
                return CommandResult(0, "", "")
            if command == ("git", "pull", "--ff-only", "origin", "main"):
                return CommandResult(0, "", "")
            raise AssertionError(command)

        service = ApplicationUpdateService(
            project_root=Path.cwd(), command_runner=runner
        )

        result = await service.check_and_update()

        self.assertEqual(result["state"], "updated")
        self.assertTrue(result["restart_required"])
        self.assertIn(("git", "pull", "--ff-only", "origin", "main"), commands)

    async def test_refuses_to_pull_over_local_changes(self) -> None:
        def runner(command: tuple[str, ...]) -> CommandResult:
            self.assertEqual(command, ("git", "status", "--porcelain"))
            return CommandResult(0, " M src/ugassistant/api/app.py\n", "")

        service = ApplicationUpdateService(
            project_root=Path.cwd(), command_runner=runner
        )

        with self.assertRaisesRegex(RuntimeError, "git_worktree_dirty"):
            await service.check_and_update()


class ApplicationUpdateApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_update_endpoint_returns_the_application_update_result(self) -> None:
        class StubApplicationUpdateService:
            async def check_and_update(self) -> dict[str, object]:
                return {
                    "state": "up_to_date",
                    "installed_revision": INSTALLED,
                    "remote_revision": INSTALLED,
                    "restart_required": False,
                    "message": "UGAssistant ya esta actualizado.",
                }

        with tempfile.TemporaryDirectory() as temporary_directory:
            app = create_app(
                AppSettings(project_root=Path(temporary_directory)),
                SimulatedCameraAdapter(),
                SimulatedAudioAdapter(),
                SimulatedTTSAdapter(),
                application_update_service=StubApplicationUpdateService(),  # type: ignore[arg-type]
            )

            response = await route_endpoint(app, "/api/application/update")()

        self.assertEqual(response["state"], "up_to_date")


if __name__ == "__main__":
    unittest.main()
