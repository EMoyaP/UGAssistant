from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Callable, Sequence


class ApplicationUpdateBusyError(RuntimeError):
    """Raised when an explicit application update is already in progress."""


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class ApplicationUpdateService:
    """Updates a clean checkout from its configured Git remote safely."""

    def __init__(
        self,
        *,
        project_root: Path,
        remote: str = "origin",
        branch: str = "main",
        command_runner: Callable[[Sequence[str]], CommandResult] | None = None,
    ) -> None:
        self._project_root = project_root
        self._remote = remote
        self._branch = branch
        self._command_runner = command_runner or self._run_command
        self._lock = asyncio.Lock()

    async def check_and_update(self) -> dict[str, object]:
        if self._lock.locked():
            raise ApplicationUpdateBusyError("application_update_in_progress")
        async with self._lock:
            return await asyncio.to_thread(self._check_and_update)

    def _check_and_update(self) -> dict[str, object]:
        self._require_clean_worktree()
        installed = self._revision("HEAD")
        self._run_git("fetch", self._remote, self._branch)
        remote_revision = self._revision(f"{self._remote}/{self._branch}")
        if installed == remote_revision:
            return self._result(
                "up_to_date",
                installed,
                remote_revision,
                "UGAssistant ya esta actualizado.",
                restart_required=False,
            )
        if self._is_ancestor(remote_revision, installed):
            return self._result(
                "local_ahead",
                installed,
                remote_revision,
                "La copia local contiene commits que el remoto todavia no tiene.",
                restart_required=False,
            )
        if not self._is_ancestor(installed, remote_revision):
            return self._result(
                "diverged",
                installed,
                remote_revision,
                "La copia local y origin/main han divergido; no se ha aplicado ningun cambio.",
                restart_required=False,
            )
        self._run_git("pull", "--ff-only", self._remote, self._branch)
        updated = self._revision("HEAD")
        if updated != remote_revision:
            raise RuntimeError("git_pull_did_not_reach_fetched_revision")
        return self._result(
            "updated",
            installed,
            updated,
            "UGAssistant se ha actualizado. Reinicia la aplicacion para cargar el codigo nuevo.",
            restart_required=True,
        )

    def _require_clean_worktree(self) -> None:
        status = self._run_git("status", "--porcelain")
        if status.stdout.strip():
            raise RuntimeError("git_worktree_dirty")

    def _revision(self, reference: str) -> str:
        return self._run_git("rev-parse", "--verify", reference).stdout.strip()

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = self._run_git("merge-base", "--is-ancestor", ancestor, descendant, allow_failure=True)
        if result.returncode in {0, 1}:
            return result.returncode == 0
        raise RuntimeError(self._command_error(result))

    def _run_git(self, *arguments: str, allow_failure: bool = False) -> CommandResult:
        result = self._command_runner(("git", *arguments))
        if result.returncode != 0 and not allow_failure:
            raise RuntimeError(self._command_error(result))
        return result

    @staticmethod
    def _command_error(result: CommandResult) -> str:
        detail = (result.stderr or result.stdout).strip()
        return f"git_command_failed:{detail or result.returncode}"

    def _run_command(self, command: Sequence[str]) -> CommandResult:
        try:
            completed = subprocess.run(
                command,
                cwd=self._project_root,
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"git_command_failed:{exc}") from exc
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)

    @staticmethod
    def _result(
        state: str,
        installed_revision: str,
        remote_revision: str,
        message: str,
        *,
        restart_required: bool,
    ) -> dict[str, object]:
        return {
            "state": state,
            "installed_revision": installed_revision,
            "remote_revision": remote_revision,
            "restart_required": restart_required,
            "message": message,
        }
