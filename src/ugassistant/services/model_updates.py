from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping
import urllib.error
import urllib.parse
import urllib.request


class ModelUpdateBusyError(RuntimeError):
    """Raised when a previous explicit model update is still running."""


class ModelUpdateService:
    """Checks the locked local models without silently replacing them.

    Only Ollama exposes a lightweight manifest endpoint that can be compared
    before downloading. Fixed STT, TTS and vision assets are hash-verified from
    their lock entry; a new version of those files must arrive in a reviewed
    UGAssistant release with a new lock digest.
    """

    def __init__(
        self,
        *,
        model_lock: Mapping[str, Any],
        ollama_base_url: str,
        llm_model: str,
        fixed_model_paths: Mapping[str, Path],
        local_digest_loader: Callable[[str], str | None] | None = None,
        remote_digest_loader: Callable[[str, str], str] | None = None,
        pull_model: Callable[[str], None] | None = None,
    ) -> None:
        self._models = tuple(
            model
            for model in model_lock.get("models", [])
            if isinstance(model, dict)
        )
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._llm_model = llm_model
        self._fixed_model_paths = dict(fixed_model_paths)
        self._local_digest_loader = local_digest_loader or self._load_local_digest
        self._remote_digest_loader = remote_digest_loader or self._load_remote_digest
        self._pull_model = pull_model or self._pull_model_from_ollama
        self._lock = asyncio.Lock()

    async def check_and_update(self) -> dict[str, object]:
        if self._lock.locked():
            raise ModelUpdateBusyError("model_update_in_progress")
        async with self._lock:
            return await asyncio.to_thread(self._check_and_update)

    def _check_and_update(self) -> dict[str, object]:
        llm = self._model_by_name("llm")
        locked_digest = self._required_digest(llm)
        tag = str(llm["version_or_tag"])
        if self._llm_model != tag:
            raise RuntimeError("llm_config_does_not_match_model_lock")
        remote_digest: str | None = None
        error: str | None = None
        try:
            remote_digest = self._remote_digest_loader(str(llm["official_url"]), tag)
        except RuntimeError as exc:
            error = str(exc)

        local_digest = self._local_digest_loader(tag)
        llm_status = "unavailable"
        updated = False
        detail = "No se pudo consultar el registro oficial."
        if error is None and remote_digest is not None:
            if remote_digest != locked_digest:
                llm_status = "update_available"
                detail = (
                    "Hay una revision de Gemma disponible, pero requiere una actualizacion "
                    "de UGAssistant que fije y revise su nuevo SHA-256."
                )
            elif local_digest == locked_digest:
                llm_status = "up_to_date"
                detail = "Gemma coincide con el modelo bloqueado."
            else:
                try:
                    self._pull_model(tag)
                    local_digest = self._local_digest_loader(tag)
                except RuntimeError as exc:
                    error = str(exc)
                    llm_status = "error"
                    detail = "No se pudo instalar el modelo bloqueado."
                else:
                    if local_digest == locked_digest:
                        llm_status = "updated"
                        updated = True
                        detail = "Gemma se ha actualizado al modelo bloqueado."
                    else:
                        llm_status = "error"
                        detail = "Ollama no devolvio el SHA-256 bloqueado tras actualizar."

        fixed_models = [
            self._verify_fixed_model(model)
            for model in self._models
            if model.get("logical_name") != "llm"
        ]
        return {
            "status": "error" if error is not None and llm_status == "error" else llm_status,
            "message": detail,
            "llm": {
                "model": tag,
                "state": llm_status,
                "locked_digest": locked_digest,
                "installed_digest": local_digest,
                "remote_digest": remote_digest,
                "updated": updated,
                "error": error,
            },
            "fixed_models": fixed_models,
        }

    def _model_by_name(self, logical_name: str) -> Mapping[str, Any]:
        for model in self._models:
            if model.get("logical_name") == logical_name:
                return model
        raise RuntimeError(f"model_lock_missing:{logical_name}")

    @staticmethod
    def _required_digest(model: Mapping[str, Any]) -> str:
        digest = str(model.get("sha256", "")).lower()
        if len(digest) != 64:
            raise RuntimeError(f"model_lock_invalid_digest:{model.get('logical_name')}")
        return digest

    def _verify_fixed_model(self, model: Mapping[str, Any]) -> dict[str, object]:
        logical_name = str(model.get("logical_name", "unknown"))
        path = self._fixed_model_paths.get(logical_name)
        if path is None:
            return {"logical_name": logical_name, "state": "not_managed"}
        if not path.is_file():
            return {"logical_name": logical_name, "state": "missing"}
        digest = self._file_sha256(path)
        return {
            "logical_name": logical_name,
            "state": "verified" if digest == self._required_digest(model) else "mismatch",
        }

    def _load_local_digest(self, tag: str) -> str | None:
        request = urllib.request.Request(f"{self._ollama_base_url}/api/tags")
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"ollama_unreachable:{exc}") from exc
        models = payload.get("models", []) if isinstance(payload, dict) else []
        for model in models:
            if isinstance(model, dict) and model.get("name") == tag:
                digest = str(model.get("digest", "")).lower()
                return digest if len(digest) == 64 else None
        return None

    @staticmethod
    def _load_remote_digest(official_url: str, tag: str) -> str:
        parsed = urllib.parse.urlparse(official_url)
        source = parsed.path.removeprefix("/library/") or tag
        repository, separator, version = source.rpartition(":")
        if not separator or not repository or not version:
            raise RuntimeError("ollama_registry_invalid_model_tag")
        registry_url = (
            f"https://registry.ollama.ai/v2/library/{repository}/manifests/{version}"
        )
        request = urllib.request.Request(
            registry_url,
            headers={"Accept": "application/vnd.oci.image.manifest.v1+json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                manifest = response.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ollama_registry_unreachable:{exc.reason}") from exc
        return hashlib.sha256(manifest).hexdigest()

    @staticmethod
    def _pull_model_from_ollama(tag: str) -> None:
        executable = shutil.which("ollama")
        if executable is None:
            raise RuntimeError("ollama_not_found")
        completed = subprocess.run(
            [executable, "pull", tag],
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=1800,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"ollama_pull_failed:{detail}")

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
