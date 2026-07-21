from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable, Mapping, TYPE_CHECKING
import urllib.error
import urllib.parse
import urllib.request

import yaml

if TYPE_CHECKING:
    from ugassistant.services.fixed_model_updates import FixedModelUpdateService


class ModelUpdateBusyError(RuntimeError):
    """Raised when a previous explicit model update is still running."""


@dataclass(frozen=True)
class OllamaManifest:
    digest: str
    size_bytes: int


class ModelUpdateService:
    """Updates approved local models explicitly with verification and rollback.

    The user explicitly starts the operation from the configuration modal. The
    new Ollama manifest is written to the lock before the model is pulled, and
    the old lock is restored if the download or digest verification fails.
    Fixed STT, TTS and vision assets are staged, tested locally, and restored
    from their prior working copies if a validation fails.
    """

    def __init__(
        self,
        *,
        model_lock: Mapping[str, Any],
        ollama_base_url: str,
        llm_model: str,
        fixed_model_paths: Mapping[str, Path],
        model_lock_path: Path | None = None,
        fixed_model_updater: FixedModelUpdateService | None = None,
        local_digest_loader: Callable[[str], str | None] | None = None,
        remote_manifest_loader: Callable[[str, str], OllamaManifest] | None = None,
        pull_model: Callable[[str], None] | None = None,
        backup_model: Callable[[str, str], None] | None = None,
        restore_model: Callable[[str, str], None] | None = None,
        functional_check: Callable[[str], None] | None = None,
        on_progress: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._models = tuple(
            model
            for model in model_lock.get("models", [])
            if isinstance(model, dict)
        )
        self._ollama_base_url = ollama_base_url.rstrip("/")
        self._llm_model = llm_model
        self._fixed_model_paths = dict(fixed_model_paths)
        self._model_lock_path = model_lock_path
        self._fixed_model_updater = fixed_model_updater
        self._local_digest_loader = local_digest_loader or self._load_local_digest
        self._remote_manifest_loader = (
            remote_manifest_loader or self._load_remote_manifest
        )
        self._pull_model = pull_model or self._pull_model_from_ollama
        self._backup_model = backup_model or self._copy_ollama_model
        self._restore_model = restore_model or self._copy_ollama_model
        self._functional_check = functional_check or self._run_llm_functional_check
        self._manage_ollama_backup = backup_model is not None or pull_model is None
        self._run_functional_check = functional_check is not None or pull_model is None
        self._on_progress = on_progress
        if self._fixed_model_updater is not None:
            self._fixed_model_updater.set_progress_listener(on_progress)
        self._lock = asyncio.Lock()

    def set_progress_listener(
        self,
        listener: Callable[[dict[str, object]], None] | None,
    ) -> None:
        self._on_progress = listener
        if self._fixed_model_updater is not None:
            self._fixed_model_updater.set_progress_listener(listener)

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
        local_digest = self._local_digest_loader(tag)
        self._notify(
            "llm",
            "checking",
            "Consultando la version de Gemma",
            installed_version=self._display_version(tag, local_digest),
        )
        remote_manifest: OllamaManifest | None = None
        error: str | None = None
        try:
            remote_manifest = self._remote_manifest_loader(
                str(llm["official_url"]), tag
            )
        except RuntimeError as exc:
            error = str(exc)

        llm_status = "unavailable"
        updated = False
        detail = "No se pudo consultar el registro oficial."
        if error is None and remote_manifest is not None:
            remote_version = self._display_version(tag, remote_manifest.digest)
            if remote_manifest.digest != locked_digest:
                backup_tag = self._backup_tag(tag)
                has_backup = False
                previous_lock = self._update_lock(
                    llm,
                    digest=remote_manifest.digest,
                    size_bytes=remote_manifest.size_bytes,
                )
                try:
                    if self._manage_ollama_backup and local_digest is not None:
                        self._notify(
                            "llm",
                            "backing_up",
                            "Guardando la ultima version funcional",
                            found_version=remote_version,
                        )
                        self._backup_model(tag, backup_tag)
                        has_backup = True
                    self._notify(
                        "llm",
                        "installing",
                        "Instalando la nueva version de Gemma",
                        found_version=remote_version,
                    )
                    self._pull_model(tag)
                    local_digest = self._local_digest_loader(tag)
                    if local_digest != remote_manifest.digest:
                        raise RuntimeError("ollama_updated_digest_mismatch")
                    self._notify(
                        "llm",
                        "validating",
                        "Probando Gemma localmente",
                        found_version=remote_version,
                    )
                    self._check_llm_functional(tag)
                except Exception as exc:
                    if has_backup:
                        try:
                            self._restore_model(backup_tag, tag)
                            local_digest = self._local_digest_loader(tag)
                        except Exception as restore_exc:
                            error = f"{exc}; ollama_restore_failed:{restore_exc}"
                        else:
                            error = str(exc)
                    else:
                        error = str(exc)
                    self._restore_lock(llm, previous_lock)
                    llm_status = "error"
                    detail = "Gemma no supero la prueba local; se ha restaurado la version anterior."
                    self._notify(
                        "llm",
                        "rolled_back",
                        detail,
                        installed_version=self._display_version(tag, local_digest),
                        found_version=remote_version,
                    )
                else:
                    locked_digest = remote_manifest.digest
                    llm_status = "updated"
                    updated = True
                    detail = "Gemma se ha actualizado a la revision oficial mas reciente."
                    self._notify("llm", "updated", detail, found_version=remote_version)
            elif local_digest == locked_digest:
                llm_status = "up_to_date"
                detail = "Gemma coincide con el modelo bloqueado."
                self._notify(
                    "llm",
                    "up_to_date",
                    detail,
                    installed_version=self._display_version(tag, local_digest),
                    found_version=remote_version,
                )
            else:
                try:
                    self._notify(
                        "llm",
                        "installing",
                        "Reparando Gemma con la version bloqueada",
                        found_version=remote_version,
                    )
                    self._pull_model(tag)
                    local_digest = self._local_digest_loader(tag)
                except RuntimeError as exc:
                    error = str(exc)
                    llm_status = "error"
                    detail = "No se pudo instalar el modelo bloqueado."
                else:
                    if local_digest == locked_digest:
                        try:
                            self._check_llm_functional(tag)
                        except Exception as exc:
                            error = str(exc)
                            llm_status = "error"
                            detail = "Gemma no supero la prueba local tras instalarse."
                        else:
                            llm_status = "updated"
                            updated = True
                            detail = "Gemma se ha actualizado al modelo bloqueado."
                            self._notify("llm", "updated", detail, found_version=remote_version)
                    else:
                        llm_status = "error"
                        detail = "Ollama no devolvio el SHA-256 bloqueado tras actualizar."

        fixed_models = (
            self._fixed_model_updater.check_and_update()
            if self._fixed_model_updater is not None
            else [
                self._verify_fixed_model(model)
                for model in self._models
                if model.get("logical_name") != "llm"
            ]
        )
        fixed_rollback = any(
            item.get("state") == "rolled_back"
            for item in fixed_models
            if isinstance(item, dict)
        )
        fixed_error = any(
            item.get("state") == "error"
            for item in fixed_models
            if isinstance(item, dict)
        )
        if fixed_rollback:
            detail = "Los modelos fijos no superaron las pruebas y se han restaurado."
        elif fixed_error:
            detail = "No se pudo descargar o validar un modelo fijo; no se ha activado ningun cambio."
        return {
            "status": (
                "error"
                if fixed_rollback
                or fixed_error
                or (error is not None and llm_status == "error")
                else llm_status
            ),
            "message": detail,
            "llm": {
                "model": tag,
                "state": llm_status,
                "locked_digest": locked_digest,
                "installed_digest": local_digest,
                "remote_digest": (
                    remote_manifest.digest if remote_manifest is not None else None
                ),
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

    def _notify(
        self,
        logical_name: str,
        state: str,
        message: str,
        *,
        installed_version: str | None = None,
        found_version: str | None = None,
    ) -> None:
        if self._on_progress is None:
            return
        event: dict[str, object] = {
            "logical_name": logical_name,
            "state": state,
            "message": message,
        }
        if installed_version is not None:
            event["installed_version"] = installed_version
        if found_version is not None:
            event["found_version"] = found_version
        self._on_progress(event)

    @staticmethod
    def _display_version(tag: str, digest: str | None) -> str:
        return f"{tag} ({digest[:16]})" if digest else f"{tag} (no instalada)"

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
    def _load_remote_manifest(official_url: str, tag: str) -> OllamaManifest:
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
        try:
            payload = json.loads(manifest.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("ollama_registry_invalid_manifest") from exc
        layers = payload.get("layers", []) if isinstance(payload, dict) else []
        if not isinstance(layers, list):
            raise RuntimeError("ollama_registry_invalid_manifest")
        size_bytes = sum(
            int(layer.get("size", 0))
            for layer in layers
            if isinstance(layer, dict)
        )
        if size_bytes <= 0:
            raise RuntimeError("ollama_registry_invalid_manifest")
        return OllamaManifest(
            digest=hashlib.sha256(manifest).hexdigest(),
            size_bytes=size_bytes,
        )

    def _update_lock(
        self,
        model: Mapping[str, Any],
        *,
        digest: str,
        size_bytes: int,
    ) -> tuple[str, str, bytes | None]:
        previous_digest = self._required_digest(model)
        previous_size = str(model.get("size", ""))
        previous_contents = (
            self._model_lock_path.read_bytes()
            if self._model_lock_path is not None and self._model_lock_path.is_file()
            else None
        )
        self._set_model_lock_values(model, digest, size_bytes)
        if self._model_lock_path is not None:
            self._write_model_lock_file(digest, size_bytes)
        return previous_digest, previous_size, previous_contents

    def _restore_lock(
        self,
        model: Mapping[str, Any],
        previous: tuple[str, str, bytes | None],
    ) -> None:
        previous_digest, previous_size, previous_contents = previous
        self._set_model_lock_values(model, previous_digest, None, previous_size)
        if self._model_lock_path is not None and previous_contents is not None:
            temporary = self._model_lock_path.with_suffix(
                self._model_lock_path.suffix + ".restore"
            )
            temporary.write_bytes(previous_contents)
            os.replace(temporary, self._model_lock_path)

    @staticmethod
    def _set_model_lock_values(
        model: Mapping[str, Any],
        digest: str,
        size_bytes: int | None,
        previous_size: str | None = None,
    ) -> None:
        if not isinstance(model, dict):
            raise RuntimeError("model_lock_invalid_entry")
        model["sha256"] = digest
        if size_bytes is not None:
            model["size"] = f"{size_bytes} bytes managed_by_ollama"
        elif previous_size is not None:
            model["size"] = previous_size

    def _write_model_lock_file(self, digest: str, size_bytes: int) -> None:
        if self._model_lock_path is None:
            return
        with self._model_lock_path.open("r", encoding="utf-8") as file:
            content = yaml.safe_load(file) or {}
        models = content.get("models", []) if isinstance(content, dict) else []
        for model in models:
            if isinstance(model, dict) and model.get("logical_name") == "llm":
                model["sha256"] = digest
                model["size"] = f"{size_bytes} bytes managed_by_ollama"
                break
        else:
            raise RuntimeError("model_lock_missing:llm")
        temporary = self._model_lock_path.with_suffix(self._model_lock_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(content, file, allow_unicode=True, sort_keys=False)
        os.replace(temporary, self._model_lock_path)

    @staticmethod
    def _pull_model_from_ollama(tag: str) -> None:
        executable = shutil.which("ollama")
        if executable is None:
            raise RuntimeError("ollama_not_found")
        try:
            completed = subprocess.run(
                [executable, "pull", tag],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=1800,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"ollama_pull_failed:{exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"ollama_pull_failed:{detail}")

    @staticmethod
    def _backup_tag(tag: str) -> str:
        safe_tag = "".join(character if character.isalnum() else "-" for character in tag)
        return f"ugassistant-{safe_tag}-last-known-good"

    @staticmethod
    def _copy_ollama_model(source: str, destination: str) -> None:
        executable = shutil.which("ollama")
        if executable is None:
            raise RuntimeError("ollama_not_found")
        try:
            completed = subprocess.run(
                [executable, "cp", source, destination],
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"ollama_copy_failed:{exc}") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"ollama_copy_failed:{detail}")

    def _run_llm_functional_check(self, tag: str) -> None:
        payload = json.dumps(
            {
                "model": tag,
                "prompt": "Responde solo con la palabra listo.",
                "stream": False,
                "options": {"temperature": 0, "num_predict": 8},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self._ollama_base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"ollama_functional_check_failed:{exc}") from exc
        answer = response_payload.get("response", "") if isinstance(response_payload, dict) else ""
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("ollama_functional_check_failed:empty_response")

    def _check_llm_functional(self, tag: str) -> None:
        if self._run_functional_check:
            self._functional_check(tag)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
