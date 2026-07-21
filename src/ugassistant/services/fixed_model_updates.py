from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping
import urllib.request

import yaml


class FixedModelUpdateService:
    """Transactionally updates fixed model files and restores a tested baseline."""

    def __init__(
        self,
        *,
        project_root: Path,
        model_lock_path: Path,
        model_paths: Mapping[str, Path],
        downloader: Callable[[str, Path], None] | None = None,
        functional_check: Callable[[set[str]], None] | None = None,
        on_progress: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._project_root = project_root
        self._model_lock_path = model_lock_path
        self._model_paths = dict(model_paths)
        self._downloader = downloader or self._download
        self._functional_check = functional_check or self._run_functional_checks
        self._on_progress = on_progress

    def set_progress_listener(
        self,
        listener: Callable[[dict[str, object]], None] | None,
    ) -> None:
        self._on_progress = listener

    def check_and_update(self) -> list[dict[str, object]]:
        lock = self._load_lock()
        models = [
            model
            for model in lock.get("models", [])
            if isinstance(model, dict) and model.get("logical_name") != "llm"
        ]
        self._notify("fixed_models", "validating", "Validando los modelos instalados")
        try:
            self._functional_check(set())
        except Exception as exc:
            self._notify(
                "fixed_models",
                "error",
                "No se pudo validar la instalacion actual",
            )
            return [
                {
                    "logical_name": "fixed_models",
                    "state": "error",
                    "detail": f"baseline_check_failed:{exc}",
                }
            ]
        candidates: list[tuple[dict[str, Any], Path, str, int, str]] = []
        results: list[dict[str, object]] = []
        staging_root = self._project_root / "data" / "model-staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="update-", dir=staging_root) as directory:
            staging = Path(directory)
            for model in models:
                logical_name = str(model["logical_name"])
                target = self._model_paths.get(logical_name)
                if target is None:
                    results.append({"logical_name": logical_name, "state": "not_managed"})
                    self._notify(logical_name, "not_managed", "No esta gestionado")
                    continue
                candidate = staging / str(model["file_name"])
                source = str(model.get("update_url") or model["official_url"])
                installed_version = str(model.get("version_or_tag", "desconocida"))
                self._notify(
                    logical_name,
                    "downloading",
                    "Descargando revision oficial",
                    installed_version=installed_version,
                )
                try:
                    self._downloader(source, candidate)
                except Exception as exc:
                    self._notify(
                        logical_name,
                        "error",
                        "No se pudo descargar la revision oficial",
                        installed_version=installed_version,
                    )
                    self._notify(
                        "fixed_models",
                        "error",
                        "No se completaron las descargas de modelos fijos",
                    )
                    return [
                        *results,
                        {
                            "logical_name": logical_name,
                            "state": "error",
                            "detail": f"download_failed:{exc}",
                        },
                    ]
                digest = self._file_sha256(candidate)
                found_version = f"sha256:{digest[:16]}"
                locked_digest = str(model.get("sha256", "")).lower()
                target_digest = (
                    self._file_sha256(target) if target.is_file() else None
                )
                if digest == locked_digest and target_digest == locked_digest:
                    results.append({"logical_name": logical_name, "state": "up_to_date"})
                    self._notify(
                        logical_name,
                        "up_to_date",
                        "Ya esta actualizado",
                        installed_version=installed_version,
                        found_version=found_version,
                    )
                    continue
                state = "updated" if digest != locked_digest else "repaired"
                candidates.append(
                    (model, candidate, digest, candidate.stat().st_size, state)
                )

            if not candidates:
                self._notify(
                    "fixed_models",
                    "up_to_date",
                    "Todos los modelos fijos estan al dia",
                )
                return results
            changed = {str(model["logical_name"]) for model, *_ in candidates}
            lock_backup = self._model_lock_path.read_bytes()
            backups = self._backup_current_files(candidates)
            try:
                for model, _candidate, digest, _size, _state in candidates:
                    self._notify(
                        str(model["logical_name"]),
                        "installing",
                        "Activando la revision descargada",
                        installed_version=str(model.get("version_or_tag", "desconocida")),
                        found_version=f"sha256:{digest[:16]}",
                    )
                self._write_updated_lock(lock, candidates)
                for model, candidate, _digest, _size, _state in candidates:
                    target = self._model_paths[str(model["logical_name"])]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(candidate, target)
                for model, _candidate, digest, _size, _state in candidates:
                    self._notify(
                        str(model["logical_name"]),
                        "validating",
                        "Probando la revision instalada",
                        found_version=f"sha256:{digest[:16]}",
                    )
                self._functional_check(changed)
            except Exception as exc:
                self._restore_files(backups)
                self._restore_lock(lock_backup)
                rolled_back = [
                    *results,
                    *[
                        {
                            "logical_name": str(model["logical_name"]),
                            "state": "rolled_back",
                            "detail": str(exc),
                        }
                        for model, *_ in candidates
                    ],
                ]
                for item in rolled_back:
                    if item.get("state") == "rolled_back":
                        self._notify(
                            str(item["logical_name"]),
                            "rolled_back",
                            "Prueba fallida; se ha restaurado la version anterior",
                        )
                self._notify(
                    "fixed_models",
                    "rolled_back",
                    "Se han restaurado los modelos fijos anteriores",
                )
                return rolled_back
            completed = [
                *results,
                *[
                    {"logical_name": str(model["logical_name"]), "state": state}
                    for model, _candidate, _digest, _size, state in candidates
                ],
            ]
            for model, _candidate, digest, _size, state in candidates:
                self._notify(
                    str(model["logical_name"]),
                    state,
                    "Actualizacion terminada" if state == "updated" else "Archivo reparado",
                    found_version=f"sha256:{digest[:16]}",
                )
            group_state = "updated" if any(
                item.get("state") in {"updated", "repaired"} for item in completed
            ) else "up_to_date"
            self._notify(
                "fixed_models",
                group_state,
                "Modelos fijos actualizados" if group_state == "updated" else "Todos los modelos fijos estan al dia",
            )
            return completed

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

    def _load_lock(self) -> dict[str, Any]:
        with self._model_lock_path.open("r", encoding="utf-8") as file:
            lock = yaml.safe_load(file) or {}
        if not isinstance(lock, dict):
            raise RuntimeError("model_lock_invalid")
        return lock

    def _backup_current_files(
        self,
        candidates: list[tuple[dict[str, Any], Path, str, int, str]],
    ) -> dict[Path, Path | None]:
        backup_root = self._project_root / "data" / "model-backups" / "pending"
        if backup_root.exists():
            shutil.rmtree(backup_root)
        backups: dict[Path, Path | None] = {}
        for model, _candidate, _digest, _size, _state in candidates:
            target = self._model_paths[str(model["logical_name"])]
            if not target.is_file():
                backups[target] = None
                continue
            backup = backup_root / str(model["logical_name"]) / target.name
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup)
            backups[target] = backup
        return backups

    def _restore_files(self, backups: Mapping[Path, Path | None]) -> None:
        for target, backup in backups.items():
            if backup is None:
                target.unlink(missing_ok=True)
            elif backup.is_file():
                shutil.copy2(backup, target)

    def _write_updated_lock(
        self,
        lock: dict[str, Any],
        candidates: list[tuple[dict[str, Any], Path, str, int, str]],
    ) -> None:
        updates = {
            str(model["logical_name"]): (digest, size)
            for model, _candidate, digest, size, _state in candidates
        }
        for model in lock.get("models", []):
            if not isinstance(model, dict):
                continue
            update = updates.get(str(model.get("logical_name", "")))
            if update is None:
                continue
            digest, size = update
            model["sha256"] = digest
            model["size"] = f"{size} bytes"
            model["version_or_tag"] = f"auto-sha256:{digest[:16]}"
            if model.get("update_url"):
                model["official_url"] = str(model["update_url"])
        temporary = self._model_lock_path.with_suffix(self._model_lock_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(lock, file, allow_unicode=True, sort_keys=False)
        os.replace(temporary, self._model_lock_path)

    def _restore_lock(self, contents: bytes) -> None:
        temporary = self._model_lock_path.with_suffix(self._model_lock_path.suffix + ".restore")
        temporary.write_bytes(contents)
        os.replace(temporary, self._model_lock_path)

    def _run_functional_checks(self, changed: set[str]) -> None:
        speech_models = {"stt", "tts", "tts_config", "tts_fr", "tts_fr_config"}
        vision_models = {"face_detection", "palm_detection", "hand_pose"}
        if not changed or changed & speech_models:
            self._run_command([sys.executable, "scripts/check_stt.py", "--language", "es"])
            self._run_command([sys.executable, "scripts/check_stt.py", "--language", "fr"])
        if not changed or changed & vision_models:
            self._validate_vision_models()

    def _run_command(self, command: list[str]) -> None:
        completed = subprocess.run(
            command,
            cwd=self._project_root,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=240,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"functional_check_failed:{detail[-500:]}")

    def _validate_vision_models(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("functional_check_failed:opencv_missing") from exc
        face = self._model_paths["face_detection"]
        palm = self._model_paths["palm_detection"]
        hand = self._model_paths["hand_pose"]
        cv2.FaceDetectorYN.create(str(face), "", (320, 320), 0.75, 0.3, 5000)
        cv2.dnn.readNet(str(palm))
        cv2.dnn.readNet(str(hand))

    @staticmethod
    def _download(url: str, target: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": "UGAssistant-model-updater/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            with target.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
