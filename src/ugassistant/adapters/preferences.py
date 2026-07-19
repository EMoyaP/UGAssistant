from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import yaml

from ugassistant.domain.preferences import UserPreferences


class YAMLPreferenceStore:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> UserPreferences | None:
        if not self._path.is_file():
            return None
        with self._path.open("r", encoding="utf-8") as source:
            payload = yaml.safe_load(source) or {}
        if not isinstance(payload, dict):
            raise ValueError("Preferences file must contain a YAML mapping")
        if int(payload.get("schema_version", 0)) != self.SCHEMA_VERSION:
            raise ValueError("Unsupported preferences schema version")
        return UserPreferences.from_dict(payload)

    def save(self, preferences: UserPreferences) -> None:
        payload: dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            **preferences.to_dict(),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        with self._lock:
            try:
                with tempfile.NamedTemporaryFile(
                    "w",
                    encoding="utf-8",
                    newline="\n",
                    dir=self._path.parent,
                    prefix=f".{self._path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    yaml.safe_dump(
                        payload,
                        temporary,
                        allow_unicode=True,
                        sort_keys=False,
                    )
                    temporary.flush()
                    os.fsync(temporary.fileno())
                os.replace(temporary_path, self._path)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()
