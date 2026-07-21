from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
import sqlite3
from pathlib import Path


class MobileAccessDeniedError(RuntimeError):
    pass


@dataclass(frozen=True)
class MobileDevice:
    access_id: str
    label: str
    last_seen: str | None
    revoked: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "access_id": self.access_id,
            "label": self.label,
            "last_seen": self.last_seen,
            "revoked": self.revoked,
            "connected": self.last_seen is not None
            and datetime.fromisoformat(self.last_seen) >= datetime.now(timezone.utc) - timedelta(seconds=75),
        }


class MobileAccessStore:
    """Persistent, revocable credentials for devices on the local network."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._initialize()

    def issue(self, base_url: str) -> dict[str, str]:
        access_id = secrets.token_urlsafe(12)
        token = secrets.token_urlsafe(32)
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO mobile_devices(access_id, token_hash, label, device_id, created_at, last_seen, revoked)
                VALUES (?, ?, ?, NULL, ?, NULL, 0)
                """,
                (access_id, self._token_hash(token), "Pendiente", self._now()),
            )
        return {
            "access_id": access_id,
            "token": token,
            "url": f"{base_url}/?access={access_id}&token={token}",
        }

    def authorize(self, access_id: str, token: str, device_id: str, label: str) -> MobileDevice:
        normalized_label = " ".join(label.split())[:48] or "Android"
        with self._connection() as connection:
            row = connection.execute(
                "SELECT token_hash, device_id, label, last_seen, revoked FROM mobile_devices WHERE access_id = ?",
                (access_id,),
            ).fetchone()
            if row is None or row[4] or not secrets.compare_digest(row[0], self._token_hash(token)):
                raise MobileAccessDeniedError("mobile_access_denied")
            registered_device = row[1]
            if registered_device is not None and not secrets.compare_digest(registered_device, device_id):
                raise MobileAccessDeniedError("mobile_device_mismatch")
            connection.execute(
                "UPDATE mobile_devices SET device_id = ?, label = ?, last_seen = ? WHERE access_id = ?",
                (device_id, normalized_label, self._now(), access_id),
            )
        return MobileDevice(access_id, normalized_label, self._now(), False)

    def list_devices(self) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT access_id, label, last_seen, revoked FROM mobile_devices ORDER BY created_at DESC"
            ).fetchall()
        return [MobileDevice(*row).to_dict() for row in rows]

    def has_active_access(self) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM mobile_devices WHERE revoked = 0 LIMIT 1"
            ).fetchone()
        return row is not None

    def revoke(self, access_id: str) -> None:
        with self._connection() as connection:
            updated = connection.execute(
                "UPDATE mobile_devices SET revoked = 1 WHERE access_id = ?",
                (access_id,),
            ).rowcount
        if not updated:
            raise KeyError(access_id)

    def _initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS mobile_devices(
                    access_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL,
                    label TEXT NOT NULL,
                    device_id TEXT,
                    created_at TEXT NOT NULL,
                    last_seen TEXT,
                    revoked INTEGER NOT NULL DEFAULT 0
                )
                """
            )

    def _connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
