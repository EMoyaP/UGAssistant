from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class HomeAssistantError(RuntimeError):
    """Raised when the configured local Home Assistant instance cannot respond."""


@dataclass(frozen=True)
class IoTEntity:
    entity_id: str
    name: str
    domain: str
    state: str
    available: bool
    actions: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "domain": self.domain,
            "state": self.state,
            "available": self.available,
            "actions": list(self.actions),
        }


@dataclass(frozen=True)
class IoTStatus:
    configured: bool = False
    connected: bool = False
    detail: str = "not_configured"
    entities: tuple[IoTEntity, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "configured": self.configured,
            "connected": self.connected,
            "detail": self.detail,
            "entities": [entity.to_dict() for entity in self.entities],
        }


class IoTAdapter(Protocol):
    def configure(self, base_url: str, token: str) -> None: ...

    async def status(self) -> IoTStatus: ...

    async def refresh(self) -> IoTStatus: ...

    async def control(self, entity_id: str, action: str) -> IoTStatus: ...
