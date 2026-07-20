from __future__ import annotations

from ugassistant.domain.iot import IoTAdapter, IoTStatus


class IoTService:
    def __init__(self, adapter: IoTAdapter) -> None:
        self._adapter = adapter
        self._status = IoTStatus()

    @property
    def status(self) -> IoTStatus:
        return self._status

    def configure(self, base_url: str, token: str) -> None:
        self._adapter.configure(base_url, token)
        self._status = IoTStatus(
            configured=bool(base_url.strip() and token.strip()),
            detail="configured" if base_url.strip() and token.strip() else "not_configured",
        )

    async def refresh(self) -> IoTStatus:
        self._status = await self._adapter.refresh()
        return self._status

    async def control(self, entity_id: str, action: str) -> IoTStatus:
        self._status = await self._adapter.control(entity_id, action)
        return self._status
