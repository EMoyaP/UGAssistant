from __future__ import annotations

import asyncio
import ipaddress
import json
import os
from pathlib import Path
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ugassistant.domain.iot import HomeAssistantError, IoTEntity, IoTStatus


_ACTIONS: dict[str, tuple[str, ...]] = {
    "light": ("turn_on", "turn_off", "toggle"),
    "switch": ("turn_on", "turn_off", "toggle"),
    "fan": ("turn_on", "turn_off", "toggle"),
    "input_boolean": ("turn_on", "turn_off", "toggle"),
    "cover": ("open", "close", "stop"),
    "climate": ("turn_on", "turn_off"),
    "media_player": ("turn_on", "turn_off"),
    "scene": ("activate",),
    "script": ("activate",),
    "lock": ("lock", "unlock"),
}


class LocalHomeAssistantTokenStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> str:
        if not self._path.is_file():
            return ""
        return self._path.read_text(encoding="utf-8").strip()

    def save(self, token: str) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(token.strip(), encoding="utf-8")
        if os.name != "nt":
            self._path.chmod(0o600)


class HomeAssistantRESTAdapter:
    """Small local-network client for Home Assistant's documented REST API."""

    def __init__(self, token_store: LocalHomeAssistantTokenStore) -> None:
        self._token_store = token_store
        self._base_url = ""
        self._token = token_store.load()
        self._status = IoTStatus()

    def configure(self, base_url: str, token: str) -> None:
        normalized_url = _normalize_local_url(base_url)
        if token.strip():
            self._token_store.save(token)
            self._token = token.strip()
        self._base_url = normalized_url
        self._status = IoTStatus(
            configured=bool(self._base_url and self._token),
            detail="ready" if self._base_url and self._token else "not_configured",
        )

    async def status(self) -> IoTStatus:
        return self._status

    async def refresh(self) -> IoTStatus:
        if not self._base_url or not self._token:
            self._status = IoTStatus(detail="not_configured")
            return self._status
        try:
            payload = await asyncio.to_thread(self._request, "/api/states")
            entities = tuple(_entity_from_state(item) for item in payload if isinstance(item, dict))
            entities = tuple(entity for entity in entities if entity.actions)
            self._status = IoTStatus(
                configured=True,
                connected=True,
                detail="ready",
                entities=tuple(sorted(entities, key=lambda item: (item.domain, item.name.casefold()))),
            )
        except HomeAssistantError as exc:
            self._status = IoTStatus(configured=True, detail=str(exc))
        return self._status

    async def control(self, entity_id: str, action: str) -> IoTStatus:
        domain, separator, _ = entity_id.partition(".")
        if not separator or action not in _ACTIONS.get(domain, ()):
            raise ValueError("Unsupported IoT action")
        if action in {"open", "close", "stop"}:
            service = f"cover.{action}_cover"
        elif action in {"turn_on", "turn_off", "toggle"}:
            service = f"{domain}.{action}"
        elif action in {"lock", "unlock"}:
            service = f"lock.{action}"
        elif action == "activate":
            service = f"{domain}.turn_on"
        else:
            raise ValueError("Unsupported IoT action")
        await asyncio.to_thread(
            self._request,
            f"/api/services/{service.replace('.', '/')}",
            {"entity_id": entity_id},
        )
        return await self.refresh()

    def _request(self, path: str, payload: dict[str, str] | None = None) -> object:
        request = Request(
            f"{self._base_url}{path}",
            data=(json.dumps(payload).encode("utf-8") if payload is not None else None),
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST" if payload is not None else "GET",
        )
        try:
            with urlopen(request, timeout=4) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise HomeAssistantError(f"Home Assistant HTTP {exc.code}") from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise HomeAssistantError("Home Assistant is unreachable") from exc


def _normalize_local_url(value: str) -> str:
    parsed = urlparse(value.strip().rstrip("/"))
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Home Assistant URL must include http:// or https://")
    hostname = parsed.hostname.casefold()
    if hostname.endswith(".local") or hostname in {"localhost", "homeassistant"}:
        return parsed.geturl()
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, None)}
    except socket.gaierror as exc:
        raise ValueError("Home Assistant host cannot be resolved") from exc
    if not addresses or not all(ipaddress.ip_address(address).is_private or ipaddress.ip_address(address).is_loopback for address in addresses):
        raise ValueError("Home Assistant must be reachable on the local network")
    return parsed.geturl()


def _entity_from_state(payload: dict[str, object]) -> IoTEntity:
    entity_id = str(payload.get("entity_id", ""))
    domain = entity_id.partition(".")[0]
    attributes = payload.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    return IoTEntity(
        entity_id=entity_id,
        name=str(attributes.get("friendly_name") or entity_id),
        domain=domain,
        state=str(payload.get("state", "unknown")),
        available=str(payload.get("state", "")).casefold() not in {"unavailable", "unknown"},
        actions=_ACTIONS.get(domain, ()),
    )
