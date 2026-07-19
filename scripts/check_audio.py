from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ugassistant.adapters.portaudio import PortAudioAdapter  # noqa: E402


async def enumerate_audio() -> dict[str, object]:
    devices = await PortAudioAdapter().list_devices()
    inputs = [device.to_dict() for device in devices if device.kind == "input"]
    outputs = [device.to_dict() for device in devices if device.kind == "output"]
    return {
        "ok": bool(inputs) and bool(outputs),
        "inputs": inputs,
        "outputs": outputs,
    }


def main() -> int:
    try:
        payload = asyncio.run(enumerate_audio())
    except Exception as exc:
        payload = {
            "ok": False,
            "inputs": [],
            "outputs": [],
            "detail": str(exc),
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
