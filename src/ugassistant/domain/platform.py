from __future__ import annotations

from dataclasses import dataclass
import platform
import sys


@dataclass(frozen=True)
class PlatformInfo:
    system: str
    machine: str
    python_version: str
    is_windows: bool
    is_linux_arm64: bool
    is_raspberry_pi_target: bool


def detect_platform() -> PlatformInfo:
    system = platform.system()
    machine = platform.machine().lower()
    is_windows = system == "Windows"
    is_linux_arm64 = system == "Linux" and machine in {"aarch64", "arm64"}
    return PlatformInfo(
        system=system,
        machine=platform.machine(),
        python_version=sys.version.split()[0],
        is_windows=is_windows,
        is_linux_arm64=is_linux_arm64,
        is_raspberry_pi_target=is_linux_arm64,
    )
