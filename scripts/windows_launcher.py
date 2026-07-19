from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
import webbrowser


HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}/"
HEALTH_URL = f"{URL}health"


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def python_executable(root: Path) -> str | None:
    virtualenv_python = root / ".venv" / "Scripts" / "python.exe"
    if virtualenv_python.is_file():
        return str(virtualenv_python)
    if getattr(sys, "frozen", False):
        return None
    return sys.executable


def server_is_ready() -> bool:
    try:
        with urlopen(HEALTH_URL, timeout=0.5) as response:
            return response.status == 200
    except (URLError, OSError):
        return False


def show_error(message: str) -> None:
    if os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, "UGAssistant", 0x10)
    else:
        print(message, file=sys.stderr)


def main() -> int:
    root = project_root()
    if server_is_ready():
        webbrowser.open(URL)
        return 0

    environment = os.environ.copy()
    source_path = str(root / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not environment.get("PYTHONPATH")
        else f"{source_path}{os.pathsep}{environment['PYTHONPATH']}"
    )
    python_path = python_executable(root)
    if python_path is None:
        show_error("No se encontro el entorno .venv de UGAssistant.")
        return 1
    process = subprocess.Popen(
        [python_path, "-m", "ugassistant"],
        cwd=root,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if server_is_ready():
            webbrowser.open(URL)
            return process.wait()
        if process.poll() is not None:
            show_error("UGAssistant no pudo iniciarse. Revisa la configuracion local.")
            return process.returncode or 1
        time.sleep(0.2)

    process.terminate()
    show_error("UGAssistant no respondio en 30 segundos.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
