from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ugassistant.adapters.portaudio import PortAudioAdapter  # noqa: E402
from ugassistant.config import load_app_settings  # noqa: E402


VISION_MODELS = {
    "face_detection": (
        "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "palm_detection": (
        "palm_detection_mediapipe_2023feb.onnx",
        "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c",
    ),
    "hand_pose": (
        "handpose_estimation_mediapipe_2023feb.onnx",
        "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c",
    ),
}
STT_MODEL_SHA256 = (
    "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe"
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    data: dict[str, object]


def run_command(command: list[str], timeout: float = 5.0) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (completed.stdout or completed.stderr).strip()
    return completed.returncode == 0, output


def check_platform() -> CheckResult:
    system = platform.system()
    machine = platform.machine()
    is_supported = system == "Windows" or (
        system == "Linux" and machine.lower() in {"aarch64", "arm64"}
    )
    return CheckResult(
        name="platform",
        ok=is_supported,
        detail="Windows development host or Linux ARM64 target"
        if is_supported
        else "Expected Windows or Linux ARM64",
        data={"system": system, "machine": machine, "platform": platform.platform()},
    )


def check_python() -> CheckResult:
    version = sys.version_info
    ok = version >= (3, 10)
    return CheckResult(
        name="python",
        ok=ok,
        detail="Python >= 3.10" if ok else "Python 3.10 or newer is required",
        data={"version": sys.version.split()[0], "executable": sys.executable},
    )


def check_ollama() -> CheckResult:
    path = shutil.which("ollama")
    if not path:
        return CheckResult(
            name="ollama",
            ok=False,
            detail="ollama executable not found in PATH",
            data={},
        )
    ok, output = run_command([path, "--version"])
    model_installed = False
    models_output = ""
    if ok:
        listed, models_output = run_command([path, "list"], timeout=15.0)
        model_installed = listed and any(
            line.split(maxsplit=1)[0] == "qwen3:1.7b"
            for line in models_output.splitlines()[1:]
            if line.strip()
        )
    return CheckResult(
        name="ollama",
        ok=ok and model_installed,
        detail=(
            "Ollama and locked qwen3:1.7b model available"
            if ok and model_installed
            else output or "Ollama found but qwen3:1.7b is not installed"
        ),
        data={
            "path": path,
            "locked_model": "qwen3:1.7b",
            "model_installed": model_installed,
            "models": models_output,
        },
    )


def check_cameras() -> CheckResult:
    devices: list[str] = []
    detail = "OpenCV backend enumeration"

    try:
        import cv2
        from cv2_enumerate_cameras import enumerate_cameras

        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        devices = [
            f"{camera.index}: {camera.name}"
            for camera in enumerate_cameras(backend)
        ]
    except (ImportError, OSError, RuntimeError) as exc:
        detail = f"camera enumeration unavailable: {exc}"

    return CheckResult(
        name="camera_devices",
        ok=bool(devices),
        detail=detail if devices else "no camera devices detected",
        data={"devices": devices},
    )


def check_camera_runtime() -> CheckResult:
    try:
        import cv2
    except ImportError:
        return CheckResult(
            name="camera_runtime",
            ok=False,
            detail="opencv-python-headless is not installed",
            data={"models": {}},
        )

    models: dict[str, dict[str, object]] = {}
    all_models_valid = True
    for logical_name, (file_name, expected_hash) in VISION_MODELS.items():
        model_path = PROJECT_ROOT / "models" / "vision" / file_name
        actual_hash = ""
        if model_path.is_file():
            digest = hashlib.sha256()
            with model_path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_hash = digest.hexdigest()
        verified = actual_hash == expected_hash
        all_models_valid = all_models_valid and verified
        models[logical_name] = {
            "path": str(model_path),
            "sha256": actual_hash or None,
            "verified": verified,
        }

    ok = cv2.__version__.startswith("4.13.") and all_models_valid
    return CheckResult(
        name="camera_runtime",
        ok=ok,
        detail="OpenCV and all locked vision models verified"
        if ok
        else "OpenCV 4.13 or a locked vision model is missing/invalid",
        data={
            "opencv_version": cv2.__version__,
            "models": models,
        },
    )


def check_audio() -> CheckResult:
    try:
        devices = asyncio.run(PortAudioAdapter().list_devices())
    except Exception as exc:
        return CheckResult(
            name="audio_devices",
            ok=False,
            detail=str(exc),
            data={"inputs": [], "outputs": []},
        )

    inputs = [device.to_dict() for device in devices if device.kind == "input"]
    outputs = [device.to_dict() for device in devices if device.kind == "output"]

    return CheckResult(
        name="audio_devices",
        ok=bool(inputs) and bool(outputs),
        detail="audio input and output detected"
        if inputs and outputs
        else "missing audio input or output",
        data={"inputs": inputs, "outputs": outputs},
    )


def check_stt_runtime() -> CheckResult:
    settings = load_app_settings(PROJECT_ROOT)
    executable_path = settings.stt_executable_path(
        platform.system(),
        platform.machine(),
    )
    model_path = settings.stt_model_path
    actual_hash = ""
    if model_path.is_file():
        digest = hashlib.sha256()
        with model_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_hash = digest.hexdigest()
    model_verified = actual_hash == STT_MODEL_SHA256

    executable_ok = executable_path is not None and executable_path.is_file()
    runtime_ok = False
    runtime_detail = "unsupported platform"
    if executable_ok and executable_path is not None:
        runtime_ok, runtime_detail = run_command(
            [str(executable_path), "--help"],
            timeout=10.0,
        )
        runtime_detail = runtime_detail.splitlines()[0] if runtime_detail else "ready"
    elif executable_path is not None:
        runtime_detail = "whisper-cli executable is missing"

    ok = executable_ok and runtime_ok and model_verified
    return CheckResult(
        name="stt_runtime",
        ok=ok,
        detail=(
            "whisper.cpp and locked multilingual base model verified"
            if ok
            else "whisper.cpp runtime or locked STT model is missing/invalid"
        ),
        data={
            "executable": str(executable_path) if executable_path else None,
            "runtime_detail": runtime_detail,
            "model": str(model_path),
            "model_sha256": actual_hash or None,
            "model_verified": model_verified,
            "accepted_languages": list(settings.stt_accepted_languages),
        },
    )


def check_disk() -> CheckResult:
    usage = shutil.disk_usage(PROJECT_ROOT)
    free_gb = usage.free / (1024**3)
    ok = free_gb >= 10
    return CheckResult(
        name="disk_space",
        ok=ok,
        detail=f"{free_gb:.1f} GB free",
        data={
            "project_root": str(PROJECT_ROOT),
            "total_gb": round(usage.total / (1024**3), 1),
            "free_gb": round(free_gb, 1),
        },
    )


def main() -> int:
    checks = [
        check_platform(),
        check_python(),
        check_ollama(),
        check_cameras(),
        check_camera_runtime(),
        check_audio(),
        check_stt_runtime(),
        check_disk(),
    ]
    payload = {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
