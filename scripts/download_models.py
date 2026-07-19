from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORIES = {
    "stt": Path("models/stt"),
    "tts": Path("models/tts"),
    "tts_config": Path("models/tts"),
    "tts_fr": Path("models/tts"),
    "tts_fr_config": Path("models/tts"),
    "face_detection": Path("models/vision"),
    "palm_detection": Path("models/vision"),
    "hand_pose": Path("models/vision"),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_model(logical_name: str) -> dict[str, Any]:
    lock_path = PROJECT_ROOT / "config" / "models.lock.yaml"
    with lock_path.open("r", encoding="utf-8") as file:
        lock = yaml.safe_load(file) or {}
    for model in lock.get("models", []):
        if model.get("logical_name") == logical_name:
            return model
    raise ValueError(f"Model not found in lock: {logical_name}")


def download_locked_model(logical_name: str) -> dict[str, object]:
    if logical_name == "llm":
        return pull_locked_ollama_model()
    model = load_locked_model(logical_name)
    expected_hash = str(model.get("sha256", "")).lower()
    if len(expected_hash) != 64:
        raise ValueError(f"Locked SHA-256 is missing for {logical_name}")

    destination_directory = PROJECT_ROOT / MODEL_DIRECTORIES[logical_name]
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / str(model["file_name"])

    if destination.exists() and file_sha256(destination) == expected_hash:
        return {
            "status": "already_verified",
            "path": str(destination),
            "sha256": expected_hash,
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        str(model["official_url"]),
        headers={"User-Agent": "UGAssistant-model-downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
        actual_hash = file_sha256(temporary)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {logical_name}: {actual_hash}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "status": "downloaded_and_verified",
        "path": str(destination),
        "sha256": expected_hash,
    }


def pull_locked_ollama_model() -> dict[str, object]:
    model = load_locked_model("llm")
    tag = str(model["version_or_tag"])
    executable = shutil.which("ollama")
    if executable is None:
        raise RuntimeError("Ollama is not installed or is not available in PATH")
    completed = subprocess.run(
        [executable, "pull", tag],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=1800,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"Ollama could not pull {tag}: {detail}")
    return {
        "status": "downloaded_by_ollama",
        "model": tag,
        "detail": "Ollama manages the locked model digest locally",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download only explicitly selected locked models."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted([*MODEL_DIRECTORIES, "llm"]),
        help="Logical model name from config/models.lock.yaml.",
    )
    args = parser.parse_args()
    result = download_locked_model(args.model)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
