from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / "config" / "runtimes.lock.yaml"
TARGETS = {
    "piper_windows_amd64": PROJECT_ROOT / "tools" / "piper" / "windows-amd64",
    "piper_linux_arm64": PROJECT_ROOT / "tools" / "piper" / "linux-arm64",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtimes() -> list[dict[str, Any]]:
    with LOCK_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    runtimes = data.get("runtimes", [])
    if not isinstance(runtimes, list):
        raise ValueError(f"Expected a runtime list in {LOCK_PATH}")
    return runtimes


def select_runtime(runtime_name: str | None = None) -> dict[str, Any]:
    runtimes = load_runtimes()
    if runtime_name is not None:
        selected = next(
            (item for item in runtimes if item.get("logical_name") == runtime_name),
            None,
        )
    else:
        system_name = platform.system()
        machine = platform.machine().lower()
        selected = next(
            (
                item
                for item in runtimes
                if item.get("system") == system_name
                and machine in {str(value).lower() for value in item.get("machines", [])}
            ),
            None,
        )
    if selected is None:
        raise RuntimeError(
            f"No locked Piper runtime for {platform.system()} {platform.machine()}"
        )
    return selected


def _validate_archive_name(name: str) -> None:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")


def _extract_archive(archive: Path, destination: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as package:
            for member in package.infolist():
                _validate_archive_name(member.filename)
            package.extractall(destination)
        return
    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as package:
            for member in package.getmembers():
                _validate_archive_name(member.name)
                if member.issym() or member.islnk():
                    _validate_archive_name(member.linkname)
            package.extractall(destination)
        return
    raise ValueError(f"Unsupported Piper archive: {archive.name}")


def install_runtime(runtime_name: str | None = None) -> dict[str, object]:
    runtime = select_runtime(runtime_name)
    logical_name = str(runtime["logical_name"])
    target = TARGETS[logical_name]
    archive_executable = Path(str(runtime["executable_in_archive"]))
    installed_executable = target / archive_executable.relative_to("piper")
    marker_path = target / ".ugassistant-runtime.json"

    if installed_executable.is_file() and marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("sha256") == runtime.get("sha256"):
            return {
                "status": "already_verified",
                "runtime": logical_name,
                "executable": str(installed_executable),
            }
    if target.exists():
        raise RuntimeError(
            f"Piper target already exists but is not verified: {target}"
        )

    expected_hash = str(runtime["sha256"]).lower()
    if len(expected_hash) != 64:
        raise ValueError(f"Locked SHA-256 is missing for {logical_name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    archive_suffix = ".zip" if str(runtime["archive_name"]).endswith(".zip") else ".tar.gz"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix="ugassistant-piper-",
        suffix=archive_suffix,
        dir=target.parent,
    )
    os.close(file_descriptor)
    temporary_archive = Path(temporary_name)

    request = urllib.request.Request(
        str(runtime["official_url"]),
        headers={"User-Agent": "UGAssistant-runtime-installer/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary_archive.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
        actual_hash = file_sha256(temporary_archive)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {logical_name}: {actual_hash}"
            )

        with tempfile.TemporaryDirectory(
            prefix="ugassistant-piper-extract-",
            dir=target.parent,
        ) as extraction_directory:
            extraction_root = Path(extraction_directory)
            _extract_archive(temporary_archive, extraction_root)
            package_root = extraction_root / "piper"
            packaged_executable = extraction_root / archive_executable
            if not packaged_executable.is_file():
                raise RuntimeError(
                    f"Piper executable is missing from the verified archive: {archive_executable}"
                )
            package_root.replace(target)

        if runtime.get("system") == "Linux":
            installed_executable.chmod(installed_executable.stat().st_mode | 0o111)
        marker_path.write_text(
            json.dumps(
                {
                    "runtime": logical_name,
                    "version": runtime["version_or_tag"],
                    "sha256": expected_hash,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        temporary_archive.unlink(missing_ok=True)

    return {
        "status": "installed_and_verified",
        "runtime": logical_name,
        "executable": str(installed_executable),
        "sha256": expected_hash,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the locked Piper runtime for this platform."
    )
    parser.add_argument(
        "--runtime",
        choices=sorted(TARGETS),
        help="Override automatic Windows AMD64/Linux ARM64 detection.",
    )
    args = parser.parse_args()
    print(json.dumps(install_runtime(args.runtime), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
