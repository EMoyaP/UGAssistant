from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
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
    "whisper_cpp_windows_amd64": (
        PROJECT_ROOT / "tools" / "whisper" / "windows-amd64"
    ),
    "whisper_cpp_linux_arm64": (
        PROJECT_ROOT / "tools" / "whisper" / "linux-arm64"
    ),
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
    return [item for item in runtimes if item.get("engine") == "whisper.cpp"]


def select_runtime(runtime_name: str | None = None) -> dict[str, Any]:
    runtimes = load_runtimes()
    if runtime_name is not None:
        selected = next(
            (item for item in runtimes if item.get("logical_name") == runtime_name),
            None,
        )
    else:
        system_name = platform.system()
        machine = platform.machine().casefold()
        selected = next(
            (
                item
                for item in runtimes
                if item.get("system") == system_name
                and machine
                in {str(value).casefold() for value in item.get("machines", [])}
            ),
            None,
        )
    if selected is None:
        raise RuntimeError(
            "No locked whisper.cpp runtime for "
            f"{platform.system()} {platform.machine()}"
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
    raise ValueError(f"Unsupported whisper.cpp archive: {archive.name}")


def _install_prebuilt(
    runtime: dict[str, Any],
    extraction_root: Path,
    staging: Path,
) -> None:
    package_root = extraction_root / str(runtime["package_root_in_archive"])
    packaged_executable = extraction_root / str(runtime["executable_in_archive"])
    if not package_root.is_dir() or not packaged_executable.is_file():
        raise RuntimeError("Verified whisper.cpp archive is missing whisper-cli")
    shutil.copytree(package_root, staging)


def _install_from_source(
    runtime: dict[str, Any],
    extraction_root: Path,
    staging: Path,
) -> None:
    source_root = extraction_root / str(runtime["source_root_in_archive"])
    if not source_root.is_dir():
        raise RuntimeError("Verified whisper.cpp source archive is incomplete")
    build_root = source_root / "build"
    configure_command = [
        "cmake",
        "-S",
        str(source_root),
        "-B",
        str(build_root),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DWHISPER_BUILD_TESTS=OFF",
        "-DWHISPER_BUILD_EXAMPLES=ON",
        "-DGGML_NATIVE=ON",
    ]
    build_command = [
        "cmake",
        "--build",
        str(build_root),
        "--config",
        "Release",
        "-j",
        str(min(max(os.cpu_count() or 1, 1), 4)),
    ]
    subprocess.run(configure_command, check=True)
    subprocess.run(build_command, check=True)
    built_executable = source_root / str(runtime["built_executable"])
    if not built_executable.is_file():
        raise RuntimeError("whisper.cpp build did not produce whisper-cli")
    staging.mkdir()
    shutil.copy2(
        built_executable,
        staging / str(runtime["installed_executable"]),
    )


def install_runtime(runtime_name: str | None = None) -> dict[str, object]:
    runtime = select_runtime(runtime_name)
    logical_name = str(runtime["logical_name"])
    target = TARGETS[logical_name]
    installed_executable = target / str(runtime["installed_executable"])
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
            f"whisper.cpp target already exists but is not verified: {target}"
        )

    expected_hash = str(runtime["sha256"]).casefold()
    if len(expected_hash) != 64:
        raise ValueError(f"Locked SHA-256 is missing for {logical_name}")

    target.parent.mkdir(parents=True, exist_ok=True)
    archive_suffix = (
        ".zip"
        if str(runtime["archive_name"]).endswith(".zip")
        else ".tar.gz"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="ugassistant-whisper-",
        suffix=archive_suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    temporary_archive = Path(temporary_name)
    request = urllib.request.Request(
        str(runtime["official_url"]),
        headers={"User-Agent": "UGAssistant-runtime-installer/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with temporary_archive.open("wb") as file:
                while chunk := response.read(1024 * 1024):
                    file.write(chunk)
        actual_hash = file_sha256(temporary_archive)
        if actual_hash != expected_hash:
            raise ValueError(
                f"SHA-256 mismatch for {logical_name}: {actual_hash}"
            )

        with tempfile.TemporaryDirectory(
            prefix="ugassistant-whisper-extract-",
            dir=target.parent,
        ) as extraction_directory:
            extraction_root = Path(extraction_directory)
            _extract_archive(temporary_archive, extraction_root)
            staging = extraction_root / "installed"
            install_kind = str(runtime["install_kind"])
            if install_kind == "prebuilt":
                _install_prebuilt(runtime, extraction_root, staging)
            elif install_kind == "cmake_source":
                _install_from_source(runtime, extraction_root, staging)
            else:
                raise ValueError(f"Unknown install kind: {install_kind}")

            staged_executable = staging / str(runtime["installed_executable"])
            if not staged_executable.is_file():
                raise RuntimeError("Installed whisper.cpp executable is missing")
            if runtime.get("system") == "Linux":
                staged_executable.chmod(staged_executable.stat().st_mode | 0o111)
            (staging / marker_path.name).write_text(
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
            staging.replace(target)
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
        description="Install the locked whisper.cpp runtime for this platform."
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
