from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

import yaml

from ugassistant.services.fixed_model_updates import FixedModelUpdateService


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lock_for(data: bytes) -> dict[str, object]:
    return {
        "models": [
            {
                "logical_name": "stt",
                "file_name": "ggml-base.bin",
                "version_or_tag": "locked",
                "sha256": digest(data),
                "official_url": "https://example.invalid/model",
                "update_url": "https://example.invalid/model",
                "size": f"{len(data)} bytes",
            }
        ]
    }


class FixedModelUpdateServiceTests(unittest.TestCase):
    def test_updates_a_candidate_only_after_baseline_and_candidate_checks(self) -> None:
        current = b"current-model"
        candidate = b"candidate-model"
        checks: list[set[str]] = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "models" / "stt" / "ggml-base.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(current)
            lock_path = root / "config" / "models.lock.yaml"
            lock_path.parent.mkdir()
            lock_path.write_text(yaml.safe_dump(lock_for(current), sort_keys=False), encoding="utf-8")

            def download(_url: str, destination: Path) -> None:
                destination.write_bytes(candidate)

            service = FixedModelUpdateService(
                project_root=root,
                model_lock_path=lock_path,
                model_paths={"stt": target},
                downloader=download,
                functional_check=lambda changed: checks.append(changed),
            )
            result = service.check_and_update()
            updated_lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))

        self.assertEqual(result, [{"logical_name": "stt", "state": "updated"}])
        self.assertEqual(checks, [set(), {"stt"}])
        self.assertEqual(updated_lock["models"][0]["sha256"], digest(candidate))
        self.assertEqual(updated_lock["models"][0]["version_or_tag"], f"auto-sha256:{digest(candidate)[:16]}")
        self.assertEqual(updated_lock["models"][0]["official_url"], "https://example.invalid/model")

    def test_restores_last_functional_file_and_lock_when_candidate_check_fails(self) -> None:
        current = b"current-model"
        candidate = b"candidate-model"
        checks = 0
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "models" / "stt" / "ggml-base.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(current)
            lock_path = root / "config" / "models.lock.yaml"
            lock_path.parent.mkdir()
            original = yaml.safe_dump(lock_for(current), sort_keys=False)
            lock_path.write_text(original, encoding="utf-8")

            def download(_url: str, destination: Path) -> None:
                destination.write_bytes(candidate)

            def check(_changed: set[str]) -> None:
                nonlocal checks
                checks += 1
                if checks == 2:
                    raise RuntimeError("candidate failed")

            service = FixedModelUpdateService(
                project_root=root,
                model_lock_path=lock_path,
                model_paths={"stt": target},
                downloader=download,
                functional_check=check,
            )
            result = service.check_and_update()
            restored = target.read_bytes()
            restored_lock = lock_path.read_text(encoding="utf-8")

        self.assertEqual(restored, current)
        self.assertEqual(restored_lock, original)
        self.assertEqual(result[0]["state"], "rolled_back")

    def test_repairs_a_missing_local_file_when_the_candidate_is_already_locked(self) -> None:
        current = b"current-model"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "models" / "stt" / "ggml-base.bin"
            lock_path = root / "config" / "models.lock.yaml"
            lock_path.parent.mkdir()
            lock_path.write_text(
                yaml.safe_dump(lock_for(current), sort_keys=False), encoding="utf-8"
            )

            def download(_url: str, destination: Path) -> None:
                destination.write_bytes(current)

            service = FixedModelUpdateService(
                project_root=root,
                model_lock_path=lock_path,
                model_paths={"stt": target},
                downloader=download,
                functional_check=lambda _changed: None,
            )
            result = service.check_and_update()

            self.assertEqual(target.read_bytes(), current)

        self.assertEqual(result, [{"logical_name": "stt", "state": "repaired"}])

    def test_leaves_models_untouched_when_the_download_fails(self) -> None:
        current = b"current-model"
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "models" / "stt" / "ggml-base.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(current)
            lock_path = root / "config" / "models.lock.yaml"
            lock_path.parent.mkdir()
            original = yaml.safe_dump(lock_for(current), sort_keys=False)
            lock_path.write_text(original, encoding="utf-8")

            service = FixedModelUpdateService(
                project_root=root,
                model_lock_path=lock_path,
                model_paths={"stt": target},
                downloader=lambda _url, _destination: (_ for _ in ()).throw(RuntimeError("offline")),
                functional_check=lambda _changed: None,
            )
            result = service.check_and_update()

            self.assertEqual(target.read_bytes(), current)
            self.assertEqual(lock_path.read_text(encoding="utf-8"), original)

        self.assertEqual(result[0]["state"], "error")

    def test_reports_when_all_fixed_models_are_already_current(self) -> None:
        current = b"current-model"
        events: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            target = root / "models" / "stt" / "ggml-base.bin"
            target.parent.mkdir(parents=True)
            target.write_bytes(current)
            lock_path = root / "config" / "models.lock.yaml"
            lock_path.parent.mkdir()
            lock_path.write_text(
                yaml.safe_dump(lock_for(current), sort_keys=False), encoding="utf-8"
            )
            service = FixedModelUpdateService(
                project_root=root,
                model_lock_path=lock_path,
                model_paths={"stt": target},
                downloader=lambda _url, destination: destination.write_bytes(current),
                functional_check=lambda _changed: None,
                on_progress=events.append,
            )

            service.check_and_update()

        self.assertEqual(events[-1]["logical_name"], "fixed_models")
        self.assertEqual(events[-1]["state"], "up_to_date")


if __name__ == "__main__":
    unittest.main()
