from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from ugassistant.adapters.preferences import YAMLPreferenceStore
from ugassistant.domain.ports import AudioDevice
from ugassistant.domain.preferences import (
    DevicePreference,
    UserPreferences,
    match_device_preference,
)


class PreferenceStoreTests(unittest.TestCase):
    def test_round_trip_preserves_all_local_preferences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "data" / "preferences.yaml"
            store = YAMLPreferenceStore(path)
            expected = UserPreferences(
                camera_device=DevicePreference("USB camera", 3),
                camera_enabled=True,
                microphone_device=DevicePreference("USB microphone", 8, "ALSA"),
                microphone_enabled=True,
                speaker_device=DevicePreference("Display speakers", 10, "ALSA"),
                speaker_enabled=False,
                output_volume=0.37,
                voice_id="es_ES-davefx-medium",
                language="es_ES",
                speech_rate=0.8,
            )

            store.save(expected)
            loaded = store.load()

            self.assertEqual(loaded, expected)
            self.assertNotIn(".tmp", " ".join(item.name for item in path.parent.iterdir()))

    def test_rejects_an_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "preferences.yaml"
            path.write_text("schema_version: 99\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                YAMLPreferenceStore(path).load()

    def test_matches_stable_identity_when_runtime_index_changes(self) -> None:
        preference = DevicePreference("USB microphone", 4, "ALSA")
        devices = (
            AudioDevice(
                device_index=19,
                name="USB microphone",
                kind="input",
                available=True,
                host_api="ALSA",
            ),
        )

        selected = match_device_preference(preference, devices)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.device_index, 19)  # type: ignore[union-attr]

    def test_does_not_substitute_a_different_device_at_the_old_index(self) -> None:
        preference = DevicePreference("Preferred microphone", 4, "ALSA")
        devices = (
            AudioDevice(
                device_index=4,
                name="Different microphone",
                kind="input",
                available=True,
                host_api="ALSA",
            ),
        )

        self.assertIsNone(match_device_preference(preference, devices))

    def test_replacement_updates_only_the_requested_setting(self) -> None:
        original = UserPreferences(
            camera_device=DevicePreference("Camera", 1),
            output_volume=1.0,
        )

        updated = replace(original, output_volume=0.5)

        self.assertEqual(updated.camera_device, original.camera_device)
        self.assertEqual(updated.output_volume, 0.5)


if __name__ == "__main__":
    unittest.main()
