from __future__ import annotations

from array import array
import io
import unittest
import wave

from ugassistant.domain.lip_sync import build_lip_sync_track


class LipSyncTests(unittest.TestCase):
    def test_builds_mouth_levels_from_silence_and_voice_energy(self) -> None:
        output = io.BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(1000)
            wav_file.writeframes(
                array("h", [0] * 100 + [9000, -9000] * 50).tobytes()
            )

        track = build_lip_sync_track(output.getvalue(), interval_ms=100)

        self.assertEqual(track.duration_ms, 200)
        self.assertEqual(track.levels[0], 0.0)
        self.assertGreater(track.levels[1], 0.9)

    def test_rejects_intervals_too_small(self) -> None:
        with self.assertRaises(ValueError):
            build_lip_sync_track(b"not-a-wave", interval_ms=10)


if __name__ == "__main__":
    unittest.main()
