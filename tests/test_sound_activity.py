from __future__ import annotations

import unittest

from ugassistant.domain.sound_activity import SoundActivityDetector


class SoundActivityDetectorTests(unittest.TestCase):
    def test_requires_stable_sound_and_releases_after_silence(self) -> None:
        detector = SoundActivityDetector(
            activation_threshold=0.02,
            release_threshold=0.01,
            activation_samples=2,
            silence_seconds=0.5,
        )

        first = detector.update(0.03, now=1.0)
        activated = detector.update(0.04, now=1.1)
        held = detector.update(0.005, now=1.4)
        released = detector.update(0.005, now=1.7)

        self.assertFalse(first.active)
        self.assertTrue(activated.active)
        self.assertTrue(activated.changed)
        self.assertTrue(held.active)
        self.assertFalse(released.active)
        self.assertTrue(released.changed)

    def test_hysteresis_keeps_activity_between_thresholds(self) -> None:
        detector = SoundActivityDetector(
            activation_threshold=0.02,
            release_threshold=0.01,
            activation_samples=1,
            silence_seconds=0.2,
        )
        detector.update(0.03, now=1.0)

        snapshot = detector.update(0.015, now=2.0)

        self.assertTrue(snapshot.active)

    def test_reset_and_validation(self) -> None:
        detector = SoundActivityDetector(activation_samples=1)
        detector.update(1.0, now=1.0)
        detector.reset()
        self.assertFalse(detector.active)

        with self.assertRaises(ValueError):
            SoundActivityDetector(
                activation_threshold=0.01,
                release_threshold=0.02,
            )
        with self.assertRaises(ValueError):
            SoundActivityDetector(activation_samples=0)
        with self.assertRaises(ValueError):
            SoundActivityDetector(silence_seconds=-1)


if __name__ == "__main__":
    unittest.main()
