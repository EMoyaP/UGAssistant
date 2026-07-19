from __future__ import annotations

import unittest

from ugassistant.domain.finger_count import FingerCountStabilizer


class FingerCountStabilizerTests(unittest.TestCase):
    def test_requires_repeated_samples_before_publishing_a_count(self) -> None:
        stabilizer = FingerCountStabilizer(required_samples=2)

        self.assertIsNone(stabilizer.update(3))
        self.assertEqual(stabilizer.update(3), 3)
        self.assertEqual(stabilizer.update(4), 3)
        self.assertEqual(stabilizer.update(4), 4)

    def test_distinguishes_a_closed_fist_from_no_detected_hand(self) -> None:
        stabilizer = FingerCountStabilizer(required_samples=2)

        stabilizer.update(0)
        self.assertEqual(stabilizer.update(0), 0)
        self.assertEqual(stabilizer.update(None), 0)
        self.assertIsNone(stabilizer.update(None))

    def test_reset_clears_candidates_and_stable_value(self) -> None:
        stabilizer = FingerCountStabilizer(required_samples=1)
        self.assertEqual(stabilizer.update(10), 10)

        stabilizer.reset()

        self.assertIsNone(stabilizer.stable)

    def test_rejects_invalid_configuration_and_counts(self) -> None:
        with self.assertRaises(ValueError):
            FingerCountStabilizer(required_samples=0)

        stabilizer = FingerCountStabilizer()
        for value in (-1, 11):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stabilizer.update(value)


if __name__ == "__main__":
    unittest.main()
