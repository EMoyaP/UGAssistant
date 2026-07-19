from __future__ import annotations

import unittest
from unittest.mock import patch

from ugassistant.domain.platform import detect_platform


class PlatformDetectionTests(unittest.TestCase):
    @patch("ugassistant.domain.platform.platform.machine", return_value="AMD64")
    @patch("ugassistant.domain.platform.platform.system", return_value="Windows")
    def test_detects_windows(self, _system: object, _machine: object) -> None:
        info = detect_platform()

        self.assertTrue(info.is_windows)
        self.assertFalse(info.is_linux_arm64)

    @patch("ugassistant.domain.platform.platform.machine", return_value="aarch64")
    @patch("ugassistant.domain.platform.platform.system", return_value="Linux")
    def test_detects_linux_arm64(self, _system: object, _machine: object) -> None:
        info = detect_platform()

        self.assertFalse(info.is_windows)
        self.assertTrue(info.is_linux_arm64)
        self.assertTrue(info.is_raspberry_pi_target)


if __name__ == "__main__":
    unittest.main()
