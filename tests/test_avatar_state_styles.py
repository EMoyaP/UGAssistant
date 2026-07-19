from __future__ import annotations

from pathlib import Path
import unittest


class AvatarStateStyleTests(unittest.TestCase):
    def test_transcribing_uses_thinking_expression_and_animation(self) -> None:
        styles_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ugassistant"
            / "web"
            / "static"
            / "styles.css"
        )
        styles = styles_path.read_text(encoding="utf-8")

        for element in (
            "avatar",
            "head",
            "left-brow",
            "right-brow",
            "mouth",
            "thinking-hand",
        ):
            selector = f'.app-shell[data-state="TRANSCRIBING"] .{element}'
            self.assertIn(selector, styles)


if __name__ == "__main__":
    unittest.main()
