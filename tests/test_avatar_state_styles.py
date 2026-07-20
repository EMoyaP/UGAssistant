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

    def test_spotify_playback_uses_headphones_and_neutral_motion(self) -> None:
        styles_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ugassistant"
            / "web"
            / "static"
            / "styles.css"
        )
        styles = styles_path.read_text(encoding="utf-8")

        self.assertIn('.app-shell[data-spotify-playing="true"] .listening-headphones', styles)
        self.assertIn("animation: music-listening", styles)

    def test_timer_countdown_uses_large_high_contrast_text(self) -> None:
        styles_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ugassistant"
            / "web"
            / "static"
            / "styles.css"
        )
        styles = styles_path.read_text(encoding="utf-8")

        self.assertIn(".timer-chip strong", styles)
        self.assertIn("font-size: 30px", styles)
        self.assertIn("background: rgba(20, 24, 30, 0.98)", styles)
        self.assertIn("color: #fff", styles)


if __name__ == "__main__":
    unittest.main()
