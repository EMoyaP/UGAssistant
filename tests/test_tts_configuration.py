from __future__ import annotations

import unittest

from ugassistant.config import load_app_settings, load_model_lock


class TTSConfigurationTests(unittest.TestCase):
    def test_spanish_and_french_voices_are_locked_and_configured(self) -> None:
        settings = load_app_settings()
        locked = {
            model["logical_name"]: model
            for model in load_model_lock().get("models", [])
        }

        self.assertEqual(settings.tts_voice_id, "es_ES-davefx-medium")
        self.assertEqual(settings.tts_language, "es_ES")
        self.assertEqual(settings.tts_french_voice_id, "fr_FR-tom-medium")
        self.assertEqual(settings.tts_french_language, "fr_FR")
        self.assertEqual(
            locked["tts_fr"]["sha256"],
            "bf65074ccdeeeeaa832e75edb1c0a513c01c9a972bdf085ff8a6e71ea234fd41",
        )
        self.assertEqual(
            locked["tts_fr_config"]["sha256"],
            "2f7f885ad5a0aad802e3cc24e4f57239febdcb142b4876de5d238094674361cc",
        )


if __name__ == "__main__":
    unittest.main()
