from __future__ import annotations

import unittest

from ugassistant.config import (
    PROJECT_ROOT,
    load_app_settings,
    load_model_lock,
    load_yaml,
)


class STTConfigurationTests(unittest.TestCase):
    def test_multilingual_base_and_platform_runtimes_are_locked(self) -> None:
        settings = load_app_settings()
        models = {
            model["logical_name"]: model
            for model in load_model_lock().get("models", [])
        }
        runtimes = {
            runtime["logical_name"]: runtime
            for runtime in load_yaml(
                PROJECT_ROOT / "config" / "runtimes.lock.yaml"
            ).get("runtimes", [])
        }

        self.assertEqual(settings.stt_model_relative_path.name, "ggml-base.bin")
        self.assertEqual(settings.stt_accepted_languages, ("es", "fr"))
        self.assertEqual(models["stt"]["engine"], "whisper.cpp")
        self.assertEqual(
            models["stt"]["sha256"],
            "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe",
        )
        self.assertEqual(models["stt"]["execution"]["language"], "auto")
        self.assertEqual(
            models["stt"]["execution"]["accepted_languages"],
            ["es", "fr"],
        )
        self.assertEqual(
            runtimes["whisper_cpp_windows_amd64"]["version_or_tag"],
            "v1.8.1",
        )
        self.assertTrue(
            runtimes["whisper_cpp_linux_arm64"]["version_or_tag"].startswith(
                "v1.8.1@"
            )
        )

    def test_ollama_model_and_local_wake_words_are_fixed(self) -> None:
        settings = load_app_settings()
        models = {
            model["logical_name"]: model
            for model in load_model_lock().get("models", [])
        }

        self.assertEqual(settings.llm_model, "qwen3:1.7b")
        self.assertEqual(settings.llm_base_url, "http://127.0.0.1:11434")
        self.assertGreater(
            settings.llm_complete_max_response_characters,
            settings.llm_max_response_characters,
        )
        self.assertGreater(
            settings.llm_complete_max_tokens,
            settings.llm_max_tokens,
        )
        self.assertEqual(settings.wake_spanish_words, ("hola",))
        self.assertEqual(settings.wake_french_words, ("salut",))
        self.assertEqual(settings.stt_silence_seconds, 2.0)
        self.assertEqual(models["llm"]["version_or_tag"], "qwen3:1.7b")


if __name__ == "__main__":
    unittest.main()
