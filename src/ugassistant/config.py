from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppSettings:
    project_root: Path = PROJECT_ROOT
    host: str = "127.0.0.1"
    port: int = 8000
    max_camera_width: int = 640
    max_camera_height: int = 480
    target_face_detection_fps_min: int = 5
    target_face_detection_fps_max: int = 10
    target_ui_fps: int = 30
    camera_device_index: int = 0
    camera_enabled_by_default: bool = False
    camera_preview_fps: float = 8.0
    camera_idle_fps: float = 1.0
    camera_person_detected_fps: float = 2.0
    camera_processing_fps: float = 1.0
    camera_gesture_fps: float = 5.0
    camera_mirror_preview: bool = True
    camera_model_relative_path: Path = Path(
        "models/vision/face_detection_yunet_2023mar.onnx"
    )
    camera_score_threshold: float = 0.75
    camera_nms_threshold: float = 0.3
    hand_detection_enabled: bool = True
    hand_palm_model_relative_path: Path = Path(
        "models/vision/palm_detection_mediapipe_2023feb.onnx"
    )
    hand_pose_model_relative_path: Path = Path(
        "models/vision/handpose_estimation_mediapipe_2023feb.onnx"
    )
    hand_inference_interval_frames: int = 3
    hand_max_hands: int = 2
    hand_palm_score_threshold: float = 0.6
    hand_palm_nms_threshold: float = 0.3
    hand_score_threshold: float = 0.8
    finger_count_stable_samples: int = 2
    audio_activation_threshold: float = 0.015
    audio_release_threshold: float = 0.008
    audio_activation_samples: int = 2
    audio_silence_seconds: float = 0.8
    audio_block_duration_ms: int = 50
    audio_output_volume: float = 1.0
    audio_spatial_pan_enabled: bool = True
    audio_spatial_pan_max: float = 0.65
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3:4b-instruct"
    llm_timeout_seconds: float = 120.0
    llm_max_history_turns: int = 3
    llm_short_context_tokens: int = 2048
    llm_complete_context_tokens: int = 4096
    llm_max_tokens: int = 256
    llm_complete_max_tokens: int = 1536
    llm_temperature: float = 0.4
    wake_spanish_words: tuple[str, ...] = ("hola",)
    wake_french_words: tuple[str, ...] = ("salut",)
    wake_spanish_greeting: str = "¿Qué desea?"
    wake_french_greeting: str = "Que puis-je faire pour vous ?"
    stt_model_relative_path: Path = Path("models/stt/ggml-base.bin")
    stt_windows_executable_relative_path: Path = Path(
        "tools/whisper/windows-amd64/whisper-cli.exe"
    )
    stt_linux_arm64_executable_relative_path: Path = Path(
        "tools/whisper/linux-arm64/whisper-cli"
    )
    stt_threads: int = 4
    stt_process_timeout_seconds: float = 120.0
    stt_sample_rate: int = 16000
    stt_wait_for_speech_seconds: float = 8.0
    stt_silence_seconds: float = 2.0
    stt_max_recording_seconds: float = 20.0
    stt_pre_roll_seconds: float = 0.3
    stt_accepted_languages: tuple[str, ...] = ("es", "fr")
    tts_voice_id: str = "es_ES-davefx-medium"
    tts_voice_name: str = "DaveFX"
    tts_language: str = "es_ES"
    tts_model_relative_path: Path = Path(
        "models/tts/es_ES-davefx-medium.onnx"
    )
    tts_config_relative_path: Path = Path(
        "models/tts/es_ES-davefx-medium.onnx.json"
    )
    tts_french_voice_id: str = "fr_FR-tom-medium"
    tts_french_voice_name: str = "Tom"
    tts_french_language: str = "fr_FR"
    tts_french_model_relative_path: Path = Path(
        "models/tts/fr_FR-tom-medium.onnx"
    )
    tts_french_config_relative_path: Path = Path(
        "models/tts/fr_FR-tom-medium.onnx.json"
    )
    tts_windows_executable_relative_path: Path = Path(
        "tools/piper/windows-amd64/piper.exe"
    )
    tts_linux_arm64_executable_relative_path: Path = Path(
        "tools/piper/linux-arm64/piper"
    )
    tts_max_text_length: int = 500
    tts_process_timeout_seconds: float = 45.0
    tts_output_guard_seconds: float = 0.2
    tts_chunk_pause_seconds: float = 0.25
    tts_speech_rate: float = 0.85
    tts_length_scale: float = 1.0
    tts_noise_scale: float = 0.667
    tts_noise_w: float = 0.8
    preferences_relative_path: Path = Path("data/preferences.yaml")
    spotify_token_relative_path: Path = Path("data/spotify.tokens.json")
    spotify_redirect_uri: str = "http://127.0.0.1:8000/api/spotify/callback"
    spotify_market: str = "ES"

    @property
    def camera_model_path(self) -> Path:
        return self.project_root / self.camera_model_relative_path

    @property
    def hand_palm_model_path(self) -> Path:
        return self.project_root / self.hand_palm_model_relative_path

    @property
    def hand_pose_model_path(self) -> Path:
        return self.project_root / self.hand_pose_model_relative_path

    @property
    def tts_model_path(self) -> Path:
        return self.project_root / self.tts_model_relative_path

    @property
    def stt_model_path(self) -> Path:
        return self.project_root / self.stt_model_relative_path

    @property
    def tts_config_path(self) -> Path:
        return self.project_root / self.tts_config_relative_path

    @property
    def tts_french_model_path(self) -> Path:
        return self.project_root / self.tts_french_model_relative_path

    @property
    def tts_french_config_path(self) -> Path:
        return self.project_root / self.tts_french_config_relative_path

    @property
    def preferences_path(self) -> Path:
        return self.project_root / self.preferences_relative_path

    @property
    def spotify_token_path(self) -> Path:
        return self.project_root / self.spotify_token_relative_path

    def tts_executable_path(
        self,
        system_name: str,
        machine: str,
    ) -> Path | None:
        normalized_machine = machine.casefold()
        if system_name == "Windows" and normalized_machine in {"amd64", "x86_64"}:
            return self.project_root / self.tts_windows_executable_relative_path
        if system_name == "Linux" and normalized_machine in {"aarch64", "arm64"}:
            return self.project_root / self.tts_linux_arm64_executable_relative_path
        return None

    def stt_executable_path(
        self,
        system_name: str,
        machine: str,
    ) -> Path | None:
        normalized_machine = machine.casefold()
        if system_name == "Windows" and normalized_machine in {"amd64", "x86_64"}:
            return self.project_root / self.stt_windows_executable_relative_path
        if system_name == "Linux" and normalized_machine in {"aarch64", "arm64"}:
            return self.project_root / self.stt_linux_arm64_executable_relative_path
        return None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


def load_model_lock(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return load_yaml(project_root / "config" / "models.lock.yaml")


def load_app_settings(project_root: Path = PROJECT_ROOT) -> AppSettings:
    data = load_yaml(project_root / "config" / "app.yaml")
    app = data.get("app", {})
    performance = data.get("performance", {})
    camera = data.get("camera", {})
    camera_activity_fps = camera.get("activity_fps", {})
    hands = data.get("hands", {})
    audio = data.get("audio", {})
    llm = data.get("llm", {})
    wake = data.get("wake_word", {})
    stt = data.get("stt", {})
    tts = data.get("tts", {})
    persistence = data.get("persistence", {})
    spotify = data.get("spotify", {})
    resolution = performance.get("camera_max_resolution", [640, 480])
    detection_fps = performance.get("face_detection_target_fps", [5, 10])
    audio_activation_threshold = min(
        max(float(audio.get("activation_threshold", 0.015)), 0.0),
        1.0,
    )
    audio_release_threshold = min(
        max(float(audio.get("release_threshold", 0.008)), 0.0),
        audio_activation_threshold,
    )

    return AppSettings(
        project_root=project_root,
        host=str(app.get("host", "127.0.0.1")),
        port=int(app.get("port", 8000)),
        max_camera_width=int(resolution[0]),
        max_camera_height=int(resolution[1]),
        target_face_detection_fps_min=int(detection_fps[0]),
        target_face_detection_fps_max=int(detection_fps[1]),
        target_ui_fps=int(performance.get("ui_target_fps", 30)),
        camera_device_index=int(camera.get("device_index", 0)),
        camera_enabled_by_default=bool(camera.get("enabled_by_default", False)),
        camera_preview_fps=float(camera.get("preview_fps", 8)),
        camera_idle_fps=min(
            max(float(camera_activity_fps.get("idle", 1)), 1.0), 10.0
        ),
        camera_person_detected_fps=min(
            max(float(camera_activity_fps.get("person_detected", 2)), 1.0), 10.0
        ),
        camera_processing_fps=min(
            max(float(camera_activity_fps.get("processing", 1)), 1.0), 10.0
        ),
        camera_gesture_fps=min(
            max(float(camera_activity_fps.get("gesture", 5)), 1.0), 10.0
        ),
        camera_mirror_preview=bool(camera.get("mirror_preview", True)),
        camera_model_relative_path=Path(
            str(
                camera.get(
                    "model_path",
                    "models/vision/face_detection_yunet_2023mar.onnx",
                )
            )
        ),
        camera_score_threshold=float(camera.get("score_threshold", 0.75)),
        camera_nms_threshold=float(camera.get("nms_threshold", 0.3)),
        hand_detection_enabled=bool(hands.get("enabled", True)),
        hand_palm_model_relative_path=Path(
            str(
                hands.get(
                    "palm_model_path",
                    "models/vision/palm_detection_mediapipe_2023feb.onnx",
                )
            )
        ),
        hand_pose_model_relative_path=Path(
            str(
                hands.get(
                    "hand_pose_model_path",
                    "models/vision/handpose_estimation_mediapipe_2023feb.onnx",
                )
            )
        ),
        hand_inference_interval_frames=max(
            1, int(hands.get("inference_interval_frames", 3))
        ),
        hand_max_hands=min(max(1, int(hands.get("max_hands", 2))), 2),
        hand_palm_score_threshold=float(hands.get("palm_score_threshold", 0.6)),
        hand_palm_nms_threshold=float(hands.get("palm_nms_threshold", 0.3)),
        hand_score_threshold=float(hands.get("hand_score_threshold", 0.8)),
        finger_count_stable_samples=min(
            max(1, int(hands.get("finger_count_stable_samples", 2))),
            5,
        ),
        audio_activation_threshold=audio_activation_threshold,
        audio_release_threshold=audio_release_threshold,
        audio_activation_samples=min(
            max(1, int(audio.get("activation_samples", 2))),
            10,
        ),
        audio_silence_seconds=min(
            max(float(audio.get("silence_seconds", 0.8)), 0.1),
            5.0,
        ),
        audio_block_duration_ms=min(
            max(int(audio.get("block_duration_ms", 50)), 20),
            200,
        ),
        audio_output_volume=min(
            max(float(audio.get("output_volume", 1.0)), 0.0),
            1.0,
        ),
        audio_spatial_pan_enabled=bool(audio.get("spatial_pan_enabled", True)),
        audio_spatial_pan_max=min(
            max(float(audio.get("spatial_pan_max", 0.65)), 0.0),
            1.0,
        ),
        llm_base_url=str(llm.get("base_url", "http://127.0.0.1:11434")).rstrip("/"),
        llm_model=str(llm.get("model", "qwen3:4b-instruct")),
        llm_timeout_seconds=min(
            max(float(llm.get("timeout_seconds", 120.0)), 5.0),
            300.0,
        ),
        llm_max_history_turns=min(
            max(int(llm.get("max_history_turns", 3)), 0),
            8,
        ),
        llm_short_context_tokens=min(
            max(int(llm.get("short_context_tokens", 2048)), 1024),
            4096,
        ),
        llm_complete_context_tokens=min(
            max(int(llm.get("complete_context_tokens", 4096)), 2048),
            8192,
        ),
        llm_max_tokens=min(max(int(llm.get("max_tokens", 256)), 64), 512),
        llm_complete_max_tokens=min(
            max(int(llm.get("complete_max_tokens", 1536)), 512),
            2048,
        ),
        llm_temperature=min(
            max(float(llm.get("temperature", 0.4)), 0.0),
            2.0,
        ),
        wake_spanish_words=tuple(
            str(word)
            for word in wake.get("spanish_words", ["hola"])
            if str(word).strip()
        ),
        wake_french_words=tuple(
            str(word)
            for word in wake.get("french_words", ["salut"])
            if str(word).strip()
        ),
        wake_spanish_greeting=str(wake.get("spanish_greeting", "¿Qué desea?")),
        wake_french_greeting=str(
            wake.get("french_greeting", "Que puis-je faire pour vous ?")
        ),
        stt_model_relative_path=Path(
            str(stt.get("model_path", "models/stt/ggml-base.bin"))
        ),
        stt_windows_executable_relative_path=Path(
            str(
                stt.get(
                    "windows_executable",
                    "tools/whisper/windows-amd64/whisper-cli.exe",
                )
            )
        ),
        stt_linux_arm64_executable_relative_path=Path(
            str(
                stt.get(
                    "linux_arm64_executable",
                    "tools/whisper/linux-arm64/whisper-cli",
                )
            )
        ),
        stt_threads=min(max(int(stt.get("threads", 4)), 1), 8),
        stt_process_timeout_seconds=min(
            max(float(stt.get("process_timeout_seconds", 120.0)), 10.0),
            300.0,
        ),
        stt_sample_rate=16000,
        stt_wait_for_speech_seconds=min(
            max(float(stt.get("wait_for_speech_seconds", 8.0)), 1.0),
            30.0,
        ),
        stt_silence_seconds=min(
            max(float(stt.get("silence_seconds", 2.0)), 0.5),
            5.0,
        ),
        stt_max_recording_seconds=min(
            max(float(stt.get("max_recording_seconds", 20.0)), 3.0),
            60.0,
        ),
        stt_pre_roll_seconds=min(
            max(float(stt.get("pre_roll_seconds", 0.3)), 0.0),
            1.0,
        ),
        stt_accepted_languages=tuple(
            str(language).casefold()
            for language in stt.get("accepted_languages", ["es", "fr"])
        ),
        tts_voice_id=str(tts.get("voice_id", "es_ES-davefx-medium")),
        tts_voice_name=str(tts.get("voice_name", "DaveFX")),
        tts_language=str(tts.get("language", "es_ES")),
        tts_model_relative_path=Path(
            str(tts.get("model_path", "models/tts/es_ES-davefx-medium.onnx"))
        ),
        tts_config_relative_path=Path(
            str(
                tts.get(
                    "config_path",
                    "models/tts/es_ES-davefx-medium.onnx.json",
                )
            )
        ),
        tts_french_voice_id=str(
            tts.get("french_voice_id", "fr_FR-tom-medium")
        ),
        tts_french_voice_name=str(tts.get("french_voice_name", "Tom")),
        tts_french_language=str(tts.get("french_language", "fr_FR")),
        tts_french_model_relative_path=Path(
            str(
                tts.get(
                    "french_model_path",
                    "models/tts/fr_FR-tom-medium.onnx",
                )
            )
        ),
        tts_french_config_relative_path=Path(
            str(
                tts.get(
                    "french_config_path",
                    "models/tts/fr_FR-tom-medium.onnx.json",
                )
            )
        ),
        tts_windows_executable_relative_path=Path(
            str(
                tts.get(
                    "windows_executable",
                    "tools/piper/windows-amd64/piper.exe",
                )
            )
        ),
        tts_linux_arm64_executable_relative_path=Path(
            str(
                tts.get(
                    "linux_arm64_executable",
                    "tools/piper/linux-arm64/piper",
                )
            )
        ),
        tts_max_text_length=min(
            max(int(tts.get("max_text_length", 500)), 1),
            2000,
        ),
        tts_process_timeout_seconds=min(
            max(float(tts.get("process_timeout_seconds", 45.0)), 5.0),
            180.0,
        ),
        tts_output_guard_seconds=min(
            max(float(tts.get("output_guard_seconds", 0.2)), 0.0),
            2.0,
        ),
        tts_chunk_pause_seconds=min(
            max(float(tts.get("chunk_pause_seconds", 0.25)), 0.0),
            2.0,
        ),
        tts_speech_rate=min(
            max(float(tts.get("speech_rate", 0.85)), 0.6),
            1.3,
        ),
        tts_length_scale=max(float(tts.get("length_scale", 1.0)), 0.1),
        tts_noise_scale=max(float(tts.get("noise_scale", 0.667)), 0.0),
        tts_noise_w=max(float(tts.get("noise_w", 0.8)), 0.0),
        preferences_relative_path=Path(
            str(persistence.get("preferences_path", "data/preferences.yaml"))
        ),
        spotify_token_relative_path=Path(
            str(spotify.get("token_path", "data/spotify.tokens.json"))
        ),
        spotify_redirect_uri=str(
            spotify.get(
                "redirect_uri",
                "http://127.0.0.1:8000/api/spotify/callback",
            )
        ).strip(),
        spotify_market=str(spotify.get("market", "ES")).strip().upper()[:2] or "ES",
    )
