from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ugassistant.adapters.hand_perception import OpenCVHandPerception
from ugassistant.adapters.home_assistant import (
    HomeAssistantRESTAdapter,
    LocalHomeAssistantTokenStore,
)
from ugassistant.adapters.ollama import OllamaAdapter
from ugassistant.adapters.opencv_camera import OpenCVCameraAdapter
from ugassistant.adapters.piper_tts import PiperTTSAdapter, PiperVoiceConfig
from ugassistant.adapters.portaudio import PortAudioAdapter
from ugassistant.adapters.preferences import YAMLPreferenceStore
from ugassistant.adapters.whisper_cpp_stt import WhisperCppSTTAdapter
from ugassistant.config import AppSettings, load_app_settings, load_model_lock
from ugassistant.domain.ports import (
    AudioAdapter,
    CameraAdapter,
    CombinedGesture,
    HandGesture,
    LLMAdapter,
    STTAdapter,
    TTSAdapter,
)
from ugassistant.domain.platform import detect_platform
from ugassistant.domain.preferences import UserPreferences
from ugassistant.domain.iot import HomeAssistantError, IoTAdapter
from ugassistant.domain.state_machine import (
    AssistantState,
    AssistantStateMachine,
    InvalidStateTransition,
)
from ugassistant.services.audio import AudioDeviceService, AudioStatus
from ugassistant.services.audio import (
    AudioCaptureCancelledError,
    NoSpeechDetectedError,
)
from ugassistant.services.camera import CameraService, CameraStatus
from ugassistant.services.conversation import ConversationService, ConversationStatus
from ugassistant.services.recognition import (
    RecognitionBusyError,
    RecognitionStatus,
    UnsupportedRecognitionLanguageError,
    VoiceRecognitionService,
)
from ugassistant.services.speech import SpeechBusyError, SpeechService, SpeechStatus
from ugassistant.services.iot import IoTService
from ugassistant.services.timers import TimerService
from ugassistant.services.voice_assistant import (
    VoiceAssistantService,
    VoiceAssistantStatus,
)

logger = logging.getLogger("ugassistant")


class SpeechRequest(BaseModel):
    text: str


class ConversationRequest(BaseModel):
    text: str
    language: str = "es"
    response_detail: str = "short"


class AssistantProfileRequest(BaseModel):
    spanish_wake_word: str
    french_wake_word: str


class IoTConfigurationRequest(BaseModel):
    home_assistant_url: str = ""
    token: str = ""


class IoTControlRequest(BaseModel):
    action: str


class StateConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(websocket)

    async def broadcast(self, payload: dict[str, object]) -> None:
        async with self._lock:
            connections = list(self._connections)
        for websocket in connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                logger.exception("websocket_broadcast_failed")
                await self.disconnect(websocket)


def create_app(
    settings: AppSettings | None = None,
    camera_adapter: CameraAdapter | None = None,
    audio_adapter: AudioAdapter | None = None,
    tts_adapter: TTSAdapter | None = None,
    stt_adapter: STTAdapter | None = None,
    preference_store: YAMLPreferenceStore | None = None,
    llm_adapter: LLMAdapter | None = None,
    iot_adapter: IoTAdapter | None = None,
) -> FastAPI:
    settings = settings or load_app_settings()
    state_machine = AssistantStateMachine()
    state_manager = StateConnectionManager()
    camera_manager = StateConnectionManager()
    audio_manager = StateConnectionManager()
    speech_manager = StateConnectionManager()
    recognition_manager = StateConnectionManager()
    conversation_manager = StateConnectionManager()
    assistant_manager = StateConnectionManager()
    iot_manager = StateConnectionManager()
    static_dir = Path(__file__).resolve().parents[1] / "web" / "static"
    platform_info = detect_platform()
    preference_store = preference_store or YAMLPreferenceStore(settings.preferences_path)
    try:
        preferences = preference_store.load()
    except Exception:
        logger.exception("preferences_load_failed")
        preferences = None

    if audio_adapter is None:
        audio_adapter = PortAudioAdapter()
    if tts_adapter is None:
        piper_executable = settings.tts_executable_path(
            platform_info.system,
            platform_info.machine,
        )
        if piper_executable is None:
            piper_executable = settings.project_root / "tools" / "piper" / "unsupported"
        tts_adapter = PiperTTSAdapter(
            piper_executable,
            (
                PiperVoiceConfig(
                    voice_id=settings.tts_voice_id,
                    display_name=settings.tts_voice_name,
                    language=settings.tts_language,
                    model_path=settings.tts_model_path,
                    config_path=settings.tts_config_path,
                    length_scale=settings.tts_length_scale,
                    noise_scale=settings.tts_noise_scale,
                    noise_w=settings.tts_noise_w,
                ),
                PiperVoiceConfig(
                    voice_id=settings.tts_french_voice_id,
                    display_name=settings.tts_french_voice_name,
                    language=settings.tts_french_language,
                    model_path=settings.tts_french_model_path,
                    config_path=settings.tts_french_config_path,
                    length_scale=settings.tts_length_scale,
                    noise_scale=settings.tts_noise_scale,
                    noise_w=settings.tts_noise_w,
                ),
            ),
            process_timeout_seconds=settings.tts_process_timeout_seconds,
        )
    if stt_adapter is None:
        whisper_executable = settings.stt_executable_path(
            platform_info.system,
            platform_info.machine,
        )
        if whisper_executable is None:
            whisper_executable = (
                settings.project_root / "tools" / "whisper" / "unsupported"
            )
        stt_adapter = WhisperCppSTTAdapter(
            whisper_executable,
            settings.stt_model_path,
            threads=settings.stt_threads,
            process_timeout_seconds=settings.stt_process_timeout_seconds,
            candidate_languages=settings.stt_accepted_languages,
        )
    if llm_adapter is None:
        llm_adapter = OllamaAdapter(
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    hand_models_ready = (
        settings.hand_detection_enabled
        and settings.hand_palm_model_path.is_file()
        and settings.hand_pose_model_path.is_file()
    )
    if camera_adapter is None:
        hand_perception = None
        if hand_models_ready:
            hand_perception = OpenCVHandPerception(
                settings.hand_palm_model_path,
                settings.hand_pose_model_path,
                max_hands=settings.hand_max_hands,
                palm_score_threshold=settings.hand_palm_score_threshold,
                palm_nms_threshold=settings.hand_palm_nms_threshold,
                hand_score_threshold=settings.hand_score_threshold,
            )
        camera_adapter = OpenCVCameraAdapter(
            settings.camera_model_path,
            device_index=settings.camera_device_index,
            width=settings.max_camera_width,
            height=settings.max_camera_height,
            fps=settings.camera_preview_fps,
            mirror=settings.camera_mirror_preview,
            score_threshold=settings.camera_score_threshold,
            nms_threshold=settings.camera_nms_threshold,
            hand_perception=hand_perception,
            hand_inference_interval_frames=settings.hand_inference_interval_frames,
            finger_count_stable_samples=settings.finger_count_stable_samples,
        )

    async def broadcast_state() -> None:
        await camera_service.set_activity(state_machine.state)
        await state_manager.broadcast(state_machine.snapshot().to_dict())

    silence_gesture_active = False
    end_gesture_active = False

    async def on_camera_status(status: CameraStatus) -> None:
        nonlocal silence_gesture_active, end_gesture_active
        await camera_manager.broadcast(status.to_dict())
        has_silence_gesture = any(
            gesture.gesture == CombinedGesture.POINTING_AT_MOUTH
            for gesture in status.combined_gestures
        )
        if has_silence_gesture and not silence_gesture_active:
            silence_gesture_active = True
            if speech_service.status.busy:
                voice_assistant_service.request_end_session()
                await speech_service.interrupt()
        elif not has_silence_gesture:
            silence_gesture_active = False
        has_end_gesture = (
            any(hand.gesture == HandGesture.THUMB_DOWN for hand in status.hands)
            or any(
                gesture.gesture == CombinedGesture.THUMB_DOWN_NEAR_FACE
                for gesture in status.combined_gestures
            )
        )
        if has_end_gesture and not end_gesture_active:
            end_gesture_active = True
            if voice_assistant_service.request_end_session():
                await speech_service.interrupt(cancel_task=False)
        elif not has_end_gesture:
            end_gesture_active = False
        current_state = state_machine.state
        if status.person_detected:
            if current_state == AssistantState.SLEEPING:
                state_machine.transition_to(
                    AssistantState.IDLE,
                    detail="camera-wake-up",
                )
                await broadcast_state()
            elif current_state == AssistantState.IDLE:
                state_machine.transition_to(
                    AssistantState.PERSON_DETECTED,
                    detail="camera-presence",
                )
                await broadcast_state()
        elif current_state == AssistantState.PERSON_DETECTED:
            state_machine.transition_to(
                AssistantState.IDLE,
                detail="camera-presence-lost",
            )
            await broadcast_state()

        if not status.available and status.detail not in {
            "camera_disabled",
            "camera_closed",
            "camera_selected",
            "no_camera_selected",
        }:
            logger.warning("camera_unavailable detail=%s", status.detail)
            # Vision is optional: a disconnected camera must not disable voice use.
            if state_machine.state == AssistantState.ERROR:
                state_machine.transition_to(
                    AssistantState.IDLE,
                    detail="camera-unavailable",
                    force=True,
                )
            await broadcast_state()

    camera_service = CameraService(
        camera_adapter,
        target_fps=settings.camera_preview_fps,
        idle_fps=settings.camera_idle_fps,
        person_detected_fps=settings.camera_person_detected_fps,
        processing_fps=settings.camera_processing_fps,
        gesture_fps=settings.camera_gesture_fps,
        model_ready=settings.camera_model_path.is_file(),
        hand_model_ready=hand_models_ready,
        selected_device_index=settings.camera_device_index,
        on_status=on_camera_status,
    )

    voice_assistant_service: VoiceAssistantService | None = None

    async def on_audio_status(status: AudioStatus) -> None:
        await audio_manager.broadcast(status.to_dict())
        current_snapshot = state_machine.snapshot()
        state_changed = False
        if status.sound_detected:
            if current_snapshot.state == AssistantState.SLEEPING:
                state_machine.transition_to(
                    AssistantState.IDLE,
                    detail="audio-wake-up",
                )
                current_snapshot = state_machine.snapshot()
            if current_snapshot.state in {
                AssistantState.IDLE,
                AssistantState.PERSON_DETECTED,
            }:
                state_machine.transition_to(
                    AssistantState.LISTENING,
                    detail="audio-activity",
                )
                state_changed = True
        elif (
            current_snapshot.state == AssistantState.LISTENING
            and current_snapshot.detail == "audio-activity"
        ):
            target_state = (
                AssistantState.PERSON_DETECTED
                if camera_service.status.person_detected
                else AssistantState.IDLE
            )
            state_machine.transition_to(
                target_state,
                detail="audio-silence",
            )
            state_changed = True
        if state_changed:
            await broadcast_state()
        if voice_assistant_service is not None:
            voice_assistant_service.observe_audio(status)

    audio_service = AudioDeviceService(
        audio_adapter,
        activation_threshold=settings.audio_activation_threshold,
        release_threshold=settings.audio_release_threshold,
        activation_samples=settings.audio_activation_samples,
        silence_seconds=settings.audio_silence_seconds,
        block_duration_ms=settings.audio_block_duration_ms,
        output_volume=(
            preferences.output_volume
            if preferences is not None
            else settings.audio_output_volume
        ),
        on_status=on_audio_status,
    )
    if preferences is not None:
        audio_service.restore_device_preference(
            "input",
            preferences.microphone_device,
        )
        audio_service.restore_device_preference(
            "output",
            preferences.speaker_device,
        )

    speech_state_active = False

    async def on_speech_status(status: SpeechStatus) -> None:
        nonlocal speech_state_active
        await speech_manager.broadcast(status.to_dict())
        state_changed = False
        if status.phase == "synthesizing":
            speech_state_active = True
            state_machine.transition_to(
                AssistantState.THINKING,
                detail="tts-synthesizing",
                force=True,
            )
            state_changed = True
        elif status.phase == "playing":
            speech_state_active = True
            state_machine.transition_to(
                AssistantState.SPEAKING,
                detail="tts-playing",
                force=True,
            )
            state_changed = True
        elif status.phase == "error":
            speech_state_active = False
        elif speech_state_active and not status.busy:
            speech_state_active = False
            target_state = (
                AssistantState.PERSON_DETECTED
                if camera_service.status.person_detected
                else AssistantState.IDLE
            )
            state_machine.transition_to(
                target_state,
                detail="tts-completed",
                force=True,
            )
            state_changed = True
        if state_changed:
            await broadcast_state()

    def current_spatial_balance() -> float:
        camera_status = camera_service.status
        if (
            not settings.audio_spatial_pan_enabled
            or not camera_status.person_detected
            or camera_status.face_center_x is None
        ):
            return 0.0
        horizontal_offset = (camera_status.face_center_x - 0.5) * 2.0
        return min(max(horizontal_offset, -1.0), 1.0) * settings.audio_spatial_pan_max

    speech_service = SpeechService(
        tts_adapter,
        audio_service,
        default_voice_id=settings.tts_voice_id,
        default_speech_rate=(
            preferences.speech_rate
            if preferences is not None
            else settings.tts_speech_rate
        ),
        max_text_length=settings.tts_max_text_length,
        output_guard_seconds=settings.tts_output_guard_seconds,
        chunk_pause_seconds=settings.tts_chunk_pause_seconds,
        on_status=on_speech_status,
        balance_provider=current_spatial_balance,
    )

    async def on_recognition_status(status: RecognitionStatus) -> None:
        await recognition_manager.broadcast(status.to_dict())
        state_changed = False
        if status.phase == "listening":
            state_machine.transition_to(
                AssistantState.LISTENING,
                detail="stt-listening",
                force=True,
            )
            state_changed = True
        elif status.phase in {"transcribing", "recognized"}:
            state_machine.transition_to(
                AssistantState.TRANSCRIBING,
                detail=f"stt-{status.phase}",
                force=True,
            )
            state_changed = True
        elif status.phase in {
            "cancelled",
            "completed",
            "timeout",
            "unsupported_language",
        }:
            target_state = (
                AssistantState.PERSON_DETECTED
                if camera_service.status.person_detected
                else AssistantState.IDLE
            )
            state_machine.transition_to(
                target_state,
                detail=f"stt-{status.phase}",
                force=True,
            )
            state_changed = True
        elif status.phase == "error":
            if "did not recognize any text" in status.detail.casefold():
                target_state = (
                    AssistantState.PERSON_DETECTED
                    if camera_service.status.person_detected
                    else AssistantState.IDLE
                )
                state_machine.transition_to(
                    target_state,
                    detail="stt-no-speech-recognized",
                    force=True,
                )
            else:
                state_machine.fail(f"stt: {status.detail}")
            state_changed = True
        if state_changed:
            await broadcast_state()

    heavy_inference_lock = asyncio.Lock()
    recognition_service = VoiceRecognitionService(
        stt_adapter,
        audio_service,
        speech_service,
        accepted_languages=settings.stt_accepted_languages,
        sample_rate=settings.stt_sample_rate,
        wait_for_speech_seconds=settings.stt_wait_for_speech_seconds,
        silence_seconds=settings.stt_silence_seconds,
        max_recording_seconds=settings.stt_max_recording_seconds,
        pre_roll_seconds=settings.stt_pre_roll_seconds,
        inference_lock=heavy_inference_lock,
        on_status=on_recognition_status,
    )

    async def on_conversation_status(status: ConversationStatus) -> None:
        await conversation_manager.broadcast(status.to_dict())
        if status.phase == "thinking":
            state_machine.transition_to(
                AssistantState.THINKING,
                detail="ollama-thinking",
                force=True,
            )
            await broadcast_state()
        elif status.phase == "completed":
            target_state = (
                AssistantState.PERSON_DETECTED
                if camera_service.status.person_detected
                else AssistantState.IDLE
            )
            state_machine.transition_to(
                target_state,
                detail="ollama-completed",
                force=True,
            )
            await broadcast_state()
        elif status.phase == "error":
            state_machine.fail(f"ollama: {status.detail}")
            await broadcast_state()

    conversation_service = ConversationService(
        llm_adapter,
        inference_lock=heavy_inference_lock,
        max_history_turns=settings.llm_max_history_turns,
        short_context_tokens=settings.llm_short_context_tokens,
        complete_context_tokens=settings.llm_complete_context_tokens,
        max_tokens=settings.llm_max_tokens,
        complete_max_tokens=settings.llm_complete_max_tokens,
        temperature=settings.llm_temperature,
        on_status=on_conversation_status,
    )

    if iot_adapter is None:
        iot_adapter = HomeAssistantRESTAdapter(
            LocalHomeAssistantTokenStore(settings.home_assistant_token_path)
        )
    iot_service = IoTService(iot_adapter)
    if preferences is not None and preferences.home_assistant_url:
        iot_service.configure(preferences.home_assistant_url, "")

    async def publish_iot_status() -> dict[str, object]:
        payload = iot_service.status.to_dict()
        await iot_manager.broadcast(payload)
        return payload

    async def on_voice_assistant_status(status: VoiceAssistantStatus) -> None:
        await assistant_manager.broadcast(status.to_dict())

    timer_service = TimerService()
    voice_assistant_service = VoiceAssistantService(
        audio_service,
        recognition_service,
        speech_service,
        conversation_service,
        timer_service=timer_service,
        spanish_wake_words=settings.wake_spanish_words,
        french_wake_words=settings.wake_french_words,
        spanish_greeting=settings.wake_spanish_greeting,
        french_greeting=settings.wake_french_greeting,
        follow_up_wait_seconds=5.0,
        on_status=on_voice_assistant_status,
    )
    timer_service.set_callbacks(
        on_status=voice_assistant_service.update_timers,
        on_expired=voice_assistant_service.notify_timer_expired,
    )
    if preferences is not None:
        voice_assistant_service.configure_wake_words(
            preferences.spanish_wake_word,
            preferences.french_wake_word,
        )

    async def snapshot_preferences() -> UserPreferences:
        camera_preference = await camera_service.device_preference()
        speech_status = speech_service.status
        return UserPreferences(
            camera_device=camera_preference,
            camera_enabled=camera_service.status.enabled,
            microphone_device=audio_service.device_preference("input"),
            microphone_enabled=audio_service.status.monitoring,
            speaker_device=audio_service.device_preference("output"),
            speaker_enabled=audio_service.status.output_enabled,
            output_volume=audio_service.status.output_volume,
            voice_id=speech_status.selected_voice_id or settings.tts_voice_id,
            language=speech_status.selected_language or settings.tts_language,
            speech_rate=speech_status.speech_rate,
            spanish_wake_word=(
                voice_assistant_service.wake_words["es"][0]
                if voice_assistant_service.wake_words["es"]
                else settings.wake_spanish_words[0]
            ),
            french_wake_word=(
                voice_assistant_service.wake_words["fr"][0]
                if voice_assistant_service.wake_words["fr"]
                else settings.wake_french_words[0]
            ),
            home_assistant_url=(
                preferences.home_assistant_url if preferences is not None else ""
            ),
        )

    async def save_preferences(updated: UserPreferences) -> None:
        nonlocal preferences
        try:
            await asyncio.to_thread(preference_store.save, updated)
        except Exception:
            logger.exception("preferences_save_failed")
            raise
        preferences = updated

    async def update_preferences(**changes: object) -> None:
        current = preferences or await snapshot_preferences()
        await save_preferences(replace(current, **changes))

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            await audio_service.refresh()
            if preferences is not None:
                if (
                    preferences.speaker_enabled
                    and audio_service.status.selected_output_index is not None
                ):
                    await audio_service.enable_output()
                else:
                    await audio_service.disable_output()
                if (
                    preferences.microphone_enabled
                    and audio_service.status.selected_input_index is not None
                ):
                    await audio_service.enable_monitoring()
        except Exception:
            logger.exception("audio_device_startup_enumeration_failed")
        try:
            await speech_service.refresh()
            if preferences is not None:
                try:
                    await speech_service.select_voice(preferences.voice_id)
                except ValueError:
                    old_voice_id = preferences.voice_id
                    status = await speech_service.select_language(
                        preferences.language
                    )
                    await update_preferences(
                        voice_id=(
                            status.selected_voice_id or settings.tts_voice_id
                        ),
                        language=(
                            status.selected_language or settings.tts_language
                        ),
                    )
                    logger.info(
                        "tts_preference_migrated old_voice=%s new_voice=%s",
                        old_voice_id,
                        status.selected_voice_id,
                    )
        except Exception:
            logger.exception("tts_startup_scan_failed")
        try:
            await recognition_service.refresh()
        except Exception:
            logger.exception("stt_startup_scan_failed")
        try:
            await conversation_service.refresh()
        except Exception:
            logger.exception("ollama_startup_scan_failed")
        try:
            await iot_service.refresh()
        except Exception:
            logger.exception("iot_startup_scan_failed")
        try:
            if preferences is not None:
                await camera_service.restore_device_preference(
                    preferences.camera_device
                )
                if (
                    preferences.camera_enabled
                    and camera_service.status.selected_device_index is not None
                ):
                    await camera_service.enable()
            elif settings.camera_enabled_by_default:
                await camera_service.enable()
        except Exception:
            logger.exception("camera_startup_failed")
        if preferences is None:
            try:
                await save_preferences(await snapshot_preferences())
            except Exception:
                logger.exception("preferences_initial_save_failed")
        await timer_service.start()
        yield
        await timer_service.shutdown()
        await voice_assistant_service.shutdown()
        await recognition_service.shutdown()
        await audio_service.shutdown()
        await camera_service.shutdown()

    app = FastAPI(title="UGAssistant", version="0.9.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.state_machine = state_machine
    app.state.manager = state_manager
    app.state.camera_manager = camera_manager
    app.state.audio_manager = audio_manager
    app.state.speech_manager = speech_manager
    app.state.recognition_manager = recognition_manager
    app.state.conversation_manager = conversation_manager
    app.state.assistant_manager = assistant_manager
    app.state.iot_manager = iot_manager
    app.state.camera_service = camera_service
    app.state.audio_service = audio_service
    app.state.speech_service = speech_service
    app.state.recognition_service = recognition_service
    app.state.conversation_service = conversation_service
    app.state.voice_assistant_service = voice_assistant_service
    app.state.timer_service = timer_service
    app.state.iot_service = iot_service
    app.state.preference_store = preference_store
    app.state.shutdown_callback = None

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/debug/camera")
    async def camera_debug() -> FileResponse:
        return FileResponse(static_dir / "camera_debug.html")

    @app.get("/debug/audio")
    async def audio_debug() -> FileResponse:
        return FileResponse(static_dir / "audio_debug.html")

    @app.get("/debug/iot")
    async def iot_debug() -> FileResponse:
        return FileResponse(static_dir / "iot_debug.html")

    @app.get("/health")
    async def health() -> dict[str, object]:
        model_lock = load_model_lock(settings.project_root)
        return {
            "status": "ok",
            "state": state_machine.state.value,
            "platform": platform_info.__dict__,
            "locked_models": len(model_lock.get("models", [])),
            "camera": camera_service.status.to_dict(),
            "audio": audio_service.status.to_dict(),
            "tts": speech_service.status.to_dict(),
            "stt": recognition_service.status.to_dict(),
            "llm": conversation_service.status.to_dict(),
            "iot": iot_service.status.to_dict(),
            "voice_assistant": voice_assistant_service.status.to_dict(),
        }

    @app.get("/api/state")
    async def get_state() -> dict[str, object]:
        return state_machine.snapshot().to_dict()

    @app.get("/api/preferences")
    async def get_preferences() -> dict[str, object]:
        current = preferences or await snapshot_preferences()
        return current.to_dict()

    @app.get("/api/assistant/profile")
    async def get_assistant_profile() -> dict[str, object]:
        current = preferences or await snapshot_preferences()
        return {
            "spanish_wake_word": current.spanish_wake_word,
            "french_wake_word": current.french_wake_word,
        }

    @app.put("/api/assistant/profile")
    async def update_assistant_profile(
        request: AssistantProfileRequest,
    ) -> dict[str, object]:
        spanish = " ".join(request.spanish_wake_word.split())[:40]
        french = " ".join(request.french_wake_word.split())[:40]
        if not spanish or not french:
            raise HTTPException(status_code=422, detail="Profile values cannot be empty")
        voice_assistant_service.configure_wake_words(spanish, french)
        await update_preferences(
            spanish_wake_word=spanish,
            french_wake_word=french,
        )
        return await get_assistant_profile()

    @app.get("/api/iot")
    async def get_iot_status() -> dict[str, object]:
        try:
            await iot_service.refresh()
        except Exception as exc:
            logger.exception("iot_refresh_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return await publish_iot_status()

    @app.get("/api/iot/config")
    async def get_iot_configuration() -> dict[str, object]:
        current = preferences or await snapshot_preferences()
        return {
            "home_assistant_url": current.home_assistant_url,
            "token_configured": iot_service.status.configured,
        }

    @app.put("/api/iot/config")
    async def update_iot_configuration(
        request: IoTConfigurationRequest,
    ) -> dict[str, object]:
        current = preferences or await snapshot_preferences()
        url = request.home_assistant_url.strip() or current.home_assistant_url
        try:
            iot_service.configure(url, request.token)
            status = await iot_service.refresh()
        except (ValueError, HomeAssistantError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        await update_preferences(home_assistant_url=url)
        await iot_manager.broadcast(status.to_dict())
        return status.to_dict()

    @app.post("/api/iot/entities/{entity_id:path}")
    async def control_iot_entity(
        entity_id: str,
        request: IoTControlRequest,
    ) -> dict[str, object]:
        try:
            status = await iot_service.control(entity_id, request.action)
        except (ValueError, HomeAssistantError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        await iot_manager.broadcast(status.to_dict())
        return status.to_dict()

    @app.post("/api/system/shutdown")
    async def shutdown_system() -> dict[str, str]:
        callback = app.state.shutdown_callback
        if not callable(callback):
            raise HTTPException(
                status_code=409,
                detail="System shutdown is unavailable in this runtime",
            )
        asyncio.get_running_loop().call_later(0.15, callback)
        return {"status": "shutting_down"}

    @app.post("/api/state/{state_name}")
    async def set_state(state_name: str, force: bool = False) -> dict[str, object]:
        try:
            next_state = AssistantState[state_name.upper()]
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown state") from exc

        try:
            snapshot = state_machine.transition_to(
                next_state,
                detail="manual-dev-control" if force else "manual-transition",
                force=force,
            )
        except InvalidStateTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        payload = snapshot.to_dict()
        await state_manager.broadcast(payload)
        return payload

    @app.get("/api/camera")
    async def get_camera_status() -> dict[str, object]:
        return camera_service.status.to_dict()

    @app.get("/api/camera/devices")
    async def get_camera_devices() -> dict[str, object]:
        devices = await camera_service.list_devices()
        return {
            "devices": [device.to_dict() for device in devices],
            "selected_device_index": camera_service.status.selected_device_index,
        }

    @app.get("/api/audio/devices")
    async def get_audio_devices() -> dict[str, object]:
        try:
            status = await audio_service.refresh()
        except Exception as exc:
            logger.exception("audio_device_enumeration_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return status.to_dict()

    @app.get("/api/audio")
    async def get_audio_status() -> dict[str, object]:
        return audio_service.status.to_dict()

    @app.post("/api/audio/select/{kind}/{device_index}")
    async def select_audio_device(
        kind: str,
        device_index: int,
    ) -> dict[str, object]:
        selected = None if device_index < 0 else device_index
        try:
            status = await audio_service.select_device(kind, selected)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("audio_device_selection_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if kind == "input":
            await update_preferences(
                microphone_device=audio_service.device_preference("input"),
                microphone_enabled=status.monitoring,
            )
        else:
            await update_preferences(
                speaker_device=audio_service.device_preference("output"),
                speaker_enabled=status.output_enabled,
            )
        return status.to_dict()

    @app.post("/api/audio/enable")
    async def enable_audio_monitor() -> dict[str, object]:
        try:
            status = await audio_service.enable_monitoring()
        except Exception as exc:
            logger.exception("audio_monitor_enable_failed")
            state_machine.fail(f"audio: {exc}")
            await broadcast_state()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if (
            state_machine.state == AssistantState.ERROR
            and (state_machine.snapshot().detail or "").startswith("audio:")
        ):
            state_machine.transition_to(
                AssistantState.IDLE,
                detail="audio-recovered",
            )
            await broadcast_state()
        await update_preferences(microphone_enabled=status.monitoring)
        return status.to_dict()

    @app.post("/api/audio/disable")
    async def disable_audio_monitor() -> dict[str, object]:
        try:
            status = await audio_service.disable_monitoring()
        except Exception as exc:
            logger.exception("audio_monitor_disable_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        await update_preferences(microphone_enabled=False)
        return status.to_dict()

    @app.post("/api/audio/output/enable")
    async def enable_audio_output() -> dict[str, object]:
        try:
            status = await audio_service.enable_output()
        except Exception as exc:
            logger.exception("audio_output_enable_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        await update_preferences(speaker_enabled=True)
        return status.to_dict()

    @app.post("/api/audio/output/disable")
    async def disable_audio_output() -> dict[str, object]:
        try:
            status = await audio_service.disable_output()
        except Exception as exc:
            logger.exception("audio_output_disable_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        await update_preferences(speaker_enabled=False)
        return status.to_dict()

    @app.post("/api/audio/output/volume/{volume_percent}")
    async def set_audio_output_volume(volume_percent: int) -> dict[str, object]:
        try:
            status = await audio_service.set_output_volume(volume_percent / 100.0)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        await update_preferences(output_volume=status.output_volume)
        return status.to_dict()

    @app.get("/api/tts")
    async def get_tts_status() -> dict[str, object]:
        try:
            status = await speech_service.refresh()
        except Exception as exc:
            logger.exception("tts_status_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return status.to_dict()

    @app.post("/api/tts/select/{voice_id}")
    async def select_tts_voice(voice_id: str) -> dict[str, object]:
        try:
            status = await speech_service.select_voice(voice_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("tts_voice_selection_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        await update_preferences(
            voice_id=status.selected_voice_id or settings.tts_voice_id,
            language=status.selected_language or settings.tts_language,
        )
        return status.to_dict()

    @app.post("/api/tts/language/{language}")
    async def select_tts_language(language: str) -> dict[str, object]:
        try:
            status = await speech_service.select_language(language)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        await update_preferences(
            voice_id=status.selected_voice_id or settings.tts_voice_id,
            language=status.selected_language or settings.tts_language,
        )
        return status.to_dict()

    @app.post("/api/tts/speed/{speed_percent}")
    async def set_tts_speed(speed_percent: int) -> dict[str, object]:
        try:
            status = await speech_service.set_speech_rate(speed_percent / 100.0)
        except SpeechBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        await update_preferences(speech_rate=status.speech_rate)
        return status.to_dict()

    @app.post("/api/tts/speak")
    async def speak_text(request: SpeechRequest) -> dict[str, object]:
        try:
            status = await speech_service.speak(request.text)
        except SpeechBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("tts_speak_failed")
            state_machine.fail(f"tts: {exc}")
            await broadcast_state()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return status.to_dict()

    @app.get("/api/stt")
    async def get_stt_status() -> dict[str, object]:
        try:
            status = await recognition_service.refresh()
        except Exception as exc:
            logger.exception("stt_status_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return status.to_dict()

    @app.post("/api/stt/recognize")
    async def recognize_voice() -> dict[str, object]:
        try:
            status = await recognition_service.recognize_and_repeat()
        except RecognitionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except AudioCaptureCancelledError:
            return recognition_service.status.to_dict()
        except NoSpeechDetectedError as exc:
            raise HTTPException(status_code=408, detail=str(exc)) from exc
        except UnsupportedRecognitionLanguageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("stt_recognition_failed")
            if state_machine.state != AssistantState.ERROR:
                state_machine.fail(f"stt: {exc}")
                await broadcast_state()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        speech_status = speech_service.status
        await update_preferences(
            voice_id=speech_status.selected_voice_id or settings.tts_voice_id,
            language=speech_status.selected_language or settings.tts_language,
        )
        return status.to_dict()

    @app.post("/api/stt/cancel")
    async def cancel_voice_recognition() -> dict[str, object]:
        return recognition_service.cancel().to_dict()

    @app.get("/api/llm")
    async def get_llm_status() -> dict[str, object]:
        try:
            return (await conversation_service.refresh()).to_dict()
        except Exception as exc:
            logger.exception("ollama_status_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/llm/ask")
    async def ask_llm(request: ConversationRequest) -> dict[str, object]:
        try:
            answer = await conversation_service.answer(
                request.text,
                request.language,
                request.response_detail,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"answer": answer, "status": conversation_service.status.to_dict()}

    @app.get("/api/assistant")
    async def get_voice_assistant_status() -> dict[str, object]:
        return voice_assistant_service.status.to_dict()

    @app.post("/api/camera/select/{device_index}")
    async def select_camera(
        device_index: int,
        enable: bool | None = None,
    ) -> dict[str, object]:
        selected = None if device_index < 0 else device_index
        should_enable = (
            selected is not None
            if enable is None
            else bool(enable and selected is not None)
        )
        try:
            status = await camera_service.select_device(selected, enable=should_enable)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("camera_select_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        await update_preferences(
            camera_device=await camera_service.device_preference(),
            camera_enabled=status.enabled,
        )
        return status.to_dict()

    @app.post("/api/camera/enable")
    async def enable_camera() -> dict[str, object]:
        try:
            status = await camera_service.enable()
        except Exception as exc:
            logger.exception("camera_enable_failed")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if (
            state_machine.state == AssistantState.ERROR
            and (state_machine.snapshot().detail or "").startswith("camera:")
        ):
            state_machine.transition_to(
                AssistantState.IDLE,
                detail="camera-recovered",
            )
            await broadcast_state()
        await update_preferences(camera_enabled=True)
        return status.to_dict()

    @app.post("/api/camera/disable")
    async def disable_camera() -> dict[str, object]:
        status = await camera_service.disable()
        await update_preferences(camera_enabled=False)
        return status.to_dict()

    @app.get("/api/camera/stream")
    async def camera_stream() -> StreamingResponse:
        if not camera_service.status.enabled:
            raise HTTPException(status_code=409, detail="Camera is disabled")
        return StreamingResponse(
            camera_service.mjpeg_stream(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/ws/state")
    async def state_socket(websocket: WebSocket) -> None:
        await state_manager.connect(websocket)
        await websocket.send_json(state_machine.snapshot().to_dict())
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await state_manager.disconnect(websocket)
        except Exception:
            logger.exception("state_websocket_error")
            await state_manager.disconnect(websocket)

    @app.websocket("/ws/camera")
    async def camera_socket(websocket: WebSocket) -> None:
        await camera_manager.connect(websocket)
        await websocket.send_json(camera_service.status.to_dict())
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await camera_manager.disconnect(websocket)
        except Exception:
            logger.exception("camera_websocket_error")
            await camera_manager.disconnect(websocket)

    @app.websocket("/ws/audio")
    async def audio_socket(websocket: WebSocket) -> None:
        await audio_manager.connect(websocket)
        await websocket.send_json(audio_service.status.to_dict())
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await audio_manager.disconnect(websocket)
        except Exception:
            logger.exception("audio_websocket_error")
            await audio_manager.disconnect(websocket)

    @app.websocket("/ws/tts")
    async def tts_socket(websocket: WebSocket) -> None:
        await speech_manager.connect(websocket)
        await websocket.send_json(speech_service.status.to_dict())
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await speech_manager.disconnect(websocket)
        except Exception:
            logger.exception("tts_websocket_error")
            await speech_manager.disconnect(websocket)

    @app.websocket("/ws/stt")
    async def stt_socket(websocket: WebSocket) -> None:
        await recognition_manager.connect(websocket)
        await websocket.send_json(recognition_service.status.to_dict())
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await recognition_manager.disconnect(websocket)
        except Exception:
            logger.exception("stt_websocket_error")
            await recognition_manager.disconnect(websocket)

    @app.websocket("/ws/llm")
    async def llm_socket(websocket: WebSocket) -> None:
        await conversation_manager.connect(websocket)
        await websocket.send_json(conversation_service.status.to_dict())
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await conversation_manager.disconnect(websocket)
        except Exception:
            logger.exception("llm_websocket_error")
            await conversation_manager.disconnect(websocket)

    @app.websocket("/ws/assistant")
    async def voice_assistant_socket(websocket: WebSocket) -> None:
        await assistant_manager.connect(websocket)
        await websocket.send_json(voice_assistant_service.status.to_dict())
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await assistant_manager.disconnect(websocket)
        except Exception:
            logger.exception("voice_assistant_websocket_error")
            await assistant_manager.disconnect(websocket)

    @app.websocket("/ws/iot")
    async def iot_socket(websocket: WebSocket) -> None:
        await iot_manager.connect(websocket)
        await websocket.send_json(iot_service.status.to_dict())
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            await iot_manager.disconnect(websocket)
        except Exception:
            logger.exception("iot_websocket_error")
            await iot_manager.disconnect(websocket)

    return app
