from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ugassistant.adapters.hand_perception import OpenCVHandPerception  # noqa: E402
from ugassistant.adapters.opencv_camera import OpenCVCameraAdapter  # noqa: E402
from ugassistant.config import load_app_settings  # noqa: E402


async def probe_camera(device_index: int) -> dict[str, object]:
    settings = load_app_settings(PROJECT_ROOT)
    hand_model_ready = (
        settings.hand_detection_enabled
        and settings.hand_palm_model_path.is_file()
        and settings.hand_pose_model_path.is_file()
    )
    hand_perception = None
    if hand_model_ready:
        hand_perception = OpenCVHandPerception(
            settings.hand_palm_model_path,
            settings.hand_pose_model_path,
            max_hands=settings.hand_max_hands,
            palm_score_threshold=settings.hand_palm_score_threshold,
            palm_nms_threshold=settings.hand_palm_nms_threshold,
            hand_score_threshold=settings.hand_score_threshold,
        )
    adapter = OpenCVCameraAdapter(
        settings.camera_model_path,
        device_index=device_index,
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
    try:
        await adapter.open()
        frame = await adapter.read_frame()
        return {
            "ok": True,
            "device_index": device_index,
            "width": frame.width,
            "height": frame.height,
            "person_detected": frame.presence.person_detected,
            "face_landmarks": (
                frame.presence.face_landmarks.to_dict()
                if frame.presence.face_landmarks is not None
                else None
            ),
            "hand_model_ready": hand_model_ready,
            "hands": [hand.to_dict() for hand in frame.hands],
            "finger_count": frame.finger_count,
            "combined_gestures": [
                gesture.to_dict() for gesture in frame.combined_gestures
            ],
            "detail": frame.presence.detail,
        }
    finally:
        await adapter.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Open and test one local camera frame.")
    parser.add_argument("--device-index", type=int, default=None)
    args = parser.parse_args()
    settings = load_app_settings(PROJECT_ROOT)
    device_index = (
        settings.camera_device_index
        if args.device_index is None
        else args.device_index
    )
    try:
        result = asyncio.run(probe_camera(device_index))
    except Exception as exc:
        result = {
            "ok": False,
            "device_index": device_index,
            "detail": str(exc),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
