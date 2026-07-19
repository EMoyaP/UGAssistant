from __future__ import annotations

from array import array
from dataclasses import dataclass
import io
import math
import sys
import wave


@dataclass(frozen=True)
class LipSyncTrack:
    interval_ms: int
    duration_ms: int
    levels: tuple[float, ...]


def build_lip_sync_track(
    wav_bytes: bytes,
    *,
    interval_ms: int = 80,
) -> LipSyncTrack:
    if interval_ms < 20:
        raise ValueError("Lip sync interval must be at least 20 ms")
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        compression = wav_file.getcomptype()
        frames = wav_file.readframes(frame_count)

    if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
        raise ValueError("WAV stream has invalid lip sync metadata")
    if sample_width != 2 or compression != "NONE":
        raise ValueError("Lip sync requires uncompressed 16-bit PCM WAV")

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    samples_per_window = max(
        channels,
        round(sample_rate * interval_ms / 1000) * channels,
    )
    raw_levels: list[float] = []
    for start in range(0, len(samples), samples_per_window):
        window = samples[start : start + samples_per_window]
        if not window:
            continue
        mean_square = sum(sample * sample for sample in window) / len(window)
        raw_levels.append(math.sqrt(mean_square) / 32768.0)

    peak = max(raw_levels, default=0.0)
    if peak < 0.001:
        normalized = [0.0 for _level in raw_levels]
    else:
        noise_floor = min(peak * 0.12, 0.018)
        usable_range = max(peak - noise_floor, 0.001)
        normalized = [
            round(
                min(max((level - noise_floor) / usable_range, 0.0), 1.0) ** 0.65,
                3,
            )
            for level in raw_levels
        ]

    return LipSyncTrack(
        interval_ms=interval_ms,
        duration_ms=round(frame_count / sample_rate * 1000),
        levels=tuple(normalized),
    )
