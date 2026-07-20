from __future__ import annotations

import io
import math
import wave
from array import array


def build_timer_alarm_wav(
    *,
    duration_seconds: float = 2.0,
    sample_rate: int = 16000,
) -> bytes:
    """Build a two-tone alarm-clock chime without external media files."""
    if duration_seconds <= 0 or sample_rate <= 0:
        raise ValueError("Alarm duration and sample rate must be positive")
    sample_count = round(duration_seconds * sample_rate)
    samples = array("h")
    for index in range(sample_count):
        elapsed = index / sample_rate
        cycle_position = elapsed % 0.72
        if cycle_position >= 0.38:
            samples.append(0)
            continue
        note_position = cycle_position % 0.19
        if note_position >= 0.14:
            samples.append(0)
            continue
        frequency = 523.25 if cycle_position < 0.19 else 659.25
        envelope = min(note_position / 0.015, (0.14 - note_position) / 0.035, 1.0)
        sample = math.sin(2 * math.pi * frequency * elapsed)
        harmonic = 0.18 * math.sin(2 * math.pi * frequency * 2 * elapsed)
        samples.append(round(10000 * envelope * (sample + harmonic)))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()
