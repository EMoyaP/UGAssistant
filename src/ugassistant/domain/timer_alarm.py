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
    """Build a short local alarm tone without external media files."""
    if duration_seconds <= 0 or sample_rate <= 0:
        raise ValueError("Alarm duration and sample rate must be positive")
    sample_count = round(duration_seconds * sample_rate)
    samples = array("h")
    for index in range(sample_count):
        elapsed = index / sample_rate
        pulse_position = elapsed % 0.4
        if pulse_position > 0.26:
            samples.append(0)
            continue
        frequency = 880.0 if int(elapsed / 0.4) % 2 == 0 else 660.0
        samples.append(round(9000 * math.sin(2 * math.pi * frequency * elapsed)))
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()
