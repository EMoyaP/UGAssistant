from __future__ import annotations

from array import array
import io
import sys
import wave


def pcm16_mono_to_wav(
    pcm_bytes: bytes,
    source_sample_rate: int,
    target_sample_rate: int = 16000,
) -> bytes:
    if source_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if len(pcm_bytes) < 2:
        raise ValueError("PCM recording is empty")
    if len(pcm_bytes) % 2:
        pcm_bytes = pcm_bytes[:-1]

    samples = array("h")
    samples.frombytes(pcm_bytes)
    if sys.byteorder == "big":
        samples.byteswap()
    if source_sample_rate != target_sample_rate:
        samples = _resample_linear(samples, source_sample_rate, target_sample_rate)

    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(target_sample_rate)
        if sys.byteorder == "big":
            samples.byteswap()
        wav_file.writeframes(samples.tobytes())
    return output.getvalue()


def wav_duration_ms(wav_bytes: bytes) -> int:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        if sample_rate <= 0:
            raise ValueError("WAV sample rate must be positive")
        return round(wav_file.getnframes() * 1000 / sample_rate)


def _resample_linear(
    samples: array[int],
    source_sample_rate: int,
    target_sample_rate: int,
) -> array[int]:
    if len(samples) <= 1:
        return array("h", samples)
    output_length = max(
        1,
        round(len(samples) * target_sample_rate / source_sample_rate),
    )
    if output_length == 1:
        return array("h", [samples[0]])

    scale = (len(samples) - 1) / (output_length - 1)
    output = array("h")
    for index in range(output_length):
        position = index * scale
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        fraction = position - lower
        value = round(samples[lower] + (samples[upper] - samples[lower]) * fraction)
        output.append(min(max(value, -32768), 32767))
    return output
