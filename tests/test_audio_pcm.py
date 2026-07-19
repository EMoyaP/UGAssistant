from __future__ import annotations

from array import array
import io
import unittest
import wave

from ugassistant.domain.audio_pcm import pcm16_mono_to_wav, wav_duration_ms


class AudioPCMTests(unittest.TestCase):
    def test_resamples_pcm16_mono_to_whisper_wav(self) -> None:
        source = array("h", [1200] * 4800).tobytes()

        wav_bytes = pcm16_mono_to_wav(source, 48000, 16000)

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.getnframes(), 1600)
        self.assertEqual(wav_duration_ms(wav_bytes), 100)

    def test_rejects_an_empty_recording(self) -> None:
        with self.assertRaises(ValueError):
            pcm16_mono_to_wav(b"", 48000)


if __name__ == "__main__":
    unittest.main()
