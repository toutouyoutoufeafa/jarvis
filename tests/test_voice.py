"""Audio plumbing tests: sample rates, normalisation, Piper discovery.

These cover the parts of the voice loop that can be checked without a
microphone or a loaded Whisper model.
"""
from __future__ import annotations

import numpy as np
import pytest

from jarvis import tts as tts_module
from jarvis.stt import STTEngine
from jarvis.tts import TTSEngine

PIPER_RATE = 22050  # what the fr_FR-siwis-medium voice actually renders


class TestResampling:
    def test_piper_rate_is_converted_to_whisper_rate(self) -> None:
        engine = STTEngine()
        audio = np.zeros(PIPER_RATE, dtype=np.int16)  # one second of silence

        out = engine._resample(audio, PIPER_RATE)

        assert len(out) == 16000
        # Without this, Whisper reads a 22050 Hz signal as 16000 Hz and the
        # speech comes out ~28% too fast.
        assert 0.70 < len(out) / len(audio) < 0.75

    def test_already_correct_rate_is_returned_unchanged(self) -> None:
        engine = STTEngine()
        audio = np.arange(16000, dtype=np.int16)

        out = engine._resample(audio, 16000)

        assert np.array_equal(out, audio)

    def test_stereo_free_dtype_is_preserved(self) -> None:
        engine = STTEngine()
        audio = np.ones(44100, dtype=np.float32)

        assert engine._resample(audio, 44100).dtype == np.float32

    def test_empty_audio_does_not_explode(self) -> None:
        engine = STTEngine()
        assert len(engine._resample(np.array([], dtype=np.int16), 22050)) == 0


class TestWhisperNormalisation:
    def test_int16_is_scaled_into_the_expected_range(self) -> None:
        engine = STTEngine()
        audio = np.array([-32768, 0, 16384], dtype=np.int16)

        out = engine._prepare_audio(audio, 16000)

        assert out.dtype == np.float32
        assert out.min() >= -1.0 and out.max() <= 1.0
        assert out[0] == pytest.approx(-1.0)

    def test_float_audio_is_not_scaled_twice(self) -> None:
        engine = STTEngine()
        audio = np.array([-1.0, 0.5, 1.0], dtype=np.float32)

        out = engine._prepare_audio(audio, 16000)

        assert np.allclose(out, audio)

    def test_normalisation_happens_after_resampling(self) -> None:
        engine = STTEngine()
        audio = np.full(PIPER_RATE, 30000, dtype=np.int16)

        out = engine._prepare_audio(audio, PIPER_RATE)

        assert len(out) == 16000
        assert out.max() <= 1.0


class TestPiperDiscovery:
    def test_path_resolution_wins(self, monkeypatch, tmp_path) -> None:
        fake = tmp_path / "piper.exe"
        fake.write_text("stub")
        monkeypatch.setattr(tts_module.shutil, "which", lambda name: str(fake))

        assert TTSEngine()._find_piper() == fake

    def test_discovery_never_executes_the_binary(self, monkeypatch, tmp_path) -> None:
        """Regressions guard: `piper --version` exits 2 on current releases,
        so an exit-code probe rejects a working install."""
        fake = tmp_path / "piper.exe"
        fake.write_text("stub")
        monkeypatch.setattr(tts_module.shutil, "which", lambda name: str(fake))

        def explode(*args, **kwargs):
            raise AssertionError("discovery must not run the binary")

        monkeypatch.setattr(tts_module.subprocess, "run", explode)
        monkeypatch.setattr(tts_module.asyncio, "create_subprocess_exec", explode)

        assert TTSEngine()._find_piper() == fake

    def test_missing_everything_returns_none(self, monkeypatch) -> None:
        monkeypatch.setattr(tts_module.shutil, "which", lambda name: None)
        monkeypatch.setattr(tts_module.Path, "exists",
                            lambda self: "piper" in str(self).lower() and False)

        assert TTSEngine()._find_piper() is None

    def test_the_installed_binary_is_actually_found(self) -> None:
        """End-to-end guard for the machine this project runs on."""
        found = TTSEngine()._find_piper()
        if found is not None:
            assert found.exists()
