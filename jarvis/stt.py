"""Speech-to-Text using faster-whisper."""
from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from loguru import logger

from jarvis.config import get_config
from jarvis.events import Event, Events, get_event_bus


class STTEngine:
    """Wrapper for faster-whisper."""

    def __init__(self) -> None:
        self.config = get_config()
        self.event_bus = get_event_bus()
        self._model: WhisperModel | None = None
        self._loading = False

    async def initialize(self) -> None:
        if self._model is not None or self._loading:
            return

        self._loading = True
        logger.info(f"Loading STT model: {self.config.stt.model} on {self.config.stt.device}")

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None,
            lambda: WhisperModel(
                self.config.stt.model,
                device=self.config.stt.device,
                compute_type=self.config.stt.compute_type,
            )
        )
        self._loading = False
        logger.info("STT model loaded")

    def _resample(self, audio: np.ndarray, src_rate: int, dst_rate: int = 16000) -> np.ndarray:
        """Linear resample to the rate Whisper expects.

        Piper's voices render at 22050 Hz. Handing that audio to Whisper as
        if it were 16000 Hz plays it ~28% too fast, which is exactly the
        kind of garbling that looks like a model problem but is not.
        """
        if src_rate == dst_rate or src_rate <= 0:
            return audio
        target_len = int(round(len(audio) * dst_rate / src_rate))
        src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype(audio.dtype, copy=False)

    def _prepare_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Resample and normalise to the float32 waveform Whisper wants."""
        audio = self._resample(audio_data, sample_rate).astype(np.float32)
        if audio.size and np.abs(audio).max() > 1.0:
            audio = audio / 32768.0  # int16 -> [-1, 1]
        return audio

    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text."""
        if self._model is None:
            await self.initialize()

        # Pass the waveform straight to the model. Writing a WAV and handing
        # over the path routes through PyAV's decode_audio, which is broken
        # against the installed av build ("metadata_errors" was rejected),
        # and costs a disk round-trip for nothing.
        audio = self._prepare_audio(audio_data, sample_rate)

        loop = asyncio.get_event_loop()
        segments, info = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(
                audio,
                language=self.config.stt.language,
                beam_size=self.config.stt.beam_size,
                # Without this Whisper happily renders "JARVIS" as "Jarville".
                hotwords=self.config.stt.hotwords or None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
        )

        text = " ".join(seg.text for seg in segments).strip()
        logger.debug(f"STT result: '{text}' (lang: {info.language}, prob: {info.language_probability:.2f})")
        return text

    async def process_event(self, event: Event) -> None:
        """Handle STT_START event."""
        audio_data = event.data.get("audio_data")
        sample_rate = event.data.get("sample_rate", 16000)

        if audio_data is None:
            await self.event_bus.emit(Event(Events.STT_ERROR, {"error": "No audio data"}))
            return

        try:
            text = await self.transcribe(audio_data, sample_rate)

            if text:
                await self.event_bus.emit(Event(Events.STT_RESULT, {"text": text}))
            else:
                logger.debug("STT returned empty text")

        except Exception as e:
            logger.error(f"STT error: {e}")
            await self.event_bus.emit(Event(Events.STT_ERROR, {"error": str(e)}))


# Global instance
_stt_engine: STTEngine | None = None


def get_stt_engine() -> STTEngine:
    global _stt_engine
    if _stt_engine is None:
        _stt_engine = STTEngine()
    return _stt_engine