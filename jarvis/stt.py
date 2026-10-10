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

    async def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe audio to text."""
        if self._model is None:
            await self.initialize()

        # Save to temporary WAV file (faster-whisper works best with files)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            # Ensure correct format: mono, 16kHz, int16
            if sample_rate != 16000:
                # Resample would go here, for now assume 16kHz
                pass
            sf.write(tmp_path, audio_data, sample_rate, subtype='PCM_16')

        try:
            loop = asyncio.get_event_loop()
            segments, info = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(
                    str(tmp_path),
                    language=self.config.stt.language,
                    beam_size=self.config.stt.beam_size,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
            )

            text = " ".join(seg.text for seg in segments).strip()
            logger.debug(f"STT result: '{text}' (lang: {info.language}, prob: {info.language_probability:.2f})")
            return text

        finally:
            tmp_path.unlink(missing_ok=True)

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