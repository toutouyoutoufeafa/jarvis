"""Text-to-Speech using Piper (local) with cloud fallback."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import soundfile as sf
from loguru import logger

from jarvis.config import get_config
from jarvis.events import Event, Events, get_event_bus


class TTSEngine:
    """TTS engine with Piper (local) and cloud fallback."""

    def __init__(self) -> None:
        self.config = get_config()
        self.event_bus = get_event_bus()
        self._piper_path: Optional[Path] = None
        self._voice_path: Optional[Path] = None
        # Actual output rate of the last synthesis. Piper's medium voices
        # render at 22050 Hz, so callers must not assume the mic rate.
        self.sample_rate: int = 16000

    async def initialize(self) -> None:
        if self.config.tts.engine == "piper":
            await self._init_piper()

    def _find_piper(self) -> Optional[Path]:
        """Locate the Piper binary.

        An exit-code probe is useless here: ``piper --version`` exits with
        code 2 on current releases, which made the old check reject a
        perfectly good install. Resolve the executable instead.
        """
        candidates = [
            shutil.which("piper"),
            str(Path.home() / ".local" / "bin" / "piper"),
            str(Path(sys.executable).parent / "Scripts" / "piper.exe"),
            str(Path(sys.executable).parent / "piper"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                return Path(candidate)
        return None

    async def _init_piper(self) -> None:
        """Find or download Piper binary and voice model."""
        self._piper_path = self._find_piper()
        if self._piper_path:
            logger.info(f"Found Piper: {self._piper_path}")
        else:
            logger.warning("Piper not found. Install with: pip install piper-tts")
            logger.warning("Or download from: https://github.com/rhasspy/piper/releases")

        # Voice model path
        voice_name = self.config.tts.voice
        # Piper voices are typically at ~/.local/share/piper/voices/
        voice_dirs = [
            Path.home() / ".local" / "share" / "piper" / "voices",
            Path("/usr/share/piper/voices"),
            Path.cwd() / "voices",
        ]
        for d in voice_dirs:
            for ext in [".onnx", ".onnx.json"]:
                vp = d / f"{voice_name}{ext}"
                if vp.exists():
                    self._voice_path = vp
                    break
            if self._voice_path:
                break

        if self._voice_path:
            logger.info(f"Using Piper voice: {self._voice_path}")
        else:
            logger.warning(f"Voice not found: {voice_name}. Download from https://huggingface.co/rhasspy/piper-voices")

    async def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text to audio (int16 array).

        Self-initializing, like ``STTEngine.transcribe``: requiring an
        explicit ``initialize()`` first only served to make the obvious
        call raise. The real output rate is recorded on ``sample_rate``
        because Piper voices are usually 22050 Hz, not the 16000 Hz that
        speech-to-text assumes.
        """
        if self.config.tts.engine == "piper" and self._piper_path is None:
            await self.initialize()

        engine = self.config.tts.engine

        if engine == "piper":
            return await self._synthesize_piper(text)
        elif engine == "openai":
            return await self._synthesize_openai(text)
        elif engine == "elevenlabs":
            return await self._synthesize_elevenlabs(text)
        else:
            raise ValueError(f"Unknown TTS engine: {engine}")

    async def _synthesize_piper(self, text: str) -> np.ndarray:
        if self._piper_path is None or self._voice_path is None:
            await self.initialize()
        if self._piper_path is None or self._voice_path is None:
            raise RuntimeError(
                "Piper or its voice is missing. Install piper-tts and download "
                "a voice into ~/.local/share/piper/voices/."
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Build command
            cmd = [
                str(self._piper_path),
                "--model", str(self._voice_path),
                "--output_file", str(tmp_path),
                "--length_scale", str(1.0 / self.config.tts.speed),
            ]

            # Run piper
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate(input=text.encode())

            if proc.returncode != 0:
                raise RuntimeError(f"Piper failed: {stderr.decode()}")

            # Load generated audio, keeping the rate it was rendered at
            audio_data, rate = sf.read(tmp_path, dtype='int16')
            self.sample_rate = int(rate)
            return audio_data

        finally:
            tmp_path.unlink(missing_ok=True)

    async def _synthesize_openai(self, text: str) -> np.ndarray:
        """OpenAI TTS API."""
        provider = self.config.llm.providers.get("openai")
        if not provider or not provider.api_key:
            raise RuntimeError("OpenAI API key not configured")

        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "model": "tts-1",
            "voice": self.config.tts.openai_voice,
            "input": text,
            "speed": self.config.tts.speed,
        }

        response = await asyncio.to_thread(requests.post, url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        # Save to temp and load
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = Path(tmp.name)

        try:
            audio_data, _ = sf.read(tmp_path, dtype='int16')
            return audio_data
        finally:
            tmp_path.unlink(missing_ok=True)

    async def _synthesize_elevenlabs(self, text: str) -> np.ndarray:
        """ElevenLabs TTS API."""
        voice_id = self.config.tts.elevenlabs_voice_id
        if not voice_id:
            raise RuntimeError("ElevenLabs voice ID not configured")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": self.config.llm.providers.get("elevenlabs", {}).api_key or "",
            "Content-Type": "application/json",
        }
        data = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        response = await asyncio.to_thread(requests.post, url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = Path(tmp.name)

        try:
            audio_data, _ = sf.read(tmp_path, dtype='int16')
            return audio_data
        finally:
            tmp_path.unlink(missing_ok=True)

    async def process_event(self, event: Event) -> None:
        """Handle TTS_START event."""
        text = event.data.get("text", "")
        if not text:
            return

        await self.event_bus.emit(Event(Events.TTS_START, {"text": text}))

        try:
            audio = await self.synthesize(text)
            await self.event_bus.emit(Event(Events.TTS_CHUNK, {"audio": audio}))
            await self.event_bus.emit(Event(Events.TTS_END, {"text": text}))
        except Exception as e:
            logger.error(f"TTS error: {e}")
            await self.event_bus.emit(Event(Events.TTS_ERROR, {"error": str(e), "text": text}))


# Global instance
_tts_engine: TTSEngine | None = None


def get_tts_engine() -> TTSEngine:
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = TTSEngine()
    return _tts_engine