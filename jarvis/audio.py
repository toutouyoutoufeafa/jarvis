"""Audio input/output handling with VAD (Voice Activity Detection)."""
from __future__ import annotations

import asyncio
import collections
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import sounddevice as sd
from loguru import logger

from jarvis.config import get_config
from jarvis.events import Event, Events, get_event_bus


@dataclass
class AudioChunk:
    data: np.ndarray  # int16, shape (samples,)
    timestamp: float
    is_speech: bool = False


class VAD:
    """Simple energy-based Voice Activity Detection."""

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000) -> None:
        self.threshold = threshold
        self.sample_rate = sample_rate
        self._energy_history: collections.deque[float] = collections.deque(maxlen=100)

    def process(self, chunk: np.ndarray) -> float:
        """Return normalized energy (0-1)."""
        # Convert to float32, normalize
        audio = chunk.astype(np.float32) / 32768.0
        energy = float(np.sqrt(np.mean(audio**2)))

        # Adaptive threshold based on recent history
        self._energy_history.append(energy)
        if len(self._energy_history) > 10:
            avg_energy = np.mean(self._energy_history)
            # Threshold is relative to average background
            adaptive_thresh = max(self.threshold, avg_energy * 3.0)
        else:
            adaptive_thresh = self.threshold

        return min(1.0, energy / max(adaptive_thresh, 1e-6))

    def is_speech(self, chunk: np.ndarray) -> bool:
        return self.process(chunk) > 1.0


class AudioInput:
    """Microphone input with VAD and speech segmentation."""

    def __init__(self) -> None:
        self.config = get_config()
        self.event_bus = get_event_bus()
        self.vad = VAD(self.config.audio.vad_threshold, self.config.audio.sample_rate)
        self._stream: sd.InputStream | None = None
        self._running = False
        self._audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
        self._silence_chunks = 0
        self._speech_started = False
        self._speech_buffer: list[np.ndarray] = []

    async def start(self) -> None:
        if self._running:
            return

        self._running = True
        logger.info("Starting audio input...")

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.warning(f"Audio input status: {status}")
            # Copy data (indata is read-only)
            chunk = indata[:, 0].copy()  # mono
            asyncio.run_coroutine_threadsafe(
                self._audio_queue.put(AudioChunk(data=chunk, timestamp=asyncio.get_event_loop().time())),
                asyncio.get_event_loop()
            )

        self._stream = sd.InputStream(
            samplerate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            blocksize=self.config.audio.chunk_size,
            dtype=np.int16,
            device=self.config.audio.input_device,
            callback=callback,
        )
        self._stream.start()

        # Start processing loop
        asyncio.create_task(self._process_loop())

        await self.event_bus.emit(Event(Events.SYSTEM_STARTUP, {"component": "audio_input"}))

    async def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio input stopped")

    async def _process_loop(self) -> None:
        silence_limit = int(self.config.audio.silence_duration * self.config.audio.sample_rate / self.config.audio.chunk_size)

        while self._running:
            chunk = await self._audio_queue.get()

            is_speech = self.vad.is_speech(chunk.data)
            chunk.is_speech = is_speech

            if is_speech:
                if not self._speech_started:
                    self._speech_started = True
                    self._speech_buffer = []
                    await self.event_bus.emit(Event(Events.SPEECH_STARTED, {}))
                    logger.debug("Speech started")

                self._speech_buffer.append(chunk.data)
                self._silence_chunks = 0
            else:
                if self._speech_started:
                    self._silence_chunks += 1
                    self._speech_buffer.append(chunk.data)

                    if self._silence_chunks >= silence_limit:
                        # Speech ended
                        await self._finalize_speech()
                        self._speech_started = False
                        self._silence_chunks = 0

    async def _finalize_speech(self) -> None:
        if not self._speech_buffer:
            return

        # Concatenate all speech chunks
        full_audio = np.concatenate(self._speech_buffer)
        self._speech_buffer = []

        await self.event_bus.emit(Event(
            Events.STT_START,
            {"audio_data": full_audio, "sample_rate": self.config.audio.sample_rate}
        ))
        logger.debug(f"Speech ended, sending to STT ({len(full_audio)} samples)")


class AudioOutput:
    """Audio playback for TTS."""

    def __init__(self) -> None:
        self.config = get_config()
        self.event_bus = get_event_bus()
        self._stream: sd.OutputStream | None = None

    async def start(self) -> None:
        if self._stream is not None:
            return

        self._stream = sd.OutputStream(
            samplerate=self.config.audio.sample_rate,
            channels=self.config.audio.channels,
            dtype=np.int16,
            device=self.config.audio.output_device,
        )
        self._stream.start()
        logger.info("Audio output started")

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def play(self, audio_data: np.ndarray) -> None:
        """Play audio data (int16)."""
        if self._stream is None:
            await self.start()

        # Ensure correct format
        if audio_data.dtype != np.int16:
            audio_data = (audio_data * 32767).astype(np.int16)

        # Play in chunks to avoid blocking
        chunk_size = self.config.audio.chunk_size
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) == 0:
                continue
            # Pad if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self._stream.write(chunk)

    async def play_file(self, path: Path) -> None:
        """Play a WAV file."""
        with wave.open(str(path), 'rb') as wf:
            audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            await self.play(audio_data)


# Global instances
_audio_input: AudioInput | None = None
_audio_output: AudioOutput | None = None


def get_audio_input() -> AudioInput:
    global _audio_input
    if _audio_input is None:
        _audio_input = AudioInput()
    return _audio_input


def get_audio_output() -> AudioOutput:
    global _audio_output
    if _audio_output is None:
        _audio_output = AudioOutput()
    return _audio_output