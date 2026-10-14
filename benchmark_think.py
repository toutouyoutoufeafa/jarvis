"""Voice round trip without a microphone: TTS -> STT.

Catches wiring bugs (sample rates, formats, engine discovery) that a
headless box cannot otherwise reach.
"""
import asyncio
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from jarvis.stt import get_stt_engine
from jarvis.tts import get_tts_engine

SENTENCE = "Bonjour Tony, JARVIS est en ligne et pret a travailler."


async def main() -> None:
    tts = get_tts_engine()
    t0 = time.perf_counter()
    audio = await tts.synthesize(SENTENCE)
    synth_time = time.perf_counter() - t0

    duration = len(audio) / tts.sample_rate
    print(f"TTS  : {synth_time:.2f}s  "
          f"{len(audio)} echantillons @ {tts.sample_rate} Hz "
          f"= {duration:.2f}s de parole")
    print(f"       dtype={audio.dtype} max={int(np.abs(audio).max())}")

    out = Path.home() / ".jarvis" / "workspace" / "tts_roundtrip.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, audio, tts.sample_rate, subtype="PCM_16")
    print(f"       wav ecrit: {out}")

    stt = get_stt_engine()
    t0 = time.perf_counter()
    # The STT engine resamples internally; declaring the real rate is what
    # makes that possible.
    text = await stt.transcribe(audio, sample_rate=tts.sample_rate)
    stt_time = time.perf_counter() - t0

    print(f"STT  : {stt_time:.2f}s -> {text!r}")
    print(f"\nALLER-RETOUR {synth_time + stt_time:.2f}s pour {duration:.2f}s de parole")

    norm = lambda s: "".join(c for c in s.lower() if c.isalnum() or c.isspace()).split()
    expected, got = norm(SENTENCE), norm(text or "")
    overlap = len(set(expected) & set(got)) / max(1, len(set(expected)))
    print(f"fidelite : {overlap * 100:.0f}% des mots retrouves")


if __name__ == "__main__":
    asyncio.run(main())
