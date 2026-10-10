"""Event bus for inter-component communication."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

from loguru import logger

T = TypeVar("T")


@dataclass
class Event:
    """Base event class."""
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid4()))
    source: str | None = None
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class EventBus:
    """Async event bus with typed handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Callable[[Event], Any]]] = defaultdict(list)
        self._wildcard_handlers: list[Callable[[Event], Any]] = []

    def subscribe(self, event_type: str, handler: Callable[[Event], Any]) -> Callable[[], None]:
        """Subscribe to an event type. Returns unsubscribe function."""
        self._handlers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__qualname__} to {event_type}")

        def unsubscribe() -> None:
            self._handlers[event_type].remove(handler)

        return unsubscribe

    def subscribe_all(self, handler: Callable[[Event], Any]) -> Callable[[], None]:
        """Subscribe to all events."""
        self._wildcard_handlers.append(handler)

        def unsubscribe() -> None:
            self._wildcard_handlers.remove(handler)

        return unsubscribe

    async def emit(self, event: Event) -> None:
        """Emit an event to all subscribers."""
        event.source = event.source or "unknown"
        logger.trace(f"Emitting {event.type} from {event.source}")

        # Call specific handlers
        handlers = self._handlers.get(event.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Handler {handler.__qualname__} failed for {event.type}: {e}")

        # Call wildcard handlers
        for handler in self._wildcard_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Wildcard handler {handler.__qualname__} failed for {event.type}: {e}")

    def emit_sync(self, event: Event) -> None:
        """Emit event synchronously (for non-async contexts)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.emit(event))
        except RuntimeError:
            asyncio.run(self.emit(event))


# Global event bus
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


# Common event types
class Events:
    # Audio
    AUDIO_INPUT = "audio.input"
    AUDIO_OUTPUT = "audio.output"
    WAKE_WORD_DETECTED = "wake_word.detected"
    SPEECH_STARTED = "speech.started"
    SPEECH_ENDED = "speech.ended"

    # STT
    STT_START = "stt.start"
    STT_RESULT = "stt.result"
    STT_ERROR = "stt.error"

    # TTS
    TTS_START = "tts.start"
    TTS_CHUNK = "tts.chunk"
    TTS_END = "tts.end"
    TTS_ERROR = "tts.error"

    # LLM
    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"
    LLM_STREAM_CHUNK = "llm.stream_chunk"
    LLM_ERROR = "llm.error"

    # Intent/Action
    INTENT_RECOGNIZED = "intent.recognized"
    ACTION_EXECUTE = "action.execute"
    ACTION_RESULT = "action.result"
    ACTION_ERROR = "action.error"

    # System
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    ERROR = "system.error"