"""Main JARVIS application orchestrating all components."""
from __future__ import annotations

import asyncio
import re
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from loguru import logger

from jarvis.config import get_config, set_config
from jarvis.events import Event, Events, get_event_bus
from jarvis.audio import get_audio_input, get_audio_output
from jarvis.stt import get_stt_engine
from jarvis.tts import get_tts_engine
from jarvis.llm import get_llm_engine
from jarvis.intent import get_action_executor
from jarvis.plugins import get_plugin_manager
from jarvis.agent import get_agent_core, get_designer_agent, get_monitor_agent, get_reflector_agent, get_spawner, get_memory_manager


class JarvisApp:
    """Main JARVIS application."""

    def __init__(self) -> None:
        self.config = get_config()
        self.event_bus = get_event_bus()
        self._running = False
        self._tasks: list[asyncio.Task] = []

        # Core components
        self.audio_input = get_audio_input()
        self.audio_output = get_audio_output()
        self.stt = get_stt_engine()
        self.tts = get_tts_engine()
        self.llm = get_llm_engine()
        self.executor = get_action_executor()
        self.plugin_manager = get_plugin_manager()

        # Agent system
        self.agent_core = get_agent_core()
        self.designer = get_designer_agent()
        self.monitor = get_monitor_agent()
        self.reflector = get_reflector_agent()
        self.spawner = get_spawner()
        self.memory = get_memory_manager()

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        logger.remove()
        logger.add(
            sys.stderr,
            level=self.config.logging.level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        )
        log_path = Path(self.config.logging.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_path,
            level=self.config.logging.level,
            rotation=self.config.logging.rotation,
            retention=self.config.logging.retention,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        )

    async def initialize(self) -> None:
        """Initialize all components."""
        logger.info("Initializing JARVIS...")

        # Initialize LLM first (needed for intent recognition)
        await self.llm.initialize()

        # Initialize STT/TTS
        await self.stt.initialize()
        await self.tts.initialize()

        # Initialize agent system
        await self.agent_core.initialize()
        await self.designer.initialize()
        await self.reflector.initialize()
        await self.memory.initialize()

        # Start monitor agent
        await self.monitor.initialize()
        await self.monitor.start_monitoring()

        # Initialize spawner
        await self.spawner.initialize()

        # Load plugins
        if self.config.plugins.enabled:
            self.plugin_manager.load_all()
            await self.plugin_manager.initialize_all()

        # Subscribe to events
        self._subscribe_events()

        logger.info("JARVIS initialized successfully")

    def _subscribe_events(self) -> None:
        """Subscribe to internal events."""
        # STT -> Intent / Agent
        self.event_bus.subscribe(Events.STT_RESULT, self._on_stt_result)

        # LLM -> TTS
        self.event_bus.subscribe(Events.LLM_RESPONSE, self._on_llm_response)
        self.event_bus.subscribe(Events.LLM_STREAM_CHUNK, self._on_llm_stream_chunk)

        # TTS -> Audio Output
        self.event_bus.subscribe(Events.TTS_CHUNK, self._on_tts_chunk)

        # Agent events
        self.event_bus.subscribe("agent.task_update", self._on_agent_update)
        self.event_bus.subscribe("monitor.alert", self._on_monitor_alert)
        self.event_bus.subscribe("spawner.agent_done", self._on_subagent_done)

        # Errors
        self.event_bus.subscribe(Events.STT_ERROR, self._on_error)
        self.event_bus.subscribe(Events.TTS_ERROR, self._on_error)
        self.event_bus.subscribe(Events.LLM_ERROR, self._on_error)
        self.event_bus.subscribe(Events.ACTION_ERROR, self._on_error)

    async def _on_stt_result(self, event: Event) -> None:
        """Handle STT result -> process through intent/executor or agent system."""
        text = event.data.get("text", "").strip()
        if not text:
            return

        logger.info(f"User: {text}")

        # Check for wake word (simple version)
        if self.config.assistant.wake_word and self.config.assistant.wake_word.lower() not in text.lower():
            # In a real implementation, wake word detection would be separate
            # For now, process everything
            pass

        # Check if this is a complex task requiring agent system
        if await self._should_use_agent(text):
            logger.info(f"Delegating to agent system: {text}")
            task = await self.agent_core.run_task(text)
            if task.result:
                await self.event_bus.emit(Event(Events.TTS_START, {"text": task.result}))
            elif task.error:
                await self.event_bus.emit(Event(Events.TTS_START, {
                    "text": f"La tâche a échoué: {task.error}"
                }))
        else:
            # Simple command - use existing executor
            response = await self.executor.process_text(text)

            if response:
                await self.event_bus.emit(Event(Events.TTS_START, {"text": response}))

    async def _on_llm_response(self, event: Event) -> None:
        """Handle complete LLM response."""
        text = event.data.get("text", "")
        if text:
            await self.event_bus.emit(Event(Events.TTS_START, {"text": text}))

    async def _on_llm_stream_chunk(self, event: Event) -> None:
        """Handle streaming LLM chunk (for real-time TTS)."""
        # For now, we wait for full response
        # Could implement streaming TTS here
        pass

    async def _on_tts_chunk(self, event: Event) -> None:
        """Play TTS audio chunk."""
        audio = event.data.get("audio")
        if audio is not None:
            await self.audio_output.play(audio)

    async def _on_error(self, event: Event) -> None:
        """Handle errors."""
        error = event.data.get("error", "Unknown error")
        logger.error(f"[{event.type}] {error}")

        # Speak error if it's user-facing
        if event.type in (Events.STT_ERROR, Events.TTS_ERROR, Events.LLM_ERROR):
            await self.event_bus.emit(Event(Events.TTS_START, {
                "text": "Désolé, une erreur est survenue."
            }))

    async def _should_use_agent(self, text: str) -> bool:
        """Determine if input requires agent system vs simple executor."""
        agent_keywords = [
            "conçois", "crée", "développe", "programme", "code", "architecture",
            "surveille", "monitor", "analyse", "audite", "inspecte",
            "réfléchis", "analyse", "critique", "améliore", "optimise",
            "agent", "sous-agent", "délègue", "pipeline", "workflow",
            "refactor", "scaffold", "documentation", "review",
            "post-mortem", "apprends", "mémorise",
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in agent_keywords)

    async def _on_agent_update(self, event: Event) -> None:
        """Handle agent task updates."""
        data = event.data
        task_id = data.get("task_id")
        state = data.get("state")

        # Could emit TTS updates for long-running tasks
        if state == "executing" and data.get("iterations", 0) % 5 == 0:
            await self.event_bus.emit(Event(Events.TTS_START, {
                "text": f"Tâche {task_id} en cours, itération {data['iterations']}"
            }))

    async def _on_monitor_alert(self, event: Event) -> None:
        """Handle monitor alerts."""
        alert = event.data.get("alert", {})
        level = alert.get("level", "info")
        message = alert.get("message", "")

        if level == "critical":
            await self.event_bus.emit(Event(Events.TTS_START, {
                "text": f"Alerte critique: {message}"
            }))
        elif level == "warning":
            logger.warning(f"Monitor alert: {message}")

    async def _on_subagent_done(self, event: Event) -> None:
        """Handle sub-agent completion."""
        agent_id = event.data.get("agent_id")
        state = event.data.get("state")
        result = event.data.get("result")

        if state == "done" and result:
            await self.event_bus.emit(Event(Events.TTS_START, {
                "text": f"Sous-agent {agent_id} terminé: {str(result)[:200]}"
            }))
        elif state == "error":
            error = event.data.get("error", "Unknown error")
            await self.event_bus.emit(Event(Events.TTS_START, {
                "text": f"Sous-agent {agent_id} a échoué: {error}"
            }))

    async def start(self) -> None:
        """Start the main loop."""
        if self._running:
            return

        self._running = True
        logger.info("Starting JARVIS...")

        # Start audio
        await self.audio_input.start()
        await self.audio_output.start()

        # Emit startup
        await self.event_bus.emit(Event(Events.SYSTEM_STARTUP, {}))

        # Speak greeting
        await self.event_bus.emit(Event(Events.TTS_START, {
            "text": f"Bonjour. {self.config.assistant.name} est prêt."
        }))

        # Keep running
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop the application."""
        if not self._running:
            return

        logger.info("Stopping JARVIS...")
        self._running = False

        # Speak goodbye
        await self.event_bus.emit(Event(Events.TTS_START, {
            "text": "Au revoir."
        }))
        await asyncio.sleep(1)

        # Stop components
        await self.audio_input.stop()
        await self.audio_output.stop()
        await self.llm.shutdown()

        # Stop agent system
        await self.monitor.stop_monitoring()
        await self.spawner.terminate_all()
        await self.agent_core.stop()

        # Shutdown plugins
        await self.plugin_manager.shutdown_all()

        await self.event_bus.emit(Event(Events.SYSTEM_SHUTDOWN, {}))
        logger.info("JARVIS stopped")


@asynccontextmanager
async def jarvis_app(config_path: str | Path | None = None):
    """Context manager for JARVIS app."""
    if config_path:
        from jarvis.config import Config
        set_config(Config.from_yaml(config_path))

    app = JarvisApp()
    await app.initialize()
    try:
        yield app
    finally:
        await app.stop()


async def main() -> None:
    """Main entry point."""
    app = JarvisApp()
    await app.initialize()

    # Handle signals
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(app.stop()))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    await app.start()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass