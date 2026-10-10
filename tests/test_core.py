"""Tests for JARVIS core modules."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from jarvis.config import Config, get_config, set_config
from jarvis.events import Event, Events, EventBus, get_event_bus
from jarvis.plugins import Plugin, PluginManager, PluginMetadata


class TestConfig:
    def test_default_config(self) -> None:
        config = Config()
        assert config.assistant.name == "JARVIS"
        assert config.audio.sample_rate == 16000
        assert config.stt.engine == "faster-whisper"

    def test_load_from_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({
                "assistant": {"name": "TEST"},
                "audio": {"sample_rate": 48000},
            }, f)
            path = Path(f.name)

        try:
            config = Config.from_yaml(path)
            assert config.assistant.name == "TEST"
            assert config.audio.sample_rate == 48000
        finally:
            path.unlink()

    def test_env_expansion(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"test_path": "${TEMP}"}, f)
            path = Path(f.name)

        try:
            config = Config.from_yaml(path)
            assert config.__dict__.get("test_path") is None  # Not in model
        finally:
            path.unlink()


class TestEvents:
    @pytest.mark.asyncio
    async def test_event_bus_subscribe_emit(self) -> None:
        bus = EventBus()
        received = []

        def handler(event: Event) -> None:
            received.append(event)

        bus.subscribe("test.event", handler)
        await bus.emit(Event(type="test.event", data={"value": 42}))

        assert len(received) == 1
        assert received[0].data["value"] == 42

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self) -> None:
        bus = EventBus()
        received = []

        def handler(event: Event) -> None:
            received.append(event.type)

        bus.subscribe_all(handler)
        await bus.emit(Event(type="event.a"))
        await bus.emit(Event(type="event.b"))

        assert "event.a" in received
        assert "event.b" in received

    @pytest.mark.asyncio
    async def test_async_handler(self) -> None:
        bus = EventBus()
        received = []

        async def handler(event: Event) -> None:
            received.append(event.data)

        bus.subscribe("async.test", handler)
        await bus.emit(Event(type="async.test", data={"async": True}))

        assert received[0]["async"] is True


class TestPlugins:
    def test_plugin_metadata(self) -> None:
        meta = PluginMetadata(name="test", version="1.0", description="Test plugin")
        assert meta.name == "test"
        assert meta.enabled is True

    def test_plugin_manager_discovery(self, tmp_path: Path) -> None:
        # Create a fake plugin file
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        plugin_file = plugin_dir / "test_plugin.py"
        plugin_file.write_text("""
from jarvis.plugins import Plugin, PluginMetadata

class TestPlugin(Plugin):
    def _get_metadata(self):
        return PluginMetadata(name="test", version="1.0")

    async def _initialize(self):
        pass
""")

        pm = PluginManager()
        # Override config to use temp dir
        from jarvis.config import get_config
        cfg = get_config()
        cfg.plugins.directories = [str(plugin_dir)]

        found = pm.discover_plugins([str(plugin_dir)])
        assert len(found) == 1
        assert found[0].name == "test_plugin.py"


class TestIntent:
    @pytest.mark.asyncio
    async def test_fallback_recognize(self) -> None:
        from jarvis.intent import IntentRecognizer

        recognizer = IntentRecognizer()
        # Test keyword fallback
        intent = recognizer._fallback_recognize("lance chrome")
        assert intent.name == "launch_app"
        assert intent.entities.get("app") == "chrome"

        intent = recognizer._fallback_recognize("volume monte")
        assert intent.name == "volume_control"
        assert intent.entities.get("action") == "up"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])