"""Plugin system for extensible capabilities."""
from __future__ import annotations

import importlib
import importlib.util
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from jarvis.config import get_config, get_config_dir
from jarvis.events import EventBus, get_event_bus


class PluginMetadata(BaseModel):
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    dependencies: list[str] = []
    enabled: bool = True


class Plugin(ABC):
    """Base class for all plugins."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self.event_bus = event_bus or get_event_bus()
        self.config = get_config()
        self._metadata = self._get_metadata()
        self._initialized = False

    @abstractmethod
    def _get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @property
    def name(self) -> str:
        return self._metadata.name

    async def initialize(self) -> None:
        """Initialize plugin (called after all plugins loaded)."""
        if self._initialized:
            return
        await self._initialize()
        self._initialized = True
        logger.info(f"Plugin initialized: {self.name}")

    @abstractmethod
    async def _initialize(self) -> None:
        """Plugin-specific initialization."""

    async def shutdown(self) -> None:
        """Cleanup plugin resources."""
        if not self._initialized:
            return
        await self._shutdown()
        self._initialized = False
        logger.info(f"Plugin shutdown: {self.name}")

    async def _shutdown(self) -> None:
        """Plugin-specific shutdown."""
        pass

    def handle_event(self, event_type: str) -> callable:
        """Decorator to register event handlers."""

        def decorator(func: callable) -> callable:
            self.event_bus.subscribe(event_type, func)
            return func

        return decorator


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""

    def __init__(self) -> None:
        self.plugins: dict[str, Plugin] = {}
        self._load_order: list[str] = []

    def discover_plugins(self, directories: list[str] | None = None) -> list[Path]:
        """Find plugin modules in configured directories."""
        config = get_config()
        dirs = directories or config.plugins.directories
        # Relative entries are resolved against the config directory (falling
        # back to the project root), not the CWD, so `jarvis` works anywhere.
        base_dir = get_config_dir()
        plugin_files = []
        seen: set[Path] = set()

        for dir_str in dirs:
            path = Path(dir_str).expanduser()
            if not path.is_absolute():
                path = base_dir / path
            if not path.exists():
                # `~/.jarvis/plugins` is optional user-extended storage
                logger.debug(f"Plugin directory not found: {path}")
                continue

            for py_file in sorted(path.rglob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                if py_file.name == "plugin.py" or py_file.stem.endswith("_plugin"):
                    resolved = py_file.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    plugin_files.append(resolved)

        return plugin_files

    def load_plugin(self, path: Path) -> Plugin | None:
        """Load a single plugin from a Python file."""
        try:
            module_name = f"jarvis_plugin_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                logger.error(f"Could not load spec for {path}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find Plugin subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Plugin)
                    and attr is not Plugin
                ):
                    plugin = attr()
                    if plugin.metadata.enabled:
                        self.plugins[plugin.name] = plugin
                        self._load_order.append(plugin.name)
                        logger.info(f"Loaded plugin: {plugin.name} v{plugin.metadata.version}")
                        return plugin
                    else:
                        logger.info(f"Plugin disabled: {plugin.name}")

            logger.warning(f"No Plugin class found in {path}")
            return None

        except Exception as e:
            logger.error(f"Failed to load plugin {path}: {e}")
            return None

    def load_all(self, directories: list[str] | None = None) -> list[Plugin]:
        """Discover and load all plugins."""
        plugin_files = self.discover_plugins(directories)
        loaded = []

        for path in plugin_files:
            plugin = self.load_plugin(path)
            if plugin:
                loaded.append(plugin)

        return loaded

    async def initialize_all(self) -> None:
        """Initialize all loaded plugins in order."""
        for name in self._load_order:
            plugin = self.plugins.get(name)
            if plugin:
                try:
                    await plugin.initialize()
                except Exception as e:
                    logger.error(f"Failed to initialize plugin {name}: {e}")

    async def shutdown_all(self) -> None:
        """Shutdown all plugins in reverse order."""
        for name in reversed(self._load_order):
            plugin = self.plugins.get(name)
            if plugin:
                try:
                    await plugin.shutdown()
                except Exception as e:
                    logger.error(f"Failed to shutdown plugin {name}: {e}")

    def get_plugin(self, name: str) -> Plugin | None:
        return self.plugins.get(name)

    def list_plugins(self) -> list[PluginMetadata]:
        return [p.metadata for p in self.plugins.values()]


# Global plugin manager
_plugin_manager: PluginManager | None = None


def get_plugin_manager() -> PluginManager:
    global _plugin_manager
    if _plugin_manager is None:
        _plugin_manager = PluginManager()
    return _plugin_manager