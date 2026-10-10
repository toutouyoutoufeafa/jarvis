"""JARVIS - Personal AI Assistant inspired by Iron Man's JARVIS."""
from __future__ import annotations

from jarvis.config import Config, get_config, set_config
from jarvis.events import Event, Events, EventBus, get_event_bus
from jarvis.plugins import Plugin, PluginManager, PluginMetadata, get_plugin_manager

__version__ = "0.1.0"
__author__ = "JARVIS Project"

__all__ = [
    "Config",
    "get_config",
    "set_config",
    "Event",
    "Events",
    "EventBus",
    "get_event_bus",
    "Plugin",
    "PluginManager",
    "PluginMetadata",
    "get_plugin_manager",
]