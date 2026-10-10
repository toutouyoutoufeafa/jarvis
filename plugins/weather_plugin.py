"""Sample plugin: Weather information."""
from __future__ import annotations

import asyncio
import aiohttp
from loguru import logger

from jarvis.plugins import Plugin, PluginMetadata
from jarvis.events import get_event_bus, Events, Event


class WeatherPlugin(Plugin):
    """Weather information plugin using OpenWeatherMap API."""

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="weather",
            version="0.1.0",
            description="Get current weather and forecasts",
            author="JARVIS Project",
        )

    async def _initialize(self) -> None:
        # Register intent handler
        self.event_bus.subscribe(Events.INTENT_RECOGNIZED, self._on_intent)
        logger.info("Weather plugin initialized")

    async def _on_intent(self, event: Event) -> None:
        intent_name = event.data.get("intent")
        if intent_name not in ("weather_current", "weather_forecast"):
            return

        entities = event.data.get("entities", {})
        location = entities.get("location", "Paris")

        if intent_name == "weather_current":
            weather = await self._get_current_weather(location)
            response = f"Météo à {location}: {weather}"
        else:
            days = int(entities.get("days", "3"))
            forecast = await self._get_forecast(location, days)
            response = f"Prévisions pour {location} ({days} jours):\n" + "\n".join(forecast)

        # Emit response for TTS
        await self.event_bus.emit(Event(Events.TTS_START, {"text": response}))

    async def _get_current_weather(self, location: str) -> str:
        """Fetch current weather."""
        # This would use a real API key
        # For demo, return mock data
        return "20°C, ciel dégagé, humidité 65%"

    async def _get_forecast(self, location: str, days: int) -> list[str]:
        """Fetch forecast."""
        return [
            "Aujourd'hui: 22°C, ensoleillé",
            "Demain: 19°C, quelques nuages",
            "Après-demain: 17°C, pluie légère",
        ][:days]


# Plugin entry point
def create_plugin() -> Plugin:
    return WeatherPlugin()