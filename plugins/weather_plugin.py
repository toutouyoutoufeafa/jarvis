"""Weather plugin: real data from Open-Meteo, no API key required.

The previous version could never fire -- nothing in the system produced the
``weather_current`` / ``weather_forecast`` intents it listened for -- and it
returned mock data. It now answers the intents the recognizer emits and
hands the result back through ``event.data["answer"]``.
"""
from __future__ import annotations

import asyncio
import re

import aiohttp
from loguru import logger

from jarvis.events import Event, Events, get_event_bus
from jarvis.plugins import Plugin, PluginMetadata

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = 8.0

# WMO weather interpretation codes, as used by Open-Meteo.
WEATHER_CODES = {
    0: "ciel degage",
    1: "peu nuageux",
    2: "partiellement nuageux",
    3: "couvert",
    45: "brouillard",
    48: "brouillard givrant",
    51: "bruine legere",
    53: "bruine",
    55: "bruine dense",
    61: "pluie legere",
    63: "pluie",
    65: "pluie forte",
    71: "neige legere",
    73: "neige",
    75: "neige forte",
    80: "averses legere",
    81: "averses",
    82: "averses fortes",
    95: "orage",
    96: "orage et grele",
}

DEFAULT_LOCATION = "Paris"

# Time words a user naturally adds, and how many days of forecast they imply.
# Without this "meteo de demain" tries to geocode "de demain", fails, and the
# LLM makes up a number.
TIME_WORDS = {
    "aujourd'hui": 1,
    "aujourd’hui": 1,
    "ce soir": 1,
    "cette nuit": 1,
    "demain": 2,
    "apres-demain": 3,
    "après-demain": 3,
    "cette semaine": 7,
    "la semaine": 7,
}

# Articles and prepositions that end up attached to a city name.
_NOISE = re.compile(
    r"^(?:de|du|des|le|la|les|ce|cette|ces|pour|a|à|sur|en|il|elle|y|il fait)\s+"
    r"|\s+(?:de|du|des)$"
)


class WeatherPlugin(Plugin):
    """Current conditions and forecasts via the Open-Meteo API."""

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="weather",
            version="0.2.0",
            description="Real weather for any city (Open-Meteo, keyless)",
            author="JARVIS Project",
        )

    async def _initialize(self) -> None:
        self.event_bus.subscribe(Events.INTENT_RECOGNIZED, self._on_intent)
        logger.info("Weather plugin initialized")

    async def _on_intent(self, event: Event) -> None:
        """Answer a weather intent in place, for the caller to relay."""
        intent_name = event.data.get("intent")
        if intent_name not in ("weather_current", "weather_forecast"):
            return

        location = (event.data.get("entities", {}) or {}).get("location") \
            or DEFAULT_LOCATION
        entities = event.data.get("entities", {}) or {}
        # A bare "meteo" asks about now; an explicit "previsions" wants a
        # few days. Defaulting both to anything above 1 turned every
        # "meteo a Lyon" into a 3-day forecast.
        default_days = 3 if intent_name == "weather_forecast" else 1
        days = self._parse_days(entities.get("days"), default=default_days)
        location, days = self._clean(location, days)

        try:
            # "meteo de demain" asks for a forecast whatever intent matched.
            if days > 1 or intent_name == "weather_forecast":
                answer = await self.forecast(location, days)
            else:
                answer = await self.current_weather(location)
        except Exception as exc:  # noqa: BLE001 - degrade to the LLM
            logger.warning(f"Weather lookup failed for {location!r}: {exc}")
            return

        # The recognizer reads this back once the bus has settled.
        event.data["answer"] = answer

    @staticmethod
    def _parse_days(raw: object, default: int = 1) -> int:
        """Read a requested day count, ignoring anything unparseable."""
        if isinstance(raw, int):
            return max(1, min(raw, 7))
        try:
            return max(1, min(int(str(raw)), 7))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clean(location: str, days: int) -> tuple[str, int]:
        """Strip time words and stray articles out of the captured place.

        "meteo de demain" arrives as location="de demain"; geocoding that
        fails, and the LLM then invents a forecast. Recognising the time
        word instead turns it into a real request.
        """
        text = (location or "").strip()
        lowered = text.lower()

        for word, implied in TIME_WORDS.items():
            if word in lowered:
                days = max(days, implied)
                text = re.sub(re.escape(word), " ", text, flags=re.IGNORECASE)

        # "aujourd'hui" / "ce soir" ask about the present, however many days
        # were otherwise implied by the request.
        if any(w in lowered for w in ("aujourd", "ce soir", "cette nuit")):
            days = 1

        text = _NOISE.sub(" ", text).strip(" ,.-")
        return text or DEFAULT_LOCATION, max(1, min(days, 7))

    async def current_weather(self, location: str) -> str:
        """One-line current conditions for ``location``."""
        coords = await self._geocode(location)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                FORECAST_URL,
                params={
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "current": "temperature_2m,apparent_temperature,"
                               "relative_humidity_2m,weather_code,wind_speed_10m",
                    "timezone": "auto",
                },
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT),
            ) as response:
                response.raise_for_status()
                data = await response.json()

        current = data["current"]
        label = WEATHER_CODES.get(int(current["weather_code"]), "conditions variables")
        return (
            f"Meteo a {coords[2]}: {current['temperature_2m']:.0f}C, {label}. "
            f"Ressenti {current['apparent_temperature']:.0f}C, "
            f"humidite {current['relative_humidity_2m']:.0f}%, "
            f"vent {current['wind_speed_10m']:.0f} km/h."
        )

    async def forecast(self, location: str, days: int = 3) -> str:
        """Daily high/low and conditions for the next few days."""
        days = max(1, min(days, 7))
        coords = await self._geocode(location)
        async with aiohttp.ClientSession() as session:
            async with session.get(
                FORECAST_URL,
                params={
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                    "timezone": "auto",
                    "forecast_days": days,
                },
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT),
            ) as response:
                response.raise_for_status()
                data = await response.json()

        daily = data["daily"]
        lines = [f"Previsions pour {coords[2]}:"]
        for i, date in enumerate(daily["time"]):
            code = int(daily["weather_code"][i])
            label = WEATHER_CODES.get(code, "conditions variables")
            lines.append(
                f"- {date}: {daily['temperature_2m_min'][i]:.0f}C a "
                f"{daily['temperature_2m_max'][i]:.0f}C, {label}"
            )
        return "\n".join(lines)

    async def _geocode(self, location: str) -> tuple[float, float, str]:
        """Resolve a place name to coordinates, or raise if unknown."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                GEOCODING_URL,
                params={"name": location, "count": 1, "language": "fr",
                        "format": "json"},
                timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT),
            ) as response:
                response.raise_for_status()
                data = await response.json()

        results = data.get("results") or []
        if not results:
            raise ValueError(f"inconnue: {location}")

        place = results[0]
        name = place.get("name", location)
        country = place.get("country_code", "")
        return float(place["latitude"]), float(place["longitude"]), f"{name} ({country})"


# Plugin entry point
def create_plugin() -> Plugin:
    return WeatherPlugin()
