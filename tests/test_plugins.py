"""Plugin tests: parsing and PowerShell integration, without touching the network."""
from __future__ import annotations

import asyncio
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from jarvis.events import Event, Events
from jarvis.plugins import PluginManager

PLUGINS_DIR = Path(__file__).resolve().parent.parent / "plugins"
WEATHER_PATH = PLUGINS_DIR / "weather_plugin.py"
SYSTEM_CONTROL_PATH = PLUGINS_DIR / "system_control_plugin.py"


@pytest.fixture(scope="module")
def weather():
    """The real plugin, loaded the way the manager loads it."""
    loaded = PluginManager().load_plugin(WEATHER_PATH)
    assert loaded is not None, f"could not load {WEATHER_PATH}"
    return loaded


class TestDayParsing:
    @pytest.mark.parametrize("raw,expected", [
        (1, 1), (3, 3), (7, 7),
        ("2", 2), ("5", 5),
    ])
    def test_reads_usable_values(self, weather, raw, expected) -> None:
        assert weather._parse_days(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "demain", "beaucoup", 0.4])
    def test_falls_back_on_anything_unusable(self, weather, raw) -> None:
        """A model-generated "days" value must not raise out of the handler."""
        assert weather._parse_days(raw, default=3) == 3

    @pytest.mark.parametrize("raw", [0, -2, 99, "99"])
    def test_clamps_to_the_api_window(self, weather, raw) -> None:
        assert 1 <= weather._parse_days(raw) <= 7


class TestLocationCleanup:
    def test_plain_city_survives(self, weather) -> None:
        location, days = weather._clean("Lyon", 1)
        assert location == "Lyon"
        assert days == 1

    def test_time_words_become_days_not_places(self, weather) -> None:
        """'meteo de demain' used to be geocoded as a place named 'de demain'."""
        location, days = weather._clean("de demain", 1)
        assert days == 2
        assert location == "Paris"  # falls back to the default place

    def test_a_week_asks_for_a_full_forecast(self, weather) -> None:
        _, days = weather._clean("de cette semaine", 1)
        assert days == 7

    def test_today_asks_about_now(self, weather) -> None:
        """However many days the request implied."""
        _, days = weather._clean("aujourd'hui", 3)
        assert days == 1

    def test_leading_articles_are_dropped(self, weather) -> None:
        location, _ = weather._clean("de Nice", 1)
        assert location == "Nice"

    def test_trailing_prepositions_are_dropped(self, weather) -> None:
        location, _ = weather._clean("Nice de", 1)
        assert location == "Nice"

    def test_empty_falls_back_to_a_default_place(self, weather) -> None:
        location, _ = weather._clean("   ", 1)
        assert location == "Paris"

    @pytest.mark.parametrize("text", ["", "   ", "de", "la semaine"])
    def test_never_returns_an_empty_place(self, weather, text) -> None:
        location, _ = weather._clean(text, 1)
        assert location.strip()


class TestWeatherContract:
    def test_metadata_is_declared(self, weather) -> None:
        assert weather.name == "weather"
        assert weather.metadata.version

    @pytest.mark.asyncio
    async def test_unrelated_intents_are_left_alone(self, weather) -> None:
        event = Event(Events.INTENT_RECOGNIZED, {"intent": "launch_app"})
        await weather._on_intent(event)

        assert "answer" not in event.data


class TestSystemControlPowerShell:
    """Encoding, timeouts and JSON tolerance for the PowerShell bridge."""

    @pytest.fixture(scope="class")
    @classmethod
    def module(cls):
        spec = spec_from_file_location("system_control_under_test", SYSTEM_CONTROL_PATH)
        cls.module = module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        return cls.module

    def test_entry_point_returns_a_plugin(self, module) -> None:
        assert module.create_plugin().name == "system_control_extended"

    def test_utf8_preamble_is_added_to_every_command(self, module, monkeypatch) -> None:
        """Without it accents decode wrongly and the JSON parse fails."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(cmd=cmd, kwargs=kwargs)

            class Result:
                returncode = 0
                stdout = b"[]"
                stderr = b""

            return Result()

        async def fake_to_thread(fn, *a, **kw):
            return fn(*a, **kw)

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)

        out = asyncio.run(module.run_powershell("echo hi"))

        assert captured["cmd"][3] == module._UTF8_PREAMBLE + "echo hi"
        assert captured["cmd"][:3] == ["powershell", "-NoProfile", "-Command"]
        assert out == "[]"

    def test_timeout_is_enforced(self, module, monkeypatch) -> None:
        """A hung PowerShell must not hang the chat loop."""
        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)

            class Result:
                returncode = 0
                stdout = b""
                stderr = b""

            return Result()

        async def fake_to_thread(fn, *a, **kw):
            return fn(*a, **kw)

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)

        asyncio.run(module.run_powershell("x", timeout=5.0))

        assert captured["timeout"] == 5.0

    def test_failure_raises_with_the_stderr_text(self, module, monkeypatch) -> None:
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 1
                stdout = b""
                stderr = "commande inconnue".encode("utf-8")

            return Result()

        async def fake_to_thread(fn, *a, **kw):
            return fn(*a, **kw)

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)

        with pytest.raises(RuntimeError, match="commande inconnue"):
            asyncio.run(module.run_powershell("nope"))

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_empty_powershell_output_is_not_json(self, module, text) -> None:
        assert module.parse_json(text) is None

    def test_unparseable_output_is_tolerated(self, module) -> None:
        """A warning must not become a crash in the chat loop."""
        assert module.parse_json("not json at all") is None

    def test_single_object_is_returned_as_is(self, module) -> None:
        assert module.parse_json('{"a": 1}') == {"a": 1}

    def test_parse_json_handles_utf8(self, module) -> None:
        assert module.parse_json('[{"n": "Gestionnaire"}]') == [{"n": "Gestionnaire"}]

    def test_invalid_utf8_is_replaced_not_fatal(self, module, monkeypatch) -> None:
        def fake_run(cmd, **kwargs):
            class Result:
                returncode = 0
                stdout = b'{"t": "Gestionnaire"} '  # lone utf-8 lead byte
                stderr = b""

            return Result()

        async def fake_to_thread(fn, *a, **kw):
            return fn(*a, **kw)

        monkeypatch.setattr(module.subprocess, "run", fake_run)
        monkeypatch.setattr(module.asyncio, "to_thread", fake_to_thread)

        out = asyncio.run(module.run_powershell("x"))
        assert isinstance(out, str)  # no UnicodeDecodeError escapes
