"""Sample plugin: Extended system control actions."""
from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

from loguru import logger

from jarvis.plugins import Plugin, PluginMetadata
from jarvis.intent import Action, get_intent_recognizer
from jarvis.events import get_event_bus, Events, Event

# PowerShell must be told to emit UTF-8, and Python must capture raw bytes
# and decode that way: with text=True it decodes with the locale codepage
# (cp1252 here), which mangles any accented window title and makes the
# JSON parse fail with a confusing error.
_UTF8_PREAMBLE = (
    "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
    "$OutputEncoding = [Text.Encoding]::UTF8; "
)


async def run_powershell(command: str, timeout: float = 30.0) -> str:
    """Run a PowerShell snippet and return its UTF-8 decoded stdout."""
    result = await asyncio.to_thread(
        subprocess.run,
        ["powershell", "-NoProfile", "-Command", _UTF8_PREAMBLE + command],
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace").strip()
                           or f"powershell exited with {result.returncode}")
    return result.stdout.decode("utf-8", errors="replace")


def parse_json(text: str) -> Any:
    """Parse PowerShell JSON, tolerating empty output."""
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Unparseable PowerShell JSON: {text[:200]!r}")
        return None


class WindowManagerAction(Action):
    """Manage windows (minimize, maximize, close, list)."""

    @property
    def name(self) -> str:
        return "window_manager"

    @property
    def description(self) -> str:
        return "Manage windows: list, minimize, maximize, close, focus"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        action = entities.get("action", "").lower()
        target = entities.get("target", "")

        try:
            if action == "list":
                return await self._list_windows()
            elif action in ("minimize", "maximize", "close", "focus"):
                if not target:
                    return "Fenêtre cible manquante"
                return await self._window_action(action, target)
            else:
                return f"Action inconnue: {action}"
        except Exception as e:
            logger.error(f"Window manager error: {e}")
            return f"Erreur: {e}"

    async def _list_windows(self) -> str:
        """List every process that owns a visible window."""
        cmd = """
        Get-Process | Where-Object {$_.MainWindowTitle -ne ""} |
        Select-Object ProcessName, MainWindowTitle, Id |
        ConvertTo-Json
        """
        windows = parse_json(await run_powershell(cmd))
        if not windows:
            return "Aucune fenêtre trouvée"

        if isinstance(windows, dict):
            windows = [windows]

        lines = [
            f"  {w['ProcessName']} (PID:{w['Id']}): {str(w['MainWindowTitle'])[:60]}"
            for w in windows[:10]
        ]
        return "Fenêtres ouvertes:\n" + "\n".join(lines)

    async def _window_action(self, action: str, target: str) -> str:
        # Map actions to PowerShell commands
        actions = {
            "minimize": "(Get-Process -Id $pid).MainWindowHandle | ForEach-Object { [user32]::ShowWindow($_, 6) }",
            "maximize": "(Get-Process -Id $pid).MainWindowHandle | ForEach-Object { [user32]::ShowWindow($_, 3) }",
            "close": "Stop-Process -Id $pid -Force",
            "focus": "(Get-Process -Id $pid).MainWindowHandle | ForEach-Object { [user32]::SetForegroundWindow($_) }",
        }

        # Find PID by window title or process name
        find_cmd = f"""
        $proc = Get-Process | Where-Object {{
            $_.MainWindowTitle -like "*{target}*" -or $_.ProcessName -like "*{target}*"
        }} | Select-Object -First 1
        if ($proc) {{ $proc.Id }} else {{ exit 1 }}
        """
        try:
            pid = (await run_powershell(find_cmd, timeout=15.0)).strip()
        except RuntimeError:
            return f"Fenêtre non trouvée: {target}"

        if not pid:
            return f"Fenêtre non trouvée: {target}"

        # Execute action
        action_cmd = f"$pid = {pid}; {actions[action]}"
        await run_powershell(action_cmd, timeout=15.0)

        return f"Fenêtre {target}: {action} effectuée"


class NetworkInfoAction(Action):
    """Network information."""

    @property
    def name(self) -> str:
        return "network_info"

    @property
    def description(self) -> str:
        return "Show network adapters, IP, connections"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        try:
            # -ExpandProperty style shaping matters here: ConvertTo-Json
            # serialises a CIM object into an unreadable string, so the
            # real values are pulled out in PowerShell first.
            cmd = """
            Get-NetIPConfiguration | ForEach-Object {
                [pscustomobject]@{
                    InterfaceAlias = $_.InterfaceAlias
                    IPs = (@($_.IPv4Address) | ForEach-Object { $_.IPAddress |
                         Where-Object { $_ -match '^\\d+\\.' } }) -join ', '
                }
            } | ConvertTo-Json
            """
            data = parse_json(await run_powershell(cmd, timeout=20.0))
            if not data:
                return "Aucune interface trouvée"

            if isinstance(data, dict):
                data = [data]

            lines = [
                f"  {d['InterfaceAlias']}: {d.get('IPs') or 'N/A'}"
                for d in data if d.get("InterfaceAlias")
            ]
            return "Interfaces réseau:\n" + "\n".join(lines) if lines \
                else "Aucune interface trouvée"
        except Exception as e:
            return f"Erreur: {e}"


class SystemControlPlugin(Plugin):
    """Extended system control plugin."""

    def _get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="system_control_extended",
            version="0.1.0",
            description="Extended system control: windows, network, processes",
            author="JARVIS Project",
        )

    async def _initialize(self) -> None:
        # Register custom actions
        recognizer = get_intent_recognizer()
        recognizer.registry.register(WindowManagerAction())
        recognizer.registry.register(NetworkInfoAction())

        # Subscribe to intent recognition to handle new intents
        self.event_bus.subscribe(Events.INTENT_RECOGNIZED, self._on_intent)

        logger.info("System control extended plugin initialized")

    async def _on_intent(self, event: Event) -> None:
        intent_name = event.data.get("intent")
        if intent_name in ("window_manager", "network_info"):
            # The action executor will handle it automatically
            pass


def create_plugin() -> Plugin:
    return SystemControlPlugin()