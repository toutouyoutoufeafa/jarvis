"""Sample plugin: Extended system control actions."""
from __future__ import annotations

import asyncio
import subprocess
from loguru import logger

from jarvis.plugins import Plugin, PluginMetadata
from jarvis.intent import Action, get_intent_recognizer
from jarvis.events import get_event_bus, Events, Event


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
        # Use PowerShell to get window titles
        cmd = """
        Get-Process | Where-Object {$_.MainWindowTitle -ne ""} |
        Select-Object ProcessName, MainWindowTitle, Id |
        ConvertTo-Json
        """
        result = await asyncio.to_thread(
            subprocess.run, ["powershell", "-Command", cmd],
            capture_output=True, text=True
        )
        if result.stdout:
            import json
            windows = json.loads(result.stdout)
            if isinstance(windows, dict):
                windows = [windows]
            lines = [f"  {w['ProcessName']} (PID:{w['Id']}): {w['MainWindowTitle'][:60]}" for w in windows[:10]]
            return "Fenêtres ouvertes:\n" + "\n".join(lines)
        return "Aucune fenêtre trouvée"

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
        result = await asyncio.to_thread(
            subprocess.run, ["powershell", "-Command", find_cmd],
            capture_output=True, text=True
        )

        if result.returncode != 0 or not result.stdout.strip():
            return f"Fenêtre non trouvée: {target}"

        pid = result.stdout.strip()

        # Execute action
        action_cmd = f"$pid = {pid}; {actions[action]}"
        await asyncio.to_thread(
            subprocess.run, ["powershell", "-Command", action_cmd],
            capture_output=True
        )

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
            cmd = "Get-NetIPConfiguration | Select-Object InterfaceAlias, IPv4Address, DNSServer | ConvertTo-Json"
            result = await asyncio.to_thread(
                subprocess.run, ["powershell", "-Command", cmd],
                capture_output=True, text=True
            )
            if result.stdout:
                import json
                data = json.loads(result.stdout)
                if isinstance(data, dict):
                    data = [data]
                lines = []
                for d in data:
                    ips = d.get("IPv4Address", [])
                    ip_str = ", ".join([ip["IPAddress"] for ip in ips]) if ips else "N/A"
                    lines.append(f"  {d['InterfaceAlias']}: {ip_str}")
                return "Interfaces réseau:\n" + "\n".join(lines)
            return "Aucune interface trouvée"
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