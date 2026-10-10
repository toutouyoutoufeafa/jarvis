"""Intent recognition and action execution."""
from __future__ import annotations

import asyncio
import re
import shlex
import subprocess
import webbrowser
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil
import pyautogui
from loguru import logger
from pydantic import BaseModel

from jarvis.config import get_config
from jarvis.events import Event, Events, get_event_bus
from jarvis.llm import get_llm_engine, Message


@dataclass
class Intent:
    name: str
    confidence: float
    entities: dict[str, str]
    raw_text: str


class Action(ABC):
    """Base class for executable actions."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        """Execute action. Return result message."""
        pass

    def can_handle(self, intent_name: str) -> bool:
        return intent_name == self.name


class ActionRegistry:
    """Registry of available actions."""

    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        self._actions[action.name] = action
        logger.debug(f"Registered action: {action.name}")

    def get(self, name: str) -> Optional[Action]:
        return self._actions.get(name)

    def list_actions(self) -> list[Action]:
        return list(self._actions.values())

    def get_descriptions(self) -> str:
        return "\n".join(f"- {a.name}: {a.description}" for a in self._actions.values())


# Built-in actions
class LaunchAppAction(Action):
    @property
    def name(self) -> str:
        return "launch_app"

    @property
    def description(self) -> str:
        return "Launch an application (ex: chrome, code, notepad)"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        app = entities.get("app", "").lower()
        config = get_config()

        if app not in config.system_control.allowed_apps:
            return f"Application non autorisée: {app}"

        try:
            # Windows: use start command
            subprocess.Popen(["start", "", app], shell=True)
            return f"Lancement de {app}..."
        except Exception as e:
            logger.error(f"Failed to launch {app}: {e}")
            return f"Erreur lors du lancement de {app}: {e}"


class OpenWebsiteAction(Action):
    @property
    def name(self) -> str:
        return "open_website"

    @property
    def description(self) -> str:
        return "Open a website in default browser"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        url = entities.get("url", "")
        if not url:
            return "URL manquante"

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        try:
            webbrowser.open(url)
            return f"Ouverture de {url}"
        except Exception as e:
            return f"Erreur: {e}"


class SystemCommandAction(Action):
    @property
    def name(self) -> str:
        return "system_command"

    @property
    def description(self) -> str:
        return "Execute a system command (PowerShell/cmd)"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        cmd = entities.get("command", "")
        if not cmd:
            return "Commande manquante"

        try:
            # Run in PowerShell for Windows
            result = await asyncio.to_thread(
                subprocess.run,
                ["powershell", "-Command", cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout.strip()
            if result.stderr:
                output += f"\nErreurs: {result.stderr.strip()}"
            return output or "Commande exécutée (pas de sortie)"
        except subprocess.TimeoutExpired:
            return "Timeout: la commande a pris trop de temps"
        except Exception as e:
            return f"Erreur: {e}"


class FileOperationAction(Action):
    @property
    def name(self) -> str:
        return "file_operation"

    @property
    def description(self) -> str:
        return "File operations: list, read, create, delete"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        operation = entities.get("operation", "").lower()
        path_str = entities.get("path", "")
        content = entities.get("content", "")

        if not path_str:
            return "Chemin manquant"

        config = get_config()
        path = Path(path_str).expanduser()

        # Security check
        allowed = False
        for allowed_path in config.system_control.allowed_paths:
            try:
                path.resolve().relative_to(Path(allowed_path).expanduser().resolve())
                allowed = True
                break
            except ValueError:
                pass

        if not allowed:
            return f"Chemin non autorisé: {path}"

        try:
            if operation == "list":
                if path.is_dir():
                    items = [f"{'📁' if p.is_dir() else '📄'} {p.name}" for p in path.iterdir()]
                    return "\n".join(items) if items else "Dossier vide"
                return "Pas un dossier"

            elif operation == "read":
                if path.is_file():
                    return path.read_text(encoding="utf-8")
                return "Fichier introuvable"

            elif operation == "create":
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                return f"Fichier créé: {path}"

            elif operation == "delete":
                if path.exists():
                    path.unlink()
                    return f"Supprimé: {path}"
                return "Fichier introuvable"

            else:
                return f"Opération inconnue: {operation}"

        except Exception as e:
            return f"Erreur: {e}"


class SystemInfoAction(Action):
    @property
    def name(self) -> str:
        return "system_info"

    @property
    def description(self) -> str:
        return "Get system information (CPU, RAM, disk, battery)"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        info_type = entities.get("type", "all").lower()

        try:
            parts = []
            if info_type in ("all", "cpu"):
                cpu_percent = psutil.cpu_percent(interval=0.5)
                parts.append(f"CPU: {cpu_percent}%")

            if info_type in ("all", "ram"):
                ram = psutil.virtual_memory()
                parts.append(f"RAM: {ram.percent}% ({ram.used // 1024**3}GB / {ram.total // 1024**3}GB)")

            if info_type in ("all", "disk"):
                disk = psutil.disk_usage("/")
                parts.append(f"Disque: {disk.percent}% ({disk.used // 1024**3}GB / {disk.total // 1024**3}GB)")

            if info_type in ("all", "battery"):
                battery = psutil.sensors_battery()
                if battery:
                    parts.append(f"Batterie: {battery.percent}% {'🔌' if battery.power_plugged else '🔋'}")
                else:
                    parts.append("Batterie: N/A")

            if info_type in ("all", "processes"):
                procs = sorted(psutil.process_iter(['name', 'cpu_percent', 'memory_percent']),
                             key=lambda p: p.info['cpu_percent'] or 0, reverse=True)[:5]
                parts.append("Top processus:")
                for p in procs:
                    parts.append(f"  {p.info['name']}: CPU {p.info['cpu_percent']}% RAM {p.info['memory_percent']:.1f}%")

            return "\n".join(parts)
        except Exception as e:
            return f"Erreur: {e}"


class ScreenshotAction(Action):
    @property
    def name(self) -> str:
        return "screenshot"

    @property
    def description(self) -> str:
        return "Take a screenshot"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        try:
            path = Path.home() / "Pictures" / f"jarvis_screenshot_{int(asyncio.get_event_loop().time())}.png"
            path.parent.mkdir(exist_ok=True)
            await asyncio.to_thread(pyautogui.screenshot, str(path))
            return f"Capture d'écran sauvegardée: {path}"
        except Exception as e:
            return f"Erreur: {e}"


class ChatAction(Action):
    """Basic chat action for greetings and simple conversation (no LLM required)."""

    GREETINGS = {
        "bonjour": "Bonjour ! Je suis JARVIS, votre assistant personnel. Comment puis-je vous aider ?",
        "salut": "Salut ! Comment puis-je vous aider aujourd'hui ?",
        "hello": "Hello! How can I assist you today?",
        "coucou": "Coucou ! Qu'est-ce que je peux faire pour vous ?",
        "bonsoir": "Bonsoir ! En quoi puis-je vous aider ce soir ?",
        "hey": "Hey ! Avez-vous besoin de quelque chose ?",
    }

    @property
    def name(self) -> str:
        return "chat"

    @property
    def description(self) -> str:
        return "General conversation, greetings, questions without specific action"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        text_lower = raw_text.lower().strip()

        # Check greetings
        for greeting, response in self.GREETINGS.items():
            if greeting in text_lower:
                return response

        # Check thanks
        if any(w in text_lower for w in ["merci", "thanks", "thank you"]):
            return "De rien ! N'hésitez pas si vous avez d'autres questions."

        # Check how are you
        if any(w in text_lower for w in ["comment vas-tu", "comment ca va", "how are you", "ça va"]):
            return "Je vais bien, merci ! Prêt à vous aider. Que puis-je faire pour vous ?"

        # Try LLM for more complex queries
        try:
            from jarvis.llm import get_llm_engine
            llm = get_llm_engine()
            response = await llm.complete(
                raw_text,
                system_prompt="Tu es JARVIS, un assistant IA personnel inspire d'Iron Man. Reponds en francais, de maniere concise et utile."
            )
            return response
        except Exception:
            return (f"Je suis JARVIS. Pour l'instant, je fonctionne en mode simple sans LLM. "
                    f"Votre message: '{raw_text}'. "
                    f"Pour des reponses plus cleveres, demarrez Ollama avec 'ollama serve'.")


class VolumeControlAction(Action):
    @property
    def name(self) -> str:
        return "volume_control"

    @property
    def description(self) -> str:
        return "Control system volume (set, up, down, mute)"

    async def execute(self, entities: dict[str, str], raw_text: str) -> str:
        action = entities.get("action", "").lower()
        value = entities.get("value", "")

        try:
            if action == "set" and value:
                cmd = f"Set-Volume -VolumeLevel {float(value)/100}"
            elif action == "up":
                cmd = "(New-Object -ComObject WScript.Shell).SendKeys([char]175)"  # Volume up key
            elif action == "down":
                cmd = "(New-Object -ComObject WScript.Shell).SendKeys([char]174)"  # Volume down key
            elif action == "mute":
                cmd = "(New-Object -ComObject WScript.Shell).SendKeys([char]173)"  # Mute key
            else:
                return "Action volume inconnue (set/up/down/mute)"

            await asyncio.to_thread(subprocess.run, ["powershell", "-Command", cmd], check=True)
            return f"Volume: {action}" + (f" à {value}%" if value else "")
        except Exception as e:
            return f"Erreur: {e}"


class IntentRecognizer:
    """Recognize intent from user text using LLM."""

    def __init__(self) -> None:
        self.config = get_config()
        self.event_bus = get_event_bus()
        self.llm = get_llm_engine()
        self.registry = ActionRegistry()
        self._register_builtin_actions()

    def _register_builtin_actions(self) -> None:
        actions = [
            LaunchAppAction(),
            OpenWebsiteAction(),
            SystemCommandAction(),
            FileOperationAction(),
            SystemInfoAction(),
            ScreenshotAction(),
            VolumeControlAction(),
            ChatAction(),
        ]
        for action in actions:
            self.registry.register(action)

    def build_intent_prompt(self, text: str) -> str:
        actions_desc = self.registry.get_descriptions()
        return f"""Analyse la demande de l'utilisateur et détermine l'intention.

Actions disponibles:
{actions_desc}

Réponds UNIQUEMENT en JSON valide avec ce format:
{{
  "intent": "nom_action",
  "confidence": 0.0-1.0,
  "entities": {{"clé": "valeur"}},
  "reasoning": "explication brève"
}}

Si aucune action ne correspond, utilise "intent": "chat" avec confidence < 0.5.

Demande: "{text}"

JSON:"""

    async def recognize(self, text: str) -> Intent:
        """Recognize intent from text."""
        prompt = self.build_intent_prompt(text)

        try:
            response = await self.llm.complete(prompt)
            logger.debug(f"LLM intent response: {response}")

            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                import json
                data = json.loads(json_match.group())
                intent = Intent(
                    name=data.get("intent", "chat"),
                    confidence=float(data.get("confidence", 0.0)),
                    entities=data.get("entities", {}),
                    raw_text=text,
                )
                # Emit event for plugins
                await self.event_bus.emit(Event(Events.INTENT_RECOGNIZED, {
                    "intent": intent.name,
                    "confidence": intent.confidence,
                    "entities": intent.entities,
                    "raw_text": intent.raw_text,
                }))
                return intent
        except Exception as e:
            logger.error(f"Intent recognition failed: {e}")

        # Fallback: simple keyword matching
        intent = self._fallback_recognize(text)
        await self.event_bus.emit(Event(Events.INTENT_RECOGNIZED, {
            "intent": intent.name,
            "confidence": intent.confidence,
            "entities": intent.entities,
            "raw_text": intent.raw_text,
        }))
        return intent

    def _fallback_recognize(self, text: str) -> Intent:
        text_lower = text.lower()

        # Simple patterns
        patterns = [
            (r"(lance|ouvre|démarre)\s+(\w+)", "launch_app", {"app": r"\2"}),
            (r"(ouvre|va sur)\s+(https?://\S+|\w+\.\w+)", "open_website", {"url": r"\2"}),
            (r"(système|info|status|état)", "system_info", {}),
            (r"(capture|screenshot|écran)", "screenshot", {}),
            (r"(volume|son)\s+(monte|augmente|up)", "volume_control", {"action": "up"}),
            (r"(volume|son)\s+(baisse|descends|down)", "volume_control", {"action": "down"}),
            (r"(volume|son)\s+(muet|mute)", "volume_control", {"action": "mute"}),
            (r"(volume|son)\s+(\d+)", "volume_control", {"action": "set", "value": r"\2"}),
        ]

        for pattern, intent_name, entity_template in patterns:
            match = re.search(pattern, text_lower)
            if match:
                entities = {}
                for key, val_pattern in entity_template.items():
                    # If val_pattern starts with \, it is a group reference like \2
                    # Otherwise it is a literal value
                    # If val_pattern starts with \\, it is a group reference like \\2
                    # Otherwise it is a literal value
                    if isinstance(val_pattern, str) and val_pattern.startswith("\\"):
                        group_num = int(val_pattern[1:])
                        if group_num <= len(match.groups()):
                            entities[key] = match.group(group_num)
                    else:
                        # Literal value
                        entities[key] = val_pattern
                return Intent(name=intent_name, confidence=0.7, entities=entities, raw_text=text)

        return Intent(name="chat", confidence=0.3, entities={}, raw_text=text)


class ActionExecutor:
    """Execute recognized intents."""

    def __init__(self, recognizer: IntentRecognizer) -> None:
        self.recognizer = recognizer
        self.event_bus = get_event_bus()
        self.llm = get_llm_engine()

    async def execute_intent(self, intent: Intent) -> str:
        """Execute an intent and return result."""
        await self.event_bus.emit(Event(Events.ACTION_EXECUTE, {"intent": intent.name}))

        action = self.recognizer.registry.get(intent.name)
        if not action:
            # Fallback to LLM chat
            return await self._chat_fallback(intent.raw_text)

        try:
            result = await action.execute(intent.entities, intent.raw_text)
            await self.event_bus.emit(Event(Events.ACTION_RESULT, {"intent": intent.name, "result": result}))
            return result
        except Exception as e:
            logger.error(f"Action {intent.name} failed: {e}")
            await self.event_bus.emit(Event(Events.ACTION_ERROR, {"intent": intent.name, "error": str(e)}))
            return f"Erreur lors de l'exécution: {e}"

    async def _chat_fallback(self, text: str) -> str:
        """Fallback to general chat with LLM."""
        try:
            response = await self.llm.complete(text)
            return response
        except Exception as e:
            return f"Je ne peux pas repondre pour l'instant (LLM indisponible). Votre message: '{text}'. Demarrez Ollama avec 'ollama serve' ou configurez un fournisseur LLM dans config.yaml."

    async def process_text(self, text: str) -> str:
        """Full pipeline: recognize intent -> execute -> return response."""
        intent = await self.recognizer.recognize(text)
        logger.info(f"Intent: {intent.name} (confidence: {intent.confidence:.2f})")

        if intent.confidence < 0.5 and intent.name != "chat":
            # Low confidence, treat as chat
            intent.name = "chat"

        return await self.execute_intent(intent)


# Global instances
_intent_recognizer: IntentRecognizer | None = None
_action_executor: ActionExecutor | None = None


def get_intent_recognizer() -> IntentRecognizer:
    global _intent_recognizer
    if _intent_recognizer is None:
        _intent_recognizer = IntentRecognizer()
    return _intent_recognizer


def get_action_executor() -> ActionExecutor:
    global _action_executor
    if _action_executor is None:
        _action_executor = ActionExecutor(get_intent_recognizer())
    return _action_executor