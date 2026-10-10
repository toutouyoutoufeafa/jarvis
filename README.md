# JARVIS - Personal AI Assistant

A modular, extensible personal AI assistant inspired by JARVIS from Iron Man. Built with Python, featuring voice interaction, local-first architecture with cloud fallback, and a plugin system.

## 🎯 Features

- **Voice Interface**: Speech-to-Text (faster-whisper) + Text-to-Speech (Piper TTS)
- **Hybrid LLM**: Local (Ollama) + Cloud (OpenAI, Anthropic) with automatic fallback
- **Intent Recognition**: Natural language → structured actions
- **System Control**: Launch apps, file ops, system info, volume, screenshots, windows
- **Plugin System**: Extend with custom actions, integrations, skills
- **Privacy-First**: Runs fully local by default, cloud only when configured
- **Modular Architecture**: Clean separation of concerns, easy to extend

## 🏗 Architecture

```
jarvis/
├── config.py          # Configuration management (Pydantic + YAML)
├── events.py          # Async event bus for inter-component communication
├── plugins.py         # Plugin system (discovery, lifecycle, metadata)
├── audio.py           # Audio I/O with VAD (Voice Activity Detection)
├── stt.py             # Speech-to-Text (faster-whisper)
├── tts.py             # Text-to-Speech (Piper + cloud fallback)
├── llm.py             # LLM providers (Ollama, OpenAI, Anthropic)
├── intent.py          # Intent recognition + action execution
├── cli.py             # Typer CLI for testing & management
└── __main__.py        # Main application entry point
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Ollama** (for local LLM): `curl -fsSL https://ollama.ai/install.sh | sh`
- **Piper TTS** (for local voice): `pip install piper-tts`
- **Audio**: PortAudio (usually pre-installed on Windows)

### Installation

```bash
# Clone / navigate to project
cd E:\Jarvis

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -e ".[dev]"

# Pull a local LLM model
ollama pull llama3.1:8b

# Download a Piper voice (French)
# See: https://huggingface.co/rhasspy/piper-voices
mkdir -p ~/.local/share/piper/voices
# Download fr_FR-siwis-medium.onnx and .onnx.json to that folder
```

### Configuration

```bash
# Generate default config
jarvis config-init

# Edit config.yaml with your settings
# - Set API keys for cloud providers (optional)
# - Adjust audio devices if needed
# - Configure allowed paths/apps
```

### Run

```bash
# Start JARVIS
jarvis run

# Or with debug logging
jarvis run --debug

# Test individual components
jarvis test-stt          # Test microphone + STT
jarvis test-tts "Bonjour"  # Test TTS
jarvis test-llm "Quelle heure est-il ?"  # Test LLM
jarvis test-intent "Lance Chrome"  # Test intent recognition
```

## ⚙️ Configuration

Key settings in `config.yaml`:

```yaml
assistant:
  name: "JARVIS"
  wake_word: "jarvis"      # Future: wake word detection
  language: "fr"

stt:
  engine: "faster-whisper"
  model: "base"            # tiny, base, small, medium, large-v3
  device: "auto"           # auto, cpu, cuda

tts:
  engine: "piper"          # piper, openai, elevenlabs
  voice: "fr_FR-siwis-medium"

llm:
  default_provider: "ollama"
  providers:
    ollama:
      base_url: "http://localhost:11434"
      model: "llama3.1:8b"
    openai:
      api_key: "${OPENAI_API_KEY}"  # From .env or env var
      model: "gpt-4o-mini"

system_control:
  allowed_apps: ["code", "chrome", "firefox", "notepad", "explorer"]
  allowed_paths:
    - "C:\\Users\\${USERNAME}\\Documents"
    - "C:\\Users\\${USERNAME}\\Desktop"
```

## 🎤 Voice Commands Examples

| Command | Action |
|---------|--------|
| *"Lance Chrome"* | Launch application |
| *"Ouvre github.com"* | Open website |
| *"Crée un fichier test.txt avec du contenu"* | File operations |
| *"Montre l'info système"* | CPU/RAM/Disk info |
| *"Volume à 50"* | Set volume |
| *"Capture l'écran"* | Screenshot |
| *"Minimise la fenêtre Chrome"* | Window management (plugin) |
| *"Quelle est la météo à Paris ?"* | Weather (plugin) |

## 🔌 Writing Plugins

Create a Python file in `plugins/` or `~/.jarvis/plugins/`:

```python
# plugins/my_plugin.py
from jarvis.plugins import Plugin, PluginMetadata
from jarvis.intent import Action
from jarvis.events import get_event_bus, Events, Event

class MyAction(Action):
    @property
    def name(self) -> str: return "my_action"
    @property
    def description(self) -> str: return "Description"

    async def execute(self, entities, raw_text):
        return "Result"

class MyPlugin(Plugin):
    def _get_metadata(self):
        return PluginMetadata(name="my_plugin", version="0.1.0")

    async def _initialize(self):
        get_intent_recognizer().registry.register(MyAction())

def create_plugin():
    return MyPlugin()
```

## 🧪 Testing

```bash
# Run tests
pytest

# Lint
ruff check .

# Type check
mypy jarvis
```

## 📁 Project Structure

```
E:\Jarvis\
├── pyproject.toml         # Project config, dependencies
├── config.yaml            # User configuration
├── .env                   # API keys (not committed)
├── jarvis/                # Core package
│   ├── __init__.py
│   ├── __main__.py        # Entry point
│   ├── cli.py             # CLI commands
│   ├── config.py
│   ├── events.py
│   ├── plugins.py
│   ├── audio.py
│   ├── stt.py
│   ├── tts.py
│   ├── llm.py
│   └── intent.py
├── plugins/               # User plugins
│   ├── weather_plugin.py
│   └── system_control_plugin.py
├── logs/                  # Runtime logs
└── tests/                 # Unit tests
```

## 🔐 Privacy & Security

- **Local-first**: STT, TTS, LLM run locally by default
- **No telemetry**: Zero data leaves your machine unless you configure cloud APIs
- **Sandboxed actions**: File/app access restricted to configured paths
- **No wake word recording**: Continuous listening only processes locally

## 🗺 Roadmap

- [ ] Wake word detection (Porcupine / custom)
- [ ] Streaming TTS for lower latency
- [ ] GUI / System tray app
- [ ] Home Assistant integration
- [ ] Multi-language support
- [ ] Conversation memory / context
- [ ] Plugin marketplace

## 📄 License

MIT License - Feel free to use and modify.

---

**JARVIS** - *"Sometimes you gotta run before you can walk."* — Tony Stark