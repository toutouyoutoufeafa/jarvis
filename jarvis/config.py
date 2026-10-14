"""Configuration management for JARVIS."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioConfig(BaseSettings):
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    input_device: int | None = None
    output_device: int | None = None
    vad_threshold: float = 0.5
    silence_duration: float = 1.5


class STTConfig(BaseSettings):
    engine: str = "faster-whisper"
    model: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    language: str = "fr"
    beam_size: int = 5


class TTSConfig(BaseSettings):
    engine: str = "piper"
    voice: str = "fr_FR-siwis-medium"
    speed: float = 1.0
    openai_voice: str = "onyx"
    elevenlabs_voice_id: str = ""


class LLMProviderConfig(BaseSettings):
    base_url: str | None = None
    api_key: str | None = None
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096


class LLMConfig(BaseSettings):
    default_provider: str = "ollama"
    providers: dict[str, LLMProviderConfig] = Field(default_factory=dict)
    # Models offered in the UI selector (intersected with what Ollama actually has)
    available_models: list[str] = Field(default_factory=list)


class SystemControlConfig(BaseSettings):
    enabled: bool = True
    allowed_apps: list[str] = Field(default_factory=list)
    allowed_paths: list[str] = Field(default_factory=list)


class PluginsConfig(BaseSettings):
    enabled: bool = True
    directories: list[str] = Field(default_factory=list)


class LoggingConfig(BaseSettings):
    level: str = "INFO"
    file: str = "logs/jarvis.log"
    rotation: str = "10 MB"
    retention: str = "7 days"


class UIConfig(BaseSettings):
    tray_icon: bool = True
    notifications: bool = True
    show_waveform: bool = False


class AssistantConfig(BaseSettings):
    name: str = "JARVIS"
    wake_word: str = "jarvis"
    language: str = "fr"
    response_timeout: float = 30.0


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    system_control: SystemControlConfig = Field(default_factory=SystemControlConfig)
    plugins: PluginsConfig = Field(default_factory=PluginsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load config from YAML file with environment variable expansion."""
        path = Path(path)
        if not path.exists():
            return cls()

        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Expand environment variables in string values
        def expand_env(obj: Any) -> Any:
            if isinstance(obj, str):
                return os.path.expandvars(obj)
            if isinstance(obj, dict):
                return {k: expand_env(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [expand_env(v) for v in obj]
            return obj

        raw = expand_env(raw)
        return cls(**raw)


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config.from_yaml("config.yaml")
    return _config


def set_config(config: Config) -> None:
    global _config
    _config = config