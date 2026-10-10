"""Agent configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SandboxConfig(BaseSettings):
    enabled: bool = True
    type: str = "subprocess"  # subprocess / docker / nsjail
    docker_image: str = "python:3.11-slim"
    timeout: int = 30
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    network_enabled: bool = False
    allowed_paths: list[str] = Field(default_factory=lambda: [
        str(Path.home() / "jarvis_workspace"),
        "/tmp/jarvis",
    ])


class MemoryConfig(BaseSettings):
    enabled: bool = True
    vector_db_path: str = "~/.jarvis/memory/chroma"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    stm_max_tokens: int = 4000
    ltm_max_entries: int = 10000
    auto_summarize: bool = True
    summarize_threshold: int = 100


class PlannerConfig(BaseSettings):
    enabled: bool = True
    max_steps: int = 20
    max_retries: int = 3
    planning_model: str = ""  # Empty = use default LLM
    use_cot: bool = True


class DesignerConfig(BaseSettings):
    enabled: bool = True
    templates_path: str = "~/.jarvis/templates"
    default_language: str = "python"
    auto_test: bool = True
    lint_before_save: bool = True


class MonitorConfig(BaseSettings):
    enabled: bool = True
    check_interval: int = 60
    log_paths: list[str] = Field(default_factory=lambda: [
        "/var/log/syslog",
        "~/.jarvis/logs/jarvis.log",
    ])
    metrics_interval: int = 30
    alert_thresholds: dict[str, float] = Field(default_factory=lambda: {
        "cpu_percent": 90.0,
        "memory_percent": 85.0,
        "disk_percent": 90.0,
    })


class ReflectorConfig(BaseSettings):
    enabled: bool = True
    reflection_depth: int = 3
    critique_model: str = ""
    auto_improve: bool = True


class SpawnerConfig(BaseSettings):
    enabled: bool = True
    max_concurrent_agents: int = 5
    default_timeout: int = 300
    token_budget_per_agent: int = 50000


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    designer: DesignerConfig = Field(default_factory=DesignerConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    reflector: ReflectorConfig = Field(default_factory=ReflectorConfig)
    spawner: SpawnerConfig = Field(default_factory=SpawnerConfig)

    # Global
    default_model: str = Field(default="llama3.1:8b")
    max_iterations: int = Field(default=50)
    verbose: bool = Field(default=True)

    # Global instance
_agent_config: AgentConfig | None = None


def get_agent_config() -> AgentConfig:
    global _agent_config
    if _agent_config is None:
        _agent_config = AgentConfig()
    return _agent_config