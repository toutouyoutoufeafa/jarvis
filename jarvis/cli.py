"""CLI entry point for JARVIS."""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from jarvis.config import Config, get_config, set_config
from jarvis.plugins import get_plugin_manager

app = typer.Typer(
    name="jarvis",
    help="JARVIS - Personal AI Assistant",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    config: Path = typer.Option(None, "--config", "-c", help="Path to config.yaml"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
) -> None:
    """Start JARVIS assistant."""
    if config:
        set_config(Config.from_yaml(config))
    else:
        get_config()  # Load default

    cfg = get_config()
    if debug:
        cfg.logging.level = "DEBUG"

    from jarvis.__main__ import main
    asyncio.run(main())


@app.command()
def config_show() -> None:
    """Show current configuration."""
    cfg = get_config()
    console.print(Panel.fit(f"[bold]JARVIS Configuration[/bold]", style="blue"))

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Section")
    table.add_column("Key")
    table.add_column("Value")

    def add_rows(prefix: str, obj: object) -> None:
        for key, value in obj.__dict__.items():
            if not key.startswith("_"):
                full_key = f"{prefix}.{key}" if prefix else key
                if hasattr(value, "__dict__"):
                    add_rows(full_key, value)
                else:
                    table.add_row(prefix or "root", key, str(value))

    add_rows("", cfg)
    console.print(table)


@app.command()
def config_init(
    path: Path = typer.Option(Path("config.yaml"), "--path", "-p", help="Output path"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing"),
) -> None:
    """Generate default config.yaml."""
    if path.exists() and not force:
        console.print(f"[red]File exists: {path}[/red] Use --force to overwrite")
        raise typer.Exit(1)

    # Read the default config.yaml from project root
    default_path = Path(__file__).parent.parent / "config.yaml"
    if default_path.exists():
        content = default_path.read_text()
    else:
        content = """# JARVIS Configuration
assistant:
  name: "JARVIS"
  wake_word: "jarvis"
  language: "fr"
  response_timeout: 30.0

audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  input_device: null
  output_device: null
  vad_threshold: 0.5
  silence_duration: 1.5

stt:
  engine: "faster-whisper"
  model: "base"
  device: "auto"
  compute_type: "auto"
  language: "fr"
  beam_size: 5

tts:
  engine: "piper"
  voice: "fr_FR-siwis-medium"
  speed: 1.0
  openai_voice: "onyx"
  elevenlabs_voice_id: ""

llm:
  default_provider: "ollama"
  providers:
    ollama:
      base_url: "http://localhost:11434"
      model: "llama3.1:8b"
      temperature: 0.7
      max_tokens: 4096
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4o-mini"
      temperature: 0.7
      max_tokens: 4096

system_control:
  enabled: true
  allowed_apps:
    - "code"
    - "chrome"
    - "firefox"
    - "notepad"
    - "explorer"
    - "cmd"
    - "powershell"
  allowed_paths:
    - "C:\\Users\\${USERNAME}\\Documents"
    - "C:\\Users\\${USERNAME}\\Desktop"

plugins:
  enabled: true
  directories:
    - "plugins"
    - "~/.jarvis/plugins"

logging:
  level: "INFO"
  file: "logs/jarvis.log"
  rotation: "10 MB"
  retention: "7 days"

ui:
  tray_icon: true
  notifications: true
  show_waveform: false
"""

    path.write_text(content)
    console.print(f"[green]Created config: {path}[/green]")


@app.command()
def plugins_list() -> None:
    """List loaded plugins."""
    pm = get_plugin_manager()
    plugins = pm.list_plugins()

    if not plugins:
        console.print("[yellow]No plugins loaded[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name")
    table.add_column("Version")
    table.add_column("Description")
    table.add_column("Author")

    for p in plugins:
        table.add_row(p.name, p.version, p.description, p.author)

    console.print(table)


@app.command()
def test_stt() -> None:
    """Test STT with microphone."""
    asyncio.run(_test_stt())


async def _test_stt() -> None:
    from jarvis.audio import get_audio_input
    from jarvis.stt import get_stt_engine
    from jarvis.events import get_event_bus, Events, Event

    console.print("[bold]Test STT - Parlez maintenant...[/bold]")
    console.print("Appuyez sur Ctrl+C pour arrêter")

    audio_input = get_audio_input()
    stt = get_stt_engine()
    event_bus = get_event_bus()

    results = []

    def on_result(event: Event):
        text = event.data.get("text", "")
        if text:
            results.append(text)
            console.print(f"[green]STT:[/green] {text}")

    event_bus.subscribe(Events.STT_RESULT, on_result)

    await audio_input.start()
    await stt.initialize()

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        await audio_input.stop()
        console.print(f"\n[bold]Résultats:[/bold] {len(results)} énoncés")


@app.command()
def test_tts(text: str = typer.Argument(..., help="Text to synthesize")) -> None:
    """Test TTS synthesis."""
    asyncio.run(_test_tts(text))


async def _test_tts(text: str) -> None:
    from jarvis.tts import get_tts_engine
    from jarvis.audio import get_audio_output

    console.print(f"[bold]Test TTS:[/bold] {text}")

    tts = get_tts_engine()
    audio_output = get_audio_output()

    await tts.initialize()
    await audio_output.start()

    audio = await tts.synthesize(text)
    await audio_output.play(audio)

    console.print("[green]Done![/green]")


@app.command()
def test_llm(prompt: str = typer.Argument(..., help="Prompt for LLM")) -> None:
    """Test LLM completion."""
    asyncio.run(_test_llm(prompt))


async def _test_llm(prompt: str) -> None:
    from jarvis.llm import get_llm_engine

    console.print(f"[bold]Test LLM:[/bold] {prompt}")

    llm = get_llm_engine()
    await llm.initialize()

    response = await llm.complete(prompt)
    console.print(f"[green]Response:[/green] {response}")


@app.command()
def test_intent(text: str = typer.Argument(..., help="Text to parse for intent")) -> None:
    """Test intent recognition."""
    asyncio.run(_test_intent(text))


async def _test_intent(text: str) -> None:
    from jarvis.intent import get_intent_recognizer, get_action_executor

    console.print(f"[bold]Test Intent:[/bold] {text}")

    recognizer = get_intent_recognizer()
    executor = get_action_executor()

    intent = await recognizer.recognize(text)
    console.print(f"Intent: [cyan]{intent.name}[/cyan] (confidence: {intent.confidence:.2f})")
    console.print(f"Entities: {intent.entities}")

    result = await executor.execute_intent(intent)
    console.print(f"[green]Result:[/green] {result}")


@app.command()
def agent_run(goal: str = typer.Argument(..., help="Goal for agent")) -> None:
    """Run agent with a goal."""
    asyncio.run(_agent_run(goal))


async def _agent_run(goal: str) -> None:
    from jarvis.agent import get_agent_core

    console.print(f"[bold]Agent Run:[/bold] {goal}")

    agent = get_agent_core()
    await agent.initialize()

    task = await agent.run_task(goal)
    console.print(f"State: [cyan]{task.state.value}[/cyan]")
    console.print(f"Iterations: {task.iterations}")
    if task.result:
        console.print(f"[green]Result:[/green] {task.result}")
    if task.error:
        console.print(f"[red]Error:[/red] {task.error}")


@app.command()
def designer_code(
    spec: str = typer.Argument(..., help="Code specification"),
    language: str = typer.Option("python", "--lang", "-l"),
    output: str = typer.Option(None, "--output", "-o"),
) -> None:
    """Generate code using designer agent."""
    asyncio.run(_designer_code(spec, language, output))


async def _designer_code(spec: str, language: str, output: str | None) -> None:
    from jarvis.agent import get_designer_agent, DesignTask

    console.print(f"[bold]Designer Code:[/bold] {spec}")

    designer = get_designer_agent()
    await designer.initialize()

    task = DesignTask(type="code", spec=spec, language=language, output_path=output)
    result = await designer.generate_code(task)
    console.print(f"[green]Generated:[/green] {result.get('saved_to', 'memory')}")
    console.print(result["code"][:500] + ("..." if len(result["code"]) > 500 else ""))


@app.command()
def designer_arch(spec: str = typer.Argument(..., help="Architecture specification"), output: str = typer.Option(None, "--output", "-o")) -> None:
    """Design architecture using designer agent."""
    asyncio.run(_designer_arch(spec, output))


async def _designer_arch(spec: str, output: str | None) -> None:
    from jarvis.agent import get_designer_agent, DesignTask

    console.print(f"[bold]Designer Architecture:[/bold] {spec}")

    designer = get_designer_agent()
    await designer.initialize()

    task = DesignTask(type="architecture", spec=spec, output_path=output)
    result = await designer.design_architecture(task)
    console.print(f"[green]Generated:[/green] {result.get('saved_to', 'memory')}")
    console.print(result["architecture"][:1000] + ("..." if len(result["architecture"]) > 1000 else ""))


@app.command()
def monitor_status() -> None:
    """Get monitor agent status."""
    asyncio.run(_monitor_status())


async def _monitor_status() -> None:
    from jarvis.agent import get_monitor_agent

    monitor = get_monitor_agent()
    await monitor.initialize()

    status = await monitor.get_status()
    import json
    console.print_json(json.dumps(status))


@app.command()
def monitor_health() -> None:
    """Run health check."""
    asyncio.run(_monitor_health())


async def _monitor_health() -> None:
    from jarvis.agent import get_monitor_agent

    monitor = get_monitor_agent()
    await monitor.initialize()

    health = await monitor.run_health_check()
    import json
    console.print_json(json.dumps(health))


@app.command()
def reflector_reflect(
    topic: str = typer.Argument(..., help="Topic to reflect on"),
    depth: int = typer.Option(3, "--depth", "-d"),
    context: str = typer.Option("", "--context", "-c"),
) -> None:
    """Run reflection on a topic."""
    asyncio.run(_reflector_reflect(topic, depth, context))


async def _reflector_reflect(topic: str, depth: int, context: str) -> None:
    from jarvis.agent import get_reflector_agent

    console.print(f"[bold]Reflector:[/bold] {topic}")

    reflector = get_reflector_agent()
    await reflector.initialize()

    reflection = await reflector.reflect(topic, context, depth)
    console.print(f"[cyan]Critique:[/cyan] {reflection.critique}")
    console.print("[cyan]Improvements:[/cyan]")
    for i, imp in enumerate(reflection.improvements, 1):
        console.print(f"  {i}. {imp}")


@app.command()
def spawner_list() -> None:
    """List spawned agents."""
    asyncio.run(_spawner_list())


async def _spawner_list() -> None:
    from jarvis.agent import get_spawner

    spawner = get_spawner()
    await spawner.initialize()

    agents = spawner.list_agents()
    if not agents:
        console.print("[yellow]No agents spawned[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID")
    table.add_column("Type")
    table.add_column("Goal")
    table.add_column("State")
    table.add_column("Created")

    for a in agents:
        table.add_row(a["id"], a["type"], a["goal"], a["state"], a["created"])

    console.print(table)


@app.command()
def spawner_spawn(
    goal: str = typer.Argument(..., help="Goal for sub-agent"),
    type: str = typer.Option("generic", "--type", "-t"),
    timeout: int = typer.Option(300, "--timeout"),
) -> None:
    """Spawn a sub-agent."""
    asyncio.run(_spawner_spawn(goal, type, timeout))


async def _spawner_spawn(goal: str, type: str, timeout: int) -> None:
    from jarvis.agent import get_spawner, AgentType, SubAgentSpec

    console.print(f"[bold]Spawning sub-agent:[/bold] {goal}")

    spawner = get_spawner()
    await spawner.initialize()

    spec = SubAgentSpec(
        name=f"cli_agent_{type}",
        agent_type=AgentType(type),
        goal=goal,
        timeout=timeout,
    )

    agent = await spawner.spawn_and_wait(spec)
    console.print(f"State: [cyan]{agent.state.value}[/cyan]")
    if agent.result:
        console.print(f"[green]Result:[/green] {agent.result}")
    if agent.error:
        console.print(f"[red]Error:[/red] {agent.error}")


if __name__ == "__main__":
    app()