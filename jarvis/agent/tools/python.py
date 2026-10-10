"""Python code execution tool with sandboxing."""
from __future__ import annotations

import asyncio
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from jarvis.agent.tools import tool
from jarvis.agent.config import get_agent_config
from jarvis.agent.sandbox import get_sandbox


@tool("python_exec", "Execute Python code in sandbox")
async def python_exec(
    code: str,
    timeout: int = 30,
    packages: list[str] | None = None,
    return_vars: list[str] | None = None,
) -> dict:
    """Execute Python code in sandboxed environment."""
    config = get_agent_config()

    if not config.sandbox.enabled:
        raise PermissionError("Sandbox disabled, code execution not allowed")

    sandbox = get_sandbox()
    return await sandbox.run_python(code, timeout, packages, return_vars)


@tool("python_repl", "Run Python REPL session")
async def python_repl(
    code: str,
    session_id: str | None = None,
    timeout: int = 30,
) -> dict:
    """Run Python code in persistent REPL session."""
    sandbox = get_sandbox()
    return await sandbox.run_repl(code, session_id, timeout)


@tool("pip_install", "Install Python package in sandbox")
async def pip_install(package: str, timeout: int = 60) -> dict:
    """Install package in sandbox environment."""
    sandbox = get_sandbox()
    return await sandbox.install_package(package, timeout)


def register_python_tools() -> None:
    pass