"""Tool system for agents."""
from __future__ import annotations

import asyncio
import inspect
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Optional, get_type_hints

from loguru import logger
from pydantic import BaseModel, Field, create_model


@dataclass
class ToolParameter:
    name: str
    type: type
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: list[ToolParameter]
    returns: str = "Any"

    def to_openai_schema(self) -> dict:
        """Convert to OpenAI function calling schema."""
        properties = {}
        required = []
        for param in self.parameters:
            prop = {"type": self._python_type_to_json(param.type), "description": param.description}
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_compact_line(self) -> str:
        """One-line signature: `name(param:type!, ...) - description`.

        Far lighter than the full JSON schema, which small models choke on.
        Defaults are omitted: they cost tokens and the model rarely needs them.
        """
        params = []
        for p in self.parameters:
            type_name = (
                getattr(p.type, "__name__", str(p.type))
                .split("[")[0].split("|")[0].strip() or "any"
            )
            marker = "" if p.required else "?"
            params.append(f"{p.name}:{type_name}{marker}")
        desc = f" - {self.description}" if self.description else ""
        return f"{self.name}({', '.join(params)}){desc}"

    def _python_type_to_json(self, py_type: type) -> str:
        if py_type == str:
            return "string"
        elif py_type == int:
            return "integer"
        elif py_type == float:
            return "number"
        elif py_type == bool:
            return "boolean"
        elif py_type == list:
            return "array"
        elif py_type == dict:
            return "object"
        return "string"


# Curated per-parameter wording: docstrings here are one-liners, and an opaque
# `Parameter content` teaches a small model nothing about what to pass.
_PARAM_HINTS: dict[str, str] = {
    "path": "file or directory path",
    "content": "text to write into the file",
    "cmd": "shell command to execute",
    "command": "shell command to execute",
    "code": "python source code to execute",
    "query": "search query",
    "url": "http(s) url",
    "dest_path": "destination path",
    "src": "source path",
    "dest": "destination path",
    "dst": "destination path",
    "message": "text content",
    "repo": "path of the git repository",
    "branch": "branch name",
    "files": "list of file paths",
    "pattern": "glob or regex pattern",
    "app": "application name or executable",
    "prompt": "text prompt",
    "max_results": "maximum number of results",
    "timeout": "timeout in seconds",
    "create_dirs": "create missing parent directories",
    "recursive": "apply recursively",
}


def _parse_doc_params(doc: str | None) -> dict[str, str]:
    """Extract ``:param name: description`` entries from a docstring."""
    if not doc:
        return {}
    found = dict(re.findall(r":param\s+(\w+)\s*:\s*(.+)", doc))
    return {k: v.strip() for k, v in found.items()}


class Tool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        pass

    def get_schema(self) -> ToolSchema:
        """Auto-generate schema from the tool's real callable signature.

        ``FunctionTool.execute`` is ``**kwargs``-only, so it must expose the
        wrapped function as ``_signature_func``; otherwise every tool would
        advertise a single meaningless ``kwargs`` parameter.
        """
        target = getattr(self, "_signature_func", None) or self.execute
        sig = inspect.signature(target)
        try:
            hints = get_type_hints(target)
        except Exception:
            hints = {}
        doc_params = _parse_doc_params(getattr(target, "__doc__", None))
        params = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue  # *args/**kwargs carry no information for the model
            param_type = hints.get(name, Any)
            required = param.default == inspect.Parameter.empty
            default = param.default if not required else None
            params.append(ToolParameter(
                name=name,
                type=param_type,
                description=doc_params.get(name) or _PARAM_HINTS.get(name) or name.replace("_", " "),
                required=required,
                default=default,
            ))

        return_type = hints.get("return", Any)
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=params,
            returns=str(return_type),
        )


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def describe_all(self, full: Collection[str] | None = None) -> str:
        """Compact catalogue of every tool.

        With ``full``, only the named tools get a signature line; the rest are
        listed by name. Prompt evaluation costs ~44 tok/s on local CPU, so a
        bare name for a self-explanatory tool (``git_push``) is worth keeping
        over a redundant signature.
        """
        lines = []
        for tool in sorted(self._tools.values(), key=lambda t: t.name):
            if full is None or tool.name in full:
                lines.append(f"- {tool.get_schema().to_compact_line()}")
            else:
                lines.append(f"- {tool.name}")
        return "\n".join(lines)

    def get_schemas(self) -> list[ToolSchema]:
        return [tool.get_schema() for tool in self._tools.values()]

    def get_openai_schemas(self) -> list[dict]:
        return [tool.get_schema().to_openai_schema() for tool in self._tools.values()]

    async def execute(self, name: str, **kwargs) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Tool not found: {name}")
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            raise


# Global registry
_tool_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def tool(name: str | None = None, description: str = "") -> Callable:
    """Decorator to register a function as a tool."""

    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""

        class FunctionTool(Tool):
            def __init__(self) -> None:
                super().__init__(tool_name, tool_desc)
                # Keep the real callable so the schema can show its signature
                self._signature_func = func

            async def execute(self, **kwargs) -> Any:
                if asyncio.iscoroutinefunction(func):
                    return await func(**kwargs)
                return await asyncio.to_thread(func, **kwargs)

        get_tool_registry().register(FunctionTool())
        return func

    return decorator