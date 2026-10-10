"""Tool system for agents."""
from __future__ import annotations

import asyncio
import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints

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


class Tool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        pass

    def get_schema(self) -> ToolSchema:
        """Auto-generate schema from execute method signature."""
        sig = inspect.signature(self.execute)
        hints = get_type_hints(self.execute)
        params = []

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            param_type = hints.get(name, Any)
            required = param.default == inspect.Parameter.empty
            default = param.default if not required else None
            params.append(ToolParameter(
                name=name,
                type=param_type,
                description=f"Parameter {name}",
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

            async def execute(self, **kwargs) -> Any:
                if asyncio.iscoroutinefunction(func):
                    return await func(**kwargs)
                return await asyncio.to_thread(func, **kwargs)

        get_tool_registry().register(FunctionTool())
        return func

    return decorator