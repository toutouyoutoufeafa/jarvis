"""LLM integration with multiple providers (Ollama, OpenAI, Anthropic)."""
from __future__ import annotations

import asyncio
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import httpx
from loguru import logger
from pydantic import BaseModel

from jarvis.config import get_config
from jarvis.events import Event, Events, get_event_bus


@dataclass
class Message:
    role: str  # system, user, assistant
    content: str


def extract_json(text: str | None) -> dict | list | None:
    """Pull the first valid JSON value out of raw model output.

    Small local models routinely wrap JSON in markdown fences, add a leading
    sentence, or append stray braces. A greedy regex `\\{.*\\}` blows up on
    those, so we scan for the first character that starts a JSON value and let
    ``raw_decode`` stop at the matching end.
    """
    if not text:
        return None

    cleaned = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned, index)
        except ValueError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None


class LLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None

    async def _client_for(self, timeout: float = 60.0, **kwargs) -> httpx.AsyncClient:
        """Return an httpx client bound to the *current* event loop.

        Streamlit (and ``run_async``) create a fresh event loop per request.
        Reusing a client created in an earlier, now-closed loop raises
        ``RuntimeError: Event loop is closed``, so the client is rebuilt
        whenever the running loop changes.
        """
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            self._client = httpx.AsyncClient(timeout=timeout, **kwargs)
            self._client_loop = loop
        return self._client

    @staticmethod
    def _apply_options(payload: dict, options: Optional[dict]) -> dict:
        """Map generic generation options onto an OpenAI/Anthropic payload."""
        if not options:
            return payload
        mapping = {"num_predict": "max_tokens", "temperature": "temperature",
                   "top_p": "top_p", "stop": "stop"}
        for key, value in options.items():
            target = mapping.get(key)
            if target and value is not None:
                payload[target] = value
        return payload

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
        options: Optional[dict] = None,
    ) -> str | AsyncIterator[str]:
        """Complete a chat request.

        ``options`` carries provider-specific generation overrides (for
        Ollama: ``num_predict``, ``temperature``, ``top_p``...). Agent loops
        rely on it to keep tool decisions short and deterministic.
        """
        pass

    async def close(self) -> None:
        if self._client is None:
            return
        client, loop = self._client, self._client_loop
        self._client, self._client_loop = None, None
        if loop is None or loop.is_closed():
            return  # nothing sane to await against a dead loop
        try:
            await client.aclose()
        except RuntimeError:
            pass


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        num_ctx: int = 2048,
        keep_alive: str = "30m",
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive

    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
        options: Optional[dict] = None,
    ) -> str | AsyncIterator[str]:
        client = await self._client_for(60.0)

        merged = {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "num_ctx": self.num_ctx,
        }
        if options:
            merged.update({k: v for k, v in options.items() if v is not None})

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "think": False,  # Disable thinking for faster responses (nemotron, etc.)
            "keep_alive": self.keep_alive,
            "options": merged,
        }

        if stream:
            return self._stream_complete(client, payload)
        else:
            response = await client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]

    async def _stream_complete(self, client: httpx.AsyncClient, payload: dict) -> AsyncIterator[str]:
        async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]
                    except json.JSONDecodeError:
                        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 4096) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _openai_client(self) -> httpx.AsyncClient:
        return await self._client_for(
            60.0,
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
        options: Optional[dict] = None,
    ) -> str | AsyncIterator[str]:
        client = await self._openai_client()

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        payload = self._apply_options(payload, options)

        if stream:
            return self._stream_complete(client, payload)
        else:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _stream_complete(self, client: httpx.AsyncClient, payload: dict) -> AsyncIterator[str]:
        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        pass


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 4096) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def _anthropic_client(self) -> httpx.AsyncClient:
        return await self._client_for(
            60.0,
            base_url="https://api.anthropic.com/v1",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )

    async def complete(
        self,
        messages: list[Message],
        stream: bool = False,
        options: Optional[dict] = None,
    ) -> str | AsyncIterator[str]:
        client = await self._anthropic_client()

        # Convert messages: system prompt separate, rest as user/assistant
        system = ""
        anthropic_messages = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        payload = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        payload = self._apply_options(payload, options)
        if system:
            payload["system"] = system

        if stream:
            return self._stream_complete(client, payload)
        else:
            response = await client.post("/messages", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]

    async def _stream_complete(self, client: httpx.AsyncClient, payload: dict) -> AsyncIterator[str]:
        async with client.stream("POST", "/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if data["type"] == "content_block_delta":
                            yield data["delta"]["text"]
                    except json.JSONDecodeError:
                        pass


class LLMEngine:
    """Main LLM engine managing multiple providers."""

    def __init__(self) -> None:
        self.config = get_config()
        self.event_bus = get_event_bus()
        self._providers: dict[str, LLMProvider] = {}
        self._default_provider: Optional[LLMProvider] = None
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        name = self.config.assistant.name
        return f"""You are {name}, a personal AI assistant inspired by JARVIS from Iron Man.
You are helpful, concise, and speak naturally in French.
You can control the user's computer, run commands, search the web, and manage tasks.
Always respond in French unless asked otherwise.
Be proactive but not verbose. Use a professional yet friendly tone."""

    async def initialize(self) -> None:
        """Initialize all configured providers."""
        for name, provider_config in self.config.llm.providers.items():
            try:
                provider = self._create_provider(name, provider_config)
                if provider:
                    self._providers[name] = provider
                    logger.info(f"LLM provider loaded: {name} ({provider_config.model})")
            except Exception as e:
                logger.error(f"Failed to load LLM provider {name}: {e}")

        # Set default
        default_name = self.config.llm.default_provider
        self._default_provider = self._providers.get(default_name)
        if not self._default_provider and self._providers:
            self._default_provider = next(iter(self._providers.values()))

        if not self._default_provider:
            logger.warning("No LLM provider available!")

    def _create_provider(self, name: str, config) -> Optional[LLMProvider]:
        if name == "ollama":
            return OllamaProvider(
                config.base_url,
                config.model,
                config.temperature,
                config.max_tokens,
                config.num_ctx,
                config.keep_alive,
            )
        elif name == "openai":
            if not config.api_key:
                return None
            return OpenAIProvider(config.api_key, config.model, config.temperature, config.max_tokens)
        elif name == "anthropic":
            if not config.api_key:
                return None
            return AnthropicProvider(config.api_key, config.model, config.temperature, config.max_tokens)
        return None

    def get_provider(self, name: Optional[str] = None) -> Optional[LLMProvider]:
        if name:
            return self._providers.get(name)
        return self._default_provider

    def set_model(self, model: str) -> bool:
        """Switch the active Ollama model at runtime."""
        provider = self._providers.get(self.config.llm.default_provider)
        if isinstance(provider, OllamaProvider):
            if provider.model != model:
                logger.info(f"LLM model switched: {provider.model} -> {model}")
                provider.model = model
            return True
        return False

    def get_model(self) -> Optional[str]:
        """Return the currently active model name."""
        provider = self._providers.get(self.config.llm.default_provider)
        if isinstance(provider, OllamaProvider):
            return provider.model
        return None

    async def complete(
        self,
        user_input: str,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        history: Optional[list[Message]] = None,
        options: Optional[dict] = None,
    ) -> str | AsyncIterator[str]:
        """Complete a user request.

        Args:
            user_input: The latest user message.
            provider: Provider name override.
            system_prompt: System prompt override.
            stream: Stream the response.
            history: Prior turns (user/assistant) for conversational context.
            options: Generation overrides (Ollama: ``num_predict``,
                ``temperature``...). Used by the agent to cap output length.
        """
        # Lazy initialize providers if not yet loaded
        if not self._providers:
            await self.initialize()

        llm_provider = self.get_provider(provider)
        if not llm_provider:
            raise RuntimeError("No LLM provider available")

        messages = [Message(role="system", content=system_prompt or self._system_prompt)]
        if history:
            # Keep only role/content pairs, drop malformed entries
            for m in history:
                if getattr(m, "role", None) in ("user", "assistant") and getattr(m, "content", None):
                    messages.append(Message(role=m.role, content=m.content))
        messages.append(Message(role="user", content=user_input))

        return await llm_provider.complete(messages, stream=stream, options=options)

    async def process_event(self, event: Event) -> None:
        """Handle LLM_REQUEST event."""
        user_input = event.data.get("text", "")
        provider = event.data.get("provider")
        stream = event.data.get("stream", False)

        if not user_input:
            return

        await self.event_bus.emit(Event(Events.LLM_REQUEST, {"text": user_input}))

        try:
            if stream:
                async for chunk in await self.complete(user_input, provider, stream=True):
                    await self.event_bus.emit(Event(Events.LLM_STREAM_CHUNK, {"chunk": chunk}))
                await self.event_bus.emit(Event(Events.LLM_RESPONSE, {"text": ""}))  # End marker
            else:
                response = await self.complete(user_input, provider)
                await self.event_bus.emit(Event(Events.LLM_RESPONSE, {"text": response}))
        except Exception as e:
            logger.error(f"LLM error: {e}")
            await self.event_bus.emit(Event(Events.LLM_ERROR, {"error": str(e)}))

    async def shutdown(self) -> None:
        for provider in self._providers.values():
            await provider.close()


# Global instance
_llm_engine: LLMEngine | None = None


def get_llm_engine() -> LLMEngine:
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine()
    return _llm_engine


async def list_installed_models() -> list[str]:
    """Return models actually installed in the local Ollama server."""
    cfg = get_config()
    ollama_cfg = cfg.llm.providers.get("ollama")
    if not ollama_cfg:
        return []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{ollama_cfg.base_url.rstrip('/')}/api/tags")
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        logger.debug(f"Could not list Ollama models: {e}")
        return []


async def available_models() -> list[str]:
    """Models offered to the user: configured ones that are installed, else all installed."""
    cfg = get_config()
    installed = await list_installed_models()
    if not installed:
        return list(cfg.llm.available_models)
    configured = [m for m in cfg.llm.available_models if m in installed]
    return configured or installed