"""LLM integration with multiple providers (Ollama, OpenAI, Anthropic)."""
from __future__ import annotations

import asyncio
import json
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


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def complete(self, messages: list[Message], stream: bool = False) -> str | AsyncIterator[str]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, base_url: str, model: str, temperature: float = 0.7, max_tokens: int = 4096) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def complete(self, messages: list[Message], stream: bool = False) -> str | AsyncIterator[str]:
        client = await self._get_client()

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "think": False,  # Disable thinking for faster responses (nemotron, etc.)
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
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

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 4096) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,
            )
        return self._client

    async def complete(self, messages: list[Message], stream: bool = False) -> str | AsyncIterator[str]:
        client = await self._get_client()

        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }

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

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


class AnthropicProvider(LLMProvider):
    """Anthropic API provider."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.7, max_tokens: int = 4096) -> None:
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.anthropic.com/v1",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=60.0,
            )
        return self._client

    async def complete(self, messages: list[Message], stream: bool = False) -> str | AsyncIterator[str]:
        client = await self._get_client()

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

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


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
            return OllamaProvider(config.base_url, config.model, config.temperature, config.max_tokens)
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

    async def complete(
        self,
        user_input: str,
        provider: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """Complete a user request."""
        # Lazy initialize providers if not yet loaded
        if not self._providers:
            await self.initialize()

        llm_provider = self.get_provider(provider)
        if not llm_provider:
            raise RuntimeError("No LLM provider available")

        messages = [Message(role="system", content=system_prompt or self._system_prompt)]
        messages.append(Message(role="user", content=user_input))

        return await llm_provider.complete(messages, stream=stream)

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