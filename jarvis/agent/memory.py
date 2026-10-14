"""Agent memory system (STM + LTM with vector store)."""
from __future__ import annotations

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from loguru import logger
from sentence_transformers import SentenceTransformer

from jarvis.agent.config import get_agent_config


@dataclass
class MemoryEntry:
    id: str
    content: str
    metadata: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: list[float] | None = None
    summary: str | None = None


class MemoryStore(ABC):
    @abstractmethod
    async def add(self, entry: MemoryEntry) -> None: ...

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[MemoryEntry]: ...

    @abstractmethod
    async def get(self, entry_id: str) -> MemoryEntry | None: ...

    @abstractmethod
    async def delete(self, entry_id: str) -> bool: ...

    @abstractmethod
    async def clear(self) -> None: ...


class ChromaMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self.config = get_agent_config()
        self._client: chromadb.ClientAPI | None = None
        self._collection = None
        self._embedder: SentenceTransformer | None = None

    async def initialize(self) -> None:
        path = Path(self.config.memory.vector_db_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="jarvis_memory",
            metadata={"hnsw:space": "cosine"},
        )

        # Load embedder in thread pool
        loop = asyncio.get_event_loop()
        self._embedder = await loop.run_in_executor(
            None,
            lambda: SentenceTransformer(self.config.memory.embedding_model),
        )
        logger.info(f"Memory store initialized at {path}")

    def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            raise RuntimeError("Embedder not initialized")
        return self._embedder.encode(text).tolist()

    async def add(self, entry: MemoryEntry) -> None:
        if self._collection is None:
            await self.initialize()

        embedding = entry.embedding or self._embed(entry.content)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._collection.add(
                ids=[entry.id],
                documents=[entry.content],
                embeddings=[embedding],
                metadatas=[{
                    **entry.metadata,
                    "timestamp": entry.timestamp.isoformat(),
                    "summary": entry.summary or "",
                }],
            ),
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[MemoryEntry]:
        if self._collection is None:
            await self.initialize()

        query_embedding = self._embed(query)

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"],
            ),
        )

        entries = []
        if results["ids"] and results["ids"][0]:
            for i, entry_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                entries.append(MemoryEntry(
                    id=entry_id,
                    content=results["documents"][0][i],
                    metadata={k: v for k, v in metadata.items()
                             if k not in ("timestamp", "summary")},
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                    summary=metadata.get("summary"),
                ))
        return entries

    async def get(self, entry_id: str) -> MemoryEntry | None:
        if self._collection is None:
            await self.initialize()

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._collection.get(ids=[entry_id], include=["documents", "metadatas"]),
        )

        if results["ids"]:
            metadata = results["metadatas"][0]
            return MemoryEntry(
                id=entry_id,
                content=results["documents"][0],
                metadata={k: v for k, v in metadata.items()
                         if k not in ("timestamp", "summary")},
                timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat())),
                summary=metadata.get("summary"),
            )
        return None

    async def delete(self, entry_id: str) -> bool:
        if self._collection is None:
            return False
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._collection.delete(ids=[entry_id]),
            )
            return True
        except Exception:
            return False

    async def clear(self) -> None:
        if self._collection is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self._client.delete_collection("jarvis_memory"),
            )
            self._collection = self._client.get_or_create_collection(
                name="jarvis_memory",
                metadata={"hnsw:space": "cosine"},
            )


class InMemoryStore(MemoryStore):
    """Fallback in-memory store for testing."""
    def __init__(self) -> None:
        self._data: dict[str, MemoryEntry] = {}

    async def add(self, entry: MemoryEntry) -> None:
        self._data[entry.id] = entry

    async def search(self, query: str, limit: int = 10, filter_metadata: dict | None = None) -> list[MemoryEntry]:
        # Simple text search fallback
        results = []
        query_lower = query.lower()
        for entry in self._data.values():
            if query_lower in entry.content.lower():
                results.append(entry)
                if len(results) >= limit:
                    break
        return results

    async def get(self, entry_id: str) -> MemoryEntry | None:
        return self._data.get(entry_id)

    async def delete(self, entry_id: str) -> bool:
        if entry_id in self._data:
            del self._data[entry_id]
            return True
        return False

    async def clear(self) -> None:
        self._data.clear()


class MemoryManager:
    """Manages STM (short-term) and LTM (long-term) memory."""

    def __init__(self) -> None:
        self.config = get_agent_config()
        self.ltm: MemoryStore = ChromaMemoryStore() if self.config.memory.enabled else InMemoryStore()
        self.stm: list[MemoryEntry] = []
        self._stm_tokens = 0

    async def initialize(self) -> None:
        if hasattr(self.ltm, 'initialize'):
            await self.ltm.initialize()

    def add_stm(self, content: str, metadata: dict | None = None, summary: str | None = None) -> MemoryEntry:
        """Add to short-term memory."""
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            metadata=metadata or {},
            summary=summary,
        )
        self.stm.append(entry)
        self._stm_tokens += len(content) // 4  # rough token estimate

        # Auto-move to LTM if STM too large
        if self._stm_tokens > self.config.memory.stm_max_tokens:
            self._consolidate_stm()

        return entry

    def _consolidate_stm(self) -> None:
        """Move older STM entries to LTM."""
        if not self.stm:
            return

        # Keep last 20% in STM, move rest to LTM
        keep = max(1, len(self.stm) // 5)
        to_move = self.stm[:-keep]
        self.stm = self.stm[-keep:]

        for entry in to_move:
            asyncio.create_task(self.ltm.add(entry))

        self._stm_tokens = sum(len(e.content) // 4 for e in self.stm)
        logger.debug(f"Consolidated {len(to_move)} STM entries to LTM")

    async def search_ltm(self, query: str, limit: int = 10, filter_metadata: dict | None = None) -> list[MemoryEntry]:
        return await self.ltm.search(query, limit, filter_metadata)

    async def get_context(self, query: str, max_tokens: int = 3000) -> str:
        """Get relevant context from both STM and LTM.

        Every entry is truncated individually: joining ten full STM records
        easily reaches thousands of tokens, and on local CPU hardware prompt
        evaluation dominates latency (~38 tok/s measured).
        """
        entry_chars = max(120, (max_tokens * 4) // 8)

        context_parts: list[str] = []
        for entry in await self.search_ltm(query, limit=4):
            context_parts.append(f"[LTM] {entry.content[:entry_chars]}")
        for entry in self.stm[-8:]:
            context_parts.append(f"[STM] {entry.content[:entry_chars]}")

        context = "\n".join(p for p in context_parts if p)
        limit = max_tokens * 4
        if len(context) > limit:
            context = context[:limit] + "\n...[truncated]"
        return context

    def get_stm_summary(self) -> str:
        if not self.stm:
            return "STM: (empty)"
        return "STM:\n" + "\n".join(f"  - {e.content[:100]}..." for e in self.stm[-5:])

    async def shutdown(self) -> None:
        """Move all STM to LTM on shutdown."""
        for entry in self.stm:
            await self.ltm.add(entry)
        self.stm.clear()
        self._stm_tokens = 0


# Global instance
_memory_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager