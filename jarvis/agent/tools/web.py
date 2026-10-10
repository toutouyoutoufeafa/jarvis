"""Web tools for agents (search, fetch, scrape)."""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus, urljoin

import aiohttp
from bs4 import BeautifulSoup

from jarvis.agent.tools import tool
from jarvis.config import get_config


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


@tool("web_search", "Search the web")
async def web_search(query: str, max_results: int = 10, engine: str = "duckduckgo") -> list[SearchResult]:
    """Search the web using DuckDuckGo HTML (no API key needed)."""
    if engine != "duckduckgo":
        raise ValueError(f"Only duckduckgo supported, got: {engine}")

    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as resp:
            html = await resp.text()

    soup = BeautifulSoup(html, "html.parser")
    results = []

    for result in soup.select(".result__snippet"):
        link = result.find_previous("a", class_="result__url")
        title_elem = result.find_previous("a", class_="result__snippet")

        title = title_elem.get_text(strip=True) if title_elem else ""
        snippet = result.get_text(strip=True)
        url = link.get("href") if link else ""

        if url and not url.startswith("http"):
            url = "https://duckduckgo.com" + url

        results.append(SearchResult(title=title, url=url, snippet=snippet))
        if len(results) >= max_results:
            break

    return results


@tool("web_fetch", "Fetch and extract content from URL")
async def web_fetch(url: str, max_length: int = 50000) -> dict:
    """Fetch webpage and extract main content."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            html = await resp.text()
            content_type = resp.headers.get("Content-Type", "")

    if "text/html" not in content_type:
        return {
            "url": url,
            "content": html[:max_length],
            "content_type": content_type,
            "title": "",
        }

    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, nav, footer
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    # Try to find main content
    main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile(r"(content|main|article)"))
    if main:
        text = main.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text[:max_length]

    title = soup.title.string if soup.title else ""

    return {
        "url": url,
        "title": title.strip(),
        "content": text,
        "content_type": "text/html",
        "length": len(text),
    }


@tool("web_download", "Download file from URL")
async def web_download(url: str, dest_path: str, timeout: int = 60) -> str:
    """Download file from URL to local path."""
    from jarvis.agent.tools.fs import _resolve_path

    target = _resolve_path(dest_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            content = await resp.read()

    target.write_bytes(content)
    return f"Downloaded {len(content)} bytes to {target}"


def register_web_tools() -> None:
    pass