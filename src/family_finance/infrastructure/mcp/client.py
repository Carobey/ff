"""Load the family-finance MCP server's tools as LangChain tools.

LangGraph nodes (e.g. ``ledger``) read ledger data *through* these MCP tools
instead of calling the repository directly — that is what makes the project a
real MCP **consumer**, not just a producer.

The server runs over stdio using the current interpreter. We keep ONE long-lived
stdio session for the whole bot lifetime and bind the tools to it. Using
``client.get_tools()`` instead would open a fresh session — i.e. spawn a new
MCP-server subprocess (re-importing presidio etc.) — on *every* tool call.
Call :func:`close_finance_tools` on shutdown to tear the subprocess down.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

_SERVER_NAME = "family-finance"

_client: MultiServerMCPClient | None = None
_stack: AsyncExitStack | None = None
_tools: dict[str, BaseTool] | None = None

# One persistent stdio session над одним subprocess НЕ выдерживает конкурентного
# доступа: параллельные ноды (веер ``Send`` оркестратора, ADR 0008) одновременно
# пишут в общий stdio-стрим → anyio.BrokenResourceError / «exit cancel scope in a
# different task». Сериализуем и ленивую инициализацию, и каждый вызов через один
# lock — subprocess всё равно обрабатывает запросы по одному, так что параллелизм
# тут мнимый, а корректность — реальная.
_lock = asyncio.Lock()


def _get_client() -> MultiServerMCPClient:
    global _client
    if _client is None:
        _client = MultiServerMCPClient(
            {
                _SERVER_NAME: {
                    "transport": "stdio",
                    "command": sys.executable,
                    "args": ["-m", "family_finance.mcp_server.server"],
                }
            }
        )
    return _client


async def _ensure_tools() -> dict[str, BaseTool]:
    """Lazily start the persistent session. Caller must hold ``_lock``."""
    global _tools, _stack
    if _tools is None:
        _stack = AsyncExitStack()
        session = await _stack.enter_async_context(_get_client().session(_SERVER_NAME))
        loaded = await load_mcp_tools(session)
        _tools = {tool.name: tool for tool in loaded}
    return _tools


async def get_finance_tools() -> dict[str, BaseTool]:
    """Return finance MCP tools keyed by name, bound to one persistent session.

    The stdio subprocess starts on first use and stays up until
    :func:`close_finance_tools`, so subsequent tool calls reuse it.
    """
    async with _lock:
        return await _ensure_tools()


async def close_finance_tools() -> None:
    """Tear down the persistent MCP session/subprocess (call on bot shutdown)."""
    global _tools, _stack
    if _stack is not None:
        await _stack.aclose()
    _stack = None
    _tools = None


async def call_finance_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Invoke a finance MCP tool and decode its JSON payload.

    ``langchain-mcp-adapters`` surfaces tool output as text content blocks; we
    decode the JSON the FastMCP tool produced back into Python objects.

    The whole init+invoke is serialized on ``_lock`` so parallel orchestrator
    workers queue on the single stdio session instead of corrupting it.
    """
    async with _lock:
        tools = await _ensure_tools()
        raw = await tools[name].ainvoke(arguments)
    return _decode_payload(raw)


def _decode_payload(raw: Any) -> Any:
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        for block in raw:
            if isinstance(block, dict) and block.get("type") == "text":
                return json.loads(block["text"])
        # No text content block: FastMCP passed structured output straight
        # through (an empty list = "no results", or a list of plain dicts).
        return raw
    if isinstance(raw, dict):
        return raw
    raise ValueError(f"Unexpected MCP tool result: {raw!r}")
