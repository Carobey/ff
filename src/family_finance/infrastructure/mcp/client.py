"""Load the family-finance MCP server's tools as LangChain tools.

LangGraph nodes (e.g. ``ledger``) read ledger data *through* these MCP tools
instead of calling the repository directly — that is what makes the project a
real MCP **consumer**, not just a producer. The server is spawned over stdio
using the current interpreter; tools are cached after the first load so we
don't re-introspect on every query.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_NAME = "family-finance"

_client: MultiServerMCPClient | None = None
_tools: dict[str, BaseTool] | None = None


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


async def get_finance_tools() -> dict[str, BaseTool]:
    """Return the finance MCP tools keyed by name, loading them once."""
    global _tools
    if _tools is None:
        loaded = await _get_client().get_tools()
        _tools = {tool.name: tool for tool in loaded}
    return _tools


async def call_finance_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Invoke a finance MCP tool and decode its JSON payload.

    ``langchain-mcp-adapters`` surfaces tool output as text content blocks; we
    decode the JSON the FastMCP tool produced back into Python objects.
    """
    tools = await get_finance_tools()
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
