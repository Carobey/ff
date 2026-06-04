"""Helpers for working with LangChain ``BaseMessage`` content.

In LangChain >=0.3 ``message.content`` can be either ``str`` or
``list[dict|str]`` (content blocks — multimodal, tool calls, citations).
Casting to ``str`` silently lies in the block case (you get
``"[{'type': 'text', ...}]"``), so every node that wants plain text uses
``message_text`` to collapse blocks into a single string.
"""

from __future__ import annotations

from langchain_core.messages import BaseMessage


def message_text(message: BaseMessage) -> str:
    """Return ``message.content`` as a flat string.

    * ``str`` → returned unchanged.
    * ``list`` → concatenate all text blocks (``{"type": "text", "text": ...}``
      and bare strings); non-text blocks (images, tool calls) are skipped.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)
