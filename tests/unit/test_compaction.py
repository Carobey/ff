"""Unit tests for the thread-compaction node (ADR 0008)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from family_finance.agents import compaction as compaction_module
from family_finance.agents.compaction import compact_node


@pytest.mark.unit
async def test_compact_noop_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Below the message threshold the node is a no-op and never calls the LLM."""

    def fail_get_chat_model(*_a: object, **_k: object) -> object:
        raise AssertionError("LLM must not be called below threshold")

    monkeypatch.setattr("family_finance.agents.compaction.get_chat_model", fail_get_chat_model)

    messages = [HumanMessage(content=f"m{i}") for i in range(5)]
    result = await compact_node({"messages": messages})

    assert result == {}


@pytest.mark.unit
async def test_compact_summarizes_and_keeps_recent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Above threshold: clear all, prepend one summary, keep the recent tail."""

    class _Resp:
        content = "сводка"

    class _Model:
        async def ainvoke(self, *_a: object, **_k: object) -> _Resp:
            return _Resp()

    monkeypatch.setattr("family_finance.agents.compaction.get_chat_model", lambda *a, **k: _Model())

    keep = compaction_module._KEEP_RECENT
    total = compaction_module._COMPACT_AFTER + 5
    messages = [HumanMessage(content=f"m{i}") for i in range(total)]

    result = await compact_node({"messages": messages})
    out = result["messages"]
    assert isinstance(out, list)

    # First entry wipes the list, then one summary, then exactly the recent tail.
    assert isinstance(out[0], RemoveMessage)
    assert out[0].id == REMOVE_ALL_MESSAGES
    assert "сводка" in str(out[1].content)
    kept = out[2:]
    assert len(kept) == keep
    assert [str(m.content) for m in kept] == [f"m{i}" for i in range(total - keep, total)]


@pytest.mark.unit
async def test_compact_keeps_tool_pair_intact(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tail must not start with an orphaned ToolMessage (PR-10 guard).

    If the -_KEEP_RECENT boundary lands on a ToolMessage, its initiating
    AIMessage(tool_calls) would be summarized away, leaving a tool-result with
    no preceding tool-call → provider rejects the next turn. The split must move
    back so the pair stays together in the kept tail.
    """

    class _Resp:
        content = "сводка"

    class _Model:
        async def ainvoke(self, *_a: object, **_k: object) -> _Resp:
            return _Resp()

    monkeypatch.setattr("family_finance.agents.compaction.get_chat_model", lambda *a, **k: _Model())

    keep = compaction_module._KEEP_RECENT
    total = compaction_module._COMPACT_AFTER + 5
    boundary = total - keep  # index that would become recent[0]

    messages: list[object] = [HumanMessage(content=f"m{i}") for i in range(total)]
    # Put a tool-call/result pair straddling the boundary: AIMessage(tool_calls)
    # just before it, the ToolMessage exactly on it.
    messages[boundary - 1] = AIMessage(
        content="",
        tool_calls=[{"name": "spending_breakdown", "args": {}, "id": "call_1"}],
    )
    messages[boundary] = ToolMessage(content="результат", tool_call_id="call_1")

    result = await compact_node({"messages": messages})
    kept = result["messages"][2:]  # after RemoveMessage + summary

    # Guard pulled the AIMessage(tool_calls) into the tail, so it leads — not the
    # orphaned ToolMessage — and the pair is contiguous.
    assert isinstance(kept[0], AIMessage)
    assert kept[0].tool_calls
    assert isinstance(kept[1], ToolMessage)
    assert len(kept) == keep + 1


@pytest.mark.unit
async def test_compact_noop_on_llm_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient summarizer error must not break the turn — fall back to no-op.

    compact_node runs on the hot path of every message once a thread is long, so a
    failing LLM call has to degrade to «не сворачиваем в этот раз», not crash.
    """

    class _Model:
        async def ainvoke(self, *_a: object, **_k: object) -> object:
            raise RuntimeError("summarizer down")

    monkeypatch.setattr("family_finance.agents.compaction.get_chat_model", lambda *a, **k: _Model())

    total = compaction_module._COMPACT_AFTER + 5
    messages = [HumanMessage(content=f"m{i}") for i in range(total)]

    result = await compact_node({"messages": messages})

    # No-op: thread left intact (no RemoveMessage), supervisor proceeds normally.
    assert result == {}


@pytest.mark.unit
async def test_compact_renders_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    """The summarizer receives a role-tagged transcript of the older messages."""

    captured: dict[str, str] = {}

    class _Resp:
        content = "ok"

    class _Model:
        async def ainvoke(self, msgs: list[object], *_a: object, **_k: object) -> _Resp:
            captured["transcript"] = str(msgs[1].content)  # type: ignore[attr-defined]
            return _Resp()

    monkeypatch.setattr("family_finance.agents.compaction.get_chat_model", lambda *a, **k: _Model())

    total = compaction_module._COMPACT_AFTER + 5
    messages: list[object] = []
    for i in range(total):
        messages.append(HumanMessage(content=f"вопрос{i}"))
        messages.append(AIMessage(content=f"ответ{i}"))

    await compact_node({"messages": messages})

    assert "Пользователь: вопрос0" in captured["transcript"]
    assert "Помощник: ответ0" in captured["transcript"]
