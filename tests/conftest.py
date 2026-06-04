"""Pytest configuration."""

import pytest


@pytest.fixture(autouse=True)
def _no_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """В юнит-тестах не пытаемся достучаться до LangFuse."""
    monkeypatch.setenv("LANGFUSE_HOST", "http://disabled:9999")
