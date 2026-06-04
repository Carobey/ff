"""Application layer — Protocol-порты для границы с infrastructure.

Сценарии (categorize, ingest, report) реализованы как ноды LangGraph в `agents/`,
а не отдельными use-case модулями. В этом пакете живут только порты (`ports.py`).
"""

from __future__ import annotations
