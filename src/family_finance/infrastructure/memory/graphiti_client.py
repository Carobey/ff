"""Graphiti episodic memory client (Phase 2).

Graphiti builds a knowledge graph in FalkorDB from text episodes.
We use it for behavioural CoachAgent queries like:
  "Когда я последний раз так часто заказывал доставку?"

Architecture:
  Transaction saved in Postgres
      ↓  (fire-and-forget, не блокирует основной путь)
  add_episode("Юри купил Молоко в Пятёрочке 29.05.2026 за 523 ₽")
      ↓
  Graphiti extracts entities/edges into FalkorDB graph
      ↓
  CoachAgent: search("частая доставка") → related edges → LLM narrative

Why we don't go through ``get_chat_model`` here (per ADR 0004):
Graphiti's ``LLMClient`` Protocol has a non-trivial interface — replacing it
with a custom langchain-openrouter wrapper would re-implement entity extraction,
attribute-preamble injection, retries and token tracking. Instead we use
Graphiti's stock ``OpenAIClient`` but pass it ``langfuse.openai.AsyncOpenAI``
so every Graphiti LLM/embedding call appears in LangFuse traces. That gives
us the main observability win (cost + latency) without forking Graphiti.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import structlog
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder import OpenAIEmbedder
from graphiti_core.embedder.openai import OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType
from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI  # type: ignore[attr-defined]

from family_finance.infrastructure.settings import get_settings

logger = structlog.get_logger()

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# Graphiti makes many small LLM calls for entity extraction; use worker model
_GRAPHITI_LLM_MODEL = "google/gemini-2.5-flash"
# OpenRouter supports text-embedding-3-small
_GRAPHITI_EMBED_MODEL = "openai/text-embedding-3-small"
_EMBED_DIM = 1536


@lru_cache(maxsize=1)
def _build_graphiti() -> Graphiti:
    """Build and cache a Graphiti instance connected to FalkorDB + OpenRouter.

    NOTE: the cached instance holds an async client with no explicit close path —
    it lives for the whole process. Acceptable for a long-running single-family
    bot; if this ever becomes a short-lived/multi-tenant process, add a
    ``close()`` lifecycle instead of ``lru_cache``.


    The underlying OpenAI client is ``langfuse.openai.AsyncOpenAI``, a drop-in
    instrumented replacement: same wire protocol, but every call emits a
    LangFuse generation span (cost, latency, prompt, completion). One shared
    client is reused for LLM, embedder and reranker so they all share the
    HTTP connection pool.
    """
    s = get_settings()
    api_key = s.openrouter_api_key.get_secret_value()

    instrumented_client = LangfuseAsyncOpenAI(
        api_key=api_key,
        base_url=_OPENROUTER_BASE_URL,
    )

    llm_client = OpenAIClient(
        config=LLMConfig(
            api_key=api_key,
            model=_GRAPHITI_LLM_MODEL,
            base_url=_OPENROUTER_BASE_URL,
        ),
        client=instrumented_client,
    )

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=api_key,
            base_url=_OPENROUTER_BASE_URL,
            embedding_model=_GRAPHITI_EMBED_MODEL,
            embedding_dim=_EMBED_DIM,
        ),
        client=instrumented_client,
    )

    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(
            api_key=api_key,
            model=_GRAPHITI_LLM_MODEL,
            base_url=_OPENROUTER_BASE_URL,
        ),
        client=instrumented_client,
    )

    driver = FalkorDriver(host=s.falkordb_host, port=s.falkordb_port)
    logger.info(
        "graphiti.driver_ready",
        host=s.falkordb_host,
        port=s.falkordb_port,
    )

    return Graphiti(
        graph_driver=driver,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )


async def graphiti_init() -> None:
    """Create indices on first startup. Call once at bot startup."""
    g = _build_graphiti()
    await g.build_indices_and_constraints()
    logger.info("✅ Graphiti indices ready")


async def add_episode(
    *,
    name: str,
    body: str,
    source_description: str,
    reference_time: datetime,
    group_id: str,
) -> None:
    """Add one text episode to the knowledge graph.

    Fire-and-forget: errors are logged, not re-raised — we don't want
    episodic memory failures to break the main transaction flow.
    """
    g = _build_graphiti()
    try:
        await g.add_episode(
            name=name,
            episode_body=body,
            source_description=source_description,
            reference_time=reference_time,
            source=EpisodeType.text,
            group_id=group_id,
        )
        logger.debug("graphiti: episode added", name=name, group_id=group_id)
    except Exception:
        logger.exception("graphiti: add_episode failed, continuing without memory", name=name)


async def search_episodes(
    *,
    query: str,
    group_id: str,
    num_results: int = 10,
) -> list[object]:
    """Search the knowledge graph for edges related to *query*.

    Returns list of EntityEdge objects (graphiti_core.edges.EntityEdge).
    Returns [] on any error.
    """
    g = _build_graphiti()
    try:
        results = await g.search(
            query=query,
            group_ids=[group_id],
            num_results=num_results,
        )
        return results  # type: ignore[return-value]
    except Exception:
        logger.exception("graphiti: search failed", query=query)
        return []
