"""Postgres adapters for the categorization cascade.

* ``PostgresCategoryCatalog``        — рендерит таксономию (справочник) для промпта.
* ``PostgresMerchantRuleRepository`` — fuzzy-поиск правил «продавец → категория»
  (pg_trgm ``word_similarity``) + дозапись выученных правил (learning loop).

Деньги тут не пересекаются — только текст и коды категорий.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import asyncpg

from family_finance.application.ports import MerchantRuleHit
from family_finance.domain import Category, normalize_merchant
from family_finance.infrastructure.persistence.postgres_transactions import _get_pool
from family_finance.infrastructure.settings import get_settings


class PostgresCategoryCatalog:
    """Read-only справочник категорий из таблицы ``category``."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or get_settings().database_url.get_secret_value()

    async def render_taxonomy(self) -> str:
        """Собрать блок «КАТЕГОРИИ» для system-промпта из активных категорий."""
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT code, description
                FROM category
                WHERE active = TRUE
                ORDER BY sort_order, code
                """
            )
        return "\n".join(f"{row['code']:<22} — {row['description']}" for row in rows)


class PostgresMerchantRuleRepository:
    """fuzzy-каскад «продавец → категория» поверх ``merchant_category_rule``."""

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or get_settings().database_url.get_secret_value()

    async def lookup_many(
        self,
        *,
        family_id: uuid.UUID,
        merchants: Sequence[str],
        threshold: float,
    ) -> dict[str, MerchantRuleHit]:
        """Сопоставить продавцов правилам; ключ результата — исходный merchant_raw."""
        # Уникальные нормализованные формы → один запрос на форму, переиспользуем
        # результат для всех исходных строк с тем же нормализованным видом.
        norm_by_raw: dict[str, str] = {m: normalize_merchant(m) for m in merchants}
        unique_norms = {norm for norm in norm_by_raw.values() if norm}

        pool = await _get_pool(self._dsn)
        hits_by_norm: dict[str, MerchantRuleHit] = {}
        async with pool.acquire() as conn:
            for norm in unique_norms:
                hit = await self._best_match(conn, family_id=family_id, norm=norm)
                if hit is not None and hit.score >= threshold:
                    hits_by_norm[norm] = hit

        return {
            raw: hits_by_norm[norm] for raw, norm in norm_by_raw.items() if norm in hits_by_norm
        }

    @staticmethod
    async def _best_match(
        conn: asyncpg.Connection,
        *,
        family_id: uuid.UUID,
        norm: str,
    ) -> MerchantRuleHit | None:
        row = await conn.fetchrow(
            """
            SELECT category_code,
                   source,
                   word_similarity(merchant_norm, $1) AS score
            FROM merchant_category_rule
            WHERE family_id IS NULL OR family_id = $2
            ORDER BY word_similarity(merchant_norm, $1) DESC,
                     (family_id = $2) DESC NULLS LAST
            LIMIT 1
            """,
            norm,
            family_id,
        )
        if row is None:
            return None
        try:
            category = Category(row["category_code"])
        except ValueError:
            return None
        return MerchantRuleHit(
            category=category,
            score=float(row["score"]),
            source=row["source"],
        )

    async def upsert(
        self,
        *,
        family_id: uuid.UUID,
        merchant_raw: str,
        category: Category,
        source: str = "user",
    ) -> None:
        """Записать/обновить выученное правило семьи (learning loop)."""
        norm = normalize_merchant(merchant_raw)
        if not norm:
            return
        pool = await _get_pool(self._dsn)
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO merchant_category_rule
                    (family_id, merchant_norm, merchant_sample, category_code, source)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (family_id, merchant_norm) DO UPDATE
                SET category_code   = EXCLUDED.category_code,
                    merchant_sample = EXCLUDED.merchant_sample,
                    source          = EXCLUDED.source,
                    hit_count       = merchant_category_rule.hit_count + 1,
                    updated_at      = NOW()
                """,
                family_id,
                norm,
                merchant_raw,
                category.value,
                source,
            )
