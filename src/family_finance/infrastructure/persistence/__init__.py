"""Persistence adapters."""

from family_finance.infrastructure.persistence.postgres_transactions import (
    PostgresTransactionRepository,
)

__all__ = ["PostgresTransactionRepository"]
