"""Family и FamilyMember сущности."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from family_finance.domain.types import FamilyRole, WalletPrivacy


class Family(BaseModel):
    model_config = ConfigDict(extra="forbid")

    family_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class FamilyMember(BaseModel):
    """Член семьи. Telegram user_id → wallet mapping."""

    model_config = ConfigDict(extra="forbid")

    member_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    family_id: uuid.UUID
    name: str
    role: FamilyRole = FamilyRole.PARENT
    telegram_user_id: int
    privacy: WalletPrivacy = WalletPrivacy.PRIVATE
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
