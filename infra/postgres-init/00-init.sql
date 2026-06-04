-- Family Finance — initial schema
-- LangGraph checkpointer таблицы создаст AsyncPostgresSaver.setup() при первом запуске
-- Здесь — то что нам нужно ДО старта приложения

-- pgvector понадобится в Phase 2 для semantic memory
CREATE EXTENSION IF NOT EXISTS vector;

-- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;


-- === Domain tables (заготовка, заполнится из приложения) ===

CREATE TABLE IF NOT EXISTS family (
    family_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS family_member (
    member_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    family_id           UUID NOT NULL REFERENCES family(family_id) ON DELETE CASCADE,
    name                TEXT NOT NULL,
    role                TEXT NOT NULL DEFAULT 'parent',
    telegram_user_id    BIGINT UNIQUE,
    privacy             TEXT NOT NULL DEFAULT 'private',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_member_family ON family_member(family_id);
CREATE INDEX IF NOT EXISTS idx_member_tg ON family_member(telegram_user_id);


CREATE TABLE IF NOT EXISTS transaction (
    transaction_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    family_id           UUID NOT NULL REFERENCES family(family_id) ON DELETE CASCADE,
    member_id           UUID NOT NULL REFERENCES family_member(member_id) ON DELETE CASCADE,

    -- Битемпоральность
    occurred_at         TIMESTAMPTZ NOT NULL,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Деньги (Decimal, не float!)
    amount              NUMERIC(20, 4) NOT NULL CHECK (amount > 0),
    currency            TEXT NOT NULL DEFAULT 'RUB',
    direction           TEXT NOT NULL CHECK (direction IN ('expense','income','transfer','refund')),

    -- Что/где
    merchant_raw        TEXT NOT NULL,
    merchant_normalized TEXT,

    -- Категория
    category            TEXT NOT NULL DEFAULT 'unclassified',
    subcategory_freetext TEXT,
    confidence          REAL NOT NULL DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    needs_review        BOOLEAN NOT NULL DEFAULT FALSE,

    -- Источник
    source              TEXT NOT NULL,
    source_file         TEXT,

    -- Связь с чеком
    receipt_id          UUID,
    receipt_fns_qr      TEXT,

    -- Доп
    tags                TEXT[] DEFAULT ARRAY[]::TEXT[],

    -- Идемпотентность импорта: один и тот же CSV строка не создаст две транзакции
    import_hash         TEXT UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_tx_family_occurred ON transaction(family_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_tx_member ON transaction(member_id);
CREATE INDEX IF NOT EXISTS idx_tx_category ON transaction(category);
CREATE INDEX IF NOT EXISTS idx_tx_needs_review ON transaction(family_id) WHERE needs_review = TRUE;


CREATE TABLE IF NOT EXISTS receipt (
    receipt_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    family_id           UUID NOT NULL REFERENCES family(family_id) ON DELETE CASCADE,
    member_id           UUID NOT NULL REFERENCES family_member(member_id) ON DELETE CASCADE,
    qr_raw              TEXT NOT NULL,
    fiscal_drive        TEXT,
    fiscal_document     TEXT,
    fiscal_sign         TEXT,
    total_amount        NUMERIC(20, 4) NOT NULL,
    purchase_time       TIMESTAMPTZ NOT NULL,
    store_name          TEXT,
    items_json          JSONB,
    raw_response        JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_receipt_family ON receipt(family_id);
CREATE INDEX IF NOT EXISTS idx_receipt_time ON receipt(purchase_time DESC);
