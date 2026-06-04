-- Monthly per-category budgets per family.
--
-- A family sets a soft limit per Category enum value; the categorizer
-- compares the running month's spend against the limit and posts alerts
-- back to the chat when 80% / 100% thresholds are crossed.
--
-- One row per (family_id, category) — re-running the migration is safe.

CREATE TABLE IF NOT EXISTS budget (
    budget_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    family_id        UUID NOT NULL REFERENCES family(family_id) ON DELETE CASCADE,
    category         TEXT NOT NULL,
    monthly_limit    NUMERIC(14, 2) NOT NULL CHECK (monthly_limit > 0),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (family_id, category)
);

CREATE INDEX IF NOT EXISTS idx_budget_family ON budget(family_id);
