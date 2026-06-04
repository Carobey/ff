-- A single savings goal per family (pay-yourself-first).
--
-- The advisor agent measures net savings (income − expenses) since
-- created_at and reports progress toward target_amount. target_date is
-- optional: with it we can pace ("откладывай X/мес"), without it the goal
-- is open-ended.
--
-- One row per family — re-running the migration is safe.

CREATE TABLE IF NOT EXISTS savings_goal (
    goal_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    family_id        UUID NOT NULL REFERENCES family(family_id) ON DELETE CASCADE,
    target_amount    NUMERIC(14, 2) NOT NULL CHECK (target_amount > 0),
    target_date      DATE,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (family_id)
);

CREATE INDEX IF NOT EXISTS idx_savings_goal_family ON savings_goal(family_id);
