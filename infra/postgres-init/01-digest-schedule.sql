-- Weekly-digest delivery schedule per family member.
--
-- Stored as a 5-field cron expression (minute hour dom month dow) so the
-- scheduler can pass it straight to APScheduler's CronTrigger. NULL means
-- the user hasn't opted in — no digest is sent.
--
-- Idempotent: safe to re-run.

ALTER TABLE family_member
    ADD COLUMN IF NOT EXISTS digest_cron TEXT;
