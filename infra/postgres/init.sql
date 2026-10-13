
-- AUTONOMA PostgreSQL initialization
-- Creates application databases on first container start.
-- The default POSTGRES_DB (autonoma) is created automatically by the postgres image.

CREATE DATABASE mlflow;
CREATE DATABASE autonoma_app;

-- Decision log for recording diagnostic and remediation decisions.
-- This table is created in the default "autonoma" database.

CREATE TABLE IF NOT EXISTS decision_log (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    episode TEXT NOT NULL,
    diagnosed_cause TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    action TEXT NOT NULL,
    justification TEXT NOT NULL,
    outcome TEXT
);

-- Support chronological queries and episode-based lookups.
CREATE INDEX IF NOT EXISTS idx_decision_log_timestamp
    ON decision_log (timestamp);

CREATE INDEX IF NOT EXISTS idx_decision_log_episode
    ON decision_log (episode);
