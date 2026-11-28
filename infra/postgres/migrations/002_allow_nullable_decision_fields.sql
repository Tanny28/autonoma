
-- Allow audit records for decisions without a diagnosis,
-- confidence score, or remediation action.

ALTER TABLE decision_log
    ALTER COLUMN diagnosed_cause DROP NOT NULL,
    ALTER COLUMN confidence DROP NOT NULL,
    ALTER COLUMN action DROP NOT NULL;
