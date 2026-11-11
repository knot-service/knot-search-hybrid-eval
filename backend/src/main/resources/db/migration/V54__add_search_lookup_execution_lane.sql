ALTER TABLE search_answer_executions
    ADD COLUMN lookup_only BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX idx_search_answer_executions_queued_lane
    ON search_answer_executions (lookup_only, created_at, answer_id)
    WHERE state = 'QUEUED';
