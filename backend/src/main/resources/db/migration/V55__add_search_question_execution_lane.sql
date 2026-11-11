ALTER TABLE search_answer_executions
    ADD COLUMN question_only BOOLEAN NOT NULL DEFAULT FALSE,
    ADD CONSTRAINT ck_search_answer_executions_exclusive_lane CHECK (NOT (lookup_only AND question_only));

CREATE INDEX idx_search_answer_executions_queued_generation_lane
    ON search_answer_executions (question_only, created_at, answer_id)
    WHERE state = 'QUEUED' AND NOT lookup_only;
